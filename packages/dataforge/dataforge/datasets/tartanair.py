"""TartanAir V2: one trajectory, three camera sets on the same motion, with exact ground truth and a noise-free IMU.

``download`` range-extracts the chosen trajectories from Hugging Face (``tartanair_download``); ``convert`` writes,
per trajectory, one recording for the ``--camera-set`` it is given, so the same motion runs with each camera model:

* ``pinhole``: ``lcam_front`` + ``rcam_front``, 640x640, f = 320 px, 0.25 m baseline (``brown_conrady``, all zero).
* ``fisheye``: ``lcam_fish`` + ``rcam_fish``, 1000x1000 equidistant over 195 deg: ``kannala_brandt`` with
  k1..k4 = 0 and fx = fy = W / FOV px per radian (TartanAir's ``LinearSphere``, r = theta * W / FOV).
* ``equirect``: ``lcam_equirect``, 2048x1024 full sphere, as ``equirectangular``: longitude = (u - cx) / fx,
  latitude = (v - cy) / fy, bearing (cos lat sin lon, sin lat, cos lat cos lon) in the camera's RDF frame, with
  fx = W / 2pi, fy = H / pi and one coefficient, lon_max = pi. The pixels are stored as the dataset ships them; the
  dataset's panorama centre looks LEFT and its front is at u = 3W/4, so the camera's ``rig_T_cam`` turns the camera
  -90 deg about rig +y (its optical axis is rig -x).

Pixel centres follow OpenCV: pixel index i covers [i - 0.5, i + 0.5], so the image centre is (W/2 - 0.5, H/2 - 0.5)
(the half-pixel offset was checked on pixels by matching patches between lcam_front, lcam_fish and lcam_equirect).

Axes. TartanAir states poses and the IMU in NED: world x north, y east, z down; body x forward, y right, z down, the
body being ``lcam_front``. Here:

* The **rig frame** is lcam_front's RDF optical frame (x right, y down, z forward), p_ned = R_NED_RDF p_rdf. The IMU is
  the rig's reference (``imu_00``, identity); every camera's ``rig_T_cam`` is a fixed rotation and the stereo offset.
* The **world** is TartanAir's NED world turned Z-up (x north, y west, z up): p_world = ZUP_R_NED p_ned.
* So world_T_rig = ZUP_R_NED . world_T_body_ned . R_NED_RDF, and gyro / accel_rig = R_NED_RDF^T gyro / accel_ned.
  The accelerometer is specific force (g = 9.8 m/s^2): at rest it reads 9.8 m/s^2 along rig up.

Clock: frame k is at k * 0.1 s and IMU sample j at j * 0.01 s on ``video_time``, from 0 (``capture:start_time_ns`` = 0).
The IMU is synthesised from a pose spline whose ends swing (|accel| up to 94 m/s^2), so ``IMU_TRIM`` samples are dropped
at each end and only frames inside the remaining IMU span are kept. Ground truth is the pose file at the kept frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

import httpx
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.imu_calibration import ImuCalibration
from simplecv.rerun_custom_types import CameraDistortion, PinholeWithDistortion

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
from dataforge.datasets.tartanair_download import HF_BASE_URL, HF_REVISION, extract_members, http_range
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import GtTrajectory, ImuChannel, log_camera_source, log_gt_path, log_imu, log_rig_node, log_video_stream
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4, work_dir
from dataforge.world_up import WORLD_UP_VIEW_COORDINATES

CameraSet: TypeAlias = Literal["pinhole", "fisheye", "equirect"]
"""``--camera-set``: which cameras one recording carries; also the catalog dataset (``tartanair-<set>``)."""
RIG: int = 0
"""One rig per trajectory."""
FRAME_PERIOD_NS: int = 100_000_000
"""Images are 10 Hz."""
IMU_PERIOD_NS: int = 10_000_000
"""The IMU is 100 Hz."""
IMU_TRIM: int = 10
"""IMU samples dropped at each end: the spline's end transient."""
GRAVITY_MS2: float = 9.8
"""The gravity TartanAir synthesises its accelerometer with."""
VIDEO_CQ: int = 24
"""AV1 NVENC quality, the same for every set: higher quality than the port-wide 36, since the frames feed a tracker."""
VIDEO_GOP: int = 10
"""Keyframe interval in frames: one second at 10 Hz."""
R_NED_RDF: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
"""p_ned = R_NED_RDF p_rdf: RDF right = NED y, down = NED z, forward = NED x."""
ZUP_R_NED: Float64[ndarray, "3 3"] = np.diag([1.0, -1.0, -1.0])
"""p_world = ZUP_R_NED p_ned: north stays x, east becomes -y, down becomes -z."""
DEFAULT_TRAJECTORIES: tuple[str, ...] = (
    "ArchVizTinyHouseDay/easy/P006",
    "ArchVizTinyHouseDay/hard/P003",
    "ArchVizTinyHouseDay/easy/P003",
    "OldTownSummer/easy/P000",
    "OldTownFall/hard/P001",
)
"""The subset (recon 05 section 4): two ArchViz smoke trajectories, a 15 s ArchViz one for VIO start-up, two long outdoor ones."""


@dataclass(frozen=True, slots=True)
class TartanCamera:
    """One TartanAir camera as it is logged: intrinsics, model string, pose in the rig."""

    name: str
    """TartanAir's camera name, also the node ``name``."""
    width: int
    """Image width, pixels."""
    height: int
    """Image height, pixels."""
    fx: float
    """Horizontal scale: pixels per unit of normalised x (pinhole), per radian (fisheye, equirect)."""
    fy: float
    """Vertical scale, as ``fx``."""
    cx: float
    """Principal point x, OpenCV pixel-centre convention."""
    cy: float
    """Principal point y."""
    model: str
    """``simplecv.components.DistortionModel`` string."""
    coefficients: tuple[float, ...]
    """``DistortionCoefficients`` in that model's order."""
    camera_model: str
    """The node's ``camera_model`` tag."""
    file_suffix: str
    """What follows ``<frame>_<name>`` in TartanAir's PNG names."""
    rig_R_cam: Float64[ndarray, "3 3"] = field(default_factory=lambda: np.eye(3))
    """Rotation of the camera in the rig frame."""
    rig_t_cam: Float64[ndarray, "3"] = field(default_factory=lambda: np.zeros(3))
    """Optical centre in the rig frame, metres."""


def stereo(left: TartanCamera) -> tuple[TartanCamera, TartanCamera]:
    """The left camera and its right twin, 0.25 m along rig +x (NED +y)."""
    return left, replace(left, name=left.name.replace("lcam", "rcam"), rig_t_cam=np.array([0.25, 0.0, 0.0]))


FISHEYE_SCALE: float = 1000 / math.radians(195.0)
"""LinearSphere: r = theta * W / FOV, about 293.8 px per radian."""
CAMERA_SETS: dict[CameraSet, tuple[TartanCamera, ...]] = {
    "pinhole": stereo(TartanCamera("lcam_front", 640, 640, 320.0, 320.0, 319.5, 319.5, "brown_conrady", (0.0,) * 14, "pinhole", "")),
    "fisheye": stereo(TartanCamera("lcam_fish", 1000, 1000, FISHEYE_SCALE, FISHEYE_SCALE, 499.5, 499.5, "kannala_brandt", (0.0,) * 8, "kb4", "_image")),
    "equirect": (
        TartanCamera(
            "lcam_equirect", 2048, 1024, 1024 / math.pi, 1024 / math.pi, 1023.5, 511.5, "equirectangular", (math.pi,), "equirectangular", "_image",
            rig_R_cam=np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        ),
    ),
}  # fmt: skip
"""Every camera set; the first camera of each is the left one, whose frame the rig shares in position."""


def rig_poses(poses_ned: Float64[ndarray, "n 7"]) -> tuple[Float64[ndarray, "n 3"], Float64[ndarray, "n 4"]]:
    """world_T_rig in the Z-up world from TartanAir rows ``x y z qx qy qz qw`` (world_T_body, NED).

    Returns:
        Translations and quaternions (scalar last).
    """
    world_R_body: Float64[ndarray, "n 3 3"] = Rotation.from_quat(poses_ned[:, 3:]).as_matrix()
    world_R_rig: Float64[ndarray, "n 3 3"] = ZUP_R_NED @ world_R_body @ R_NED_RDF
    return poses_ned[:, :3] @ ZUP_R_NED.T, Rotation.from_matrix(world_R_rig).as_quat()


def imu_in_rig(values_ned: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
    """Body-frame NED vectors (gyro, specific force) in the RDF rig frame."""
    return values_ned @ R_NED_RDF


def aligned_window(num_frames: int, num_imu: int, frame_limit: int | None = None) -> tuple[Int64[ndarray, "k"], slice]:
    """Frames and IMU samples to keep: the IMU without its spline ends, and the frames inside the IMU span.

    Args:
        num_frames: Frames of the trajectory, frame k at k * 0.1 s.
        num_imu: IMU samples, sample j at j * 0.01 s.
        frame_limit: Keep only this many frames, and the IMU up to the last of them.

    Returns:
        Kept frame indices, and the slice of kept IMU samples.
    """
    imu: slice = slice(IMU_TRIM, num_imu - IMU_TRIM)
    frames_ns: Int64[ndarray, "n"] = np.arange(num_frames, dtype=np.int64) * FRAME_PERIOD_NS
    frames: Int64[ndarray, "k"] = np.flatnonzero((frames_ns >= imu.start * IMU_PERIOD_NS) & (frames_ns <= (imu.stop - 1) * IMU_PERIOD_NS))
    if frame_limit is not None:
        frames = frames[:frame_limit]
        imu = slice(imu.start, int(frames[-1]) * FRAME_PERIOD_NS // IMU_PERIOD_NS + 1)
    return frames, imu


@dataclass(frozen=True, slots=True)
class TrajectorySource:
    """One downloaded trajectory directory, ``<root>/<Env>/Data_<difficulty>/P0NN``."""

    directory: Path
    """Holds ``pose_lcam_front.txt``, ``imu/`` and one ``image_<camera>/`` per camera."""


@dataclass
class TartanAirConfig(FrameLimitedConfig):
    """TartanAir V2 trajectories (Hugging Face ``theairlabcmu/tartanair2``) as pinhole, fisheye or equirect rigs."""

    command: ClassVar[str] = "tartanair"
    """Registry key and CLI subcommand; ``--camera-set`` picks the catalog dataset."""

    _target: type = field(default_factory=lambda: TartanAirDataset)
    """Dataset class instantiated by ``setup()``."""
    camera_set: CameraSet = "fisheye"
    """Cameras the recordings carry; also the catalog dataset (``tartanair-<set>``)."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "tartanair-v2")
    """Raw tree, ``<root>/<Env>/Data_<difficulty>/P0NN/``, as the zips name their members."""
    trajectories: tuple[str, ...] = DEFAULT_TRAJECTORIES
    """``<Env>/<easy|hard>/P0NN`` to download and convert."""
    workers: int = 8
    """Members fetched at once by ``download``."""

    @property
    def name(self) -> str:
        """Catalog dataset and identity ``dataset`` part: one per camera set, since one dataset holds one layout."""
        return f"{self.command}-{self.camera_set}"


class TartanAirDataset(DataforgeDataset[TartanAirConfig, TrajectorySource]):
    """Range-extracts TartanAir V2 trajectories and converts them into base and gt layers for one camera set."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, paths.GT_LAYER)

    def __init__(self, config: TartanAirConfig) -> None:
        super().__init__(config)
        self.cameras: tuple[TartanCamera, ...] = CAMERA_SETS[config.camera_set]

    def trajectory_dir(self, trajectory: str) -> Path:
        """``<root>/<Env>/Data_<difficulty>/P0NN`` of ``<Env>/<difficulty>/P0NN``."""
        environment, difficulty, number = trajectory.split("/")
        return self.config.root / environment / f"Data_{difficulty}" / number

    def download(self) -> None:
        """Fetch every camera of every set, the IMU and the pose files of each trajectory; complete members are skipped."""
        archives: tuple[str, ...] = ("imu", *(f"image_{camera.name}" for cameras in CAMERA_SETS.values() for camera in cameras))
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            for trajectory in self.config.trajectories:
                prefix: str = f"{self.trajectory_dir(trajectory).relative_to(self.config.root).as_posix()}/"
                for archive in archives:
                    fetch, size = http_range(client, f"{HF_BASE_URL}{prefix.rsplit('/', 2)[0]}/{archive}.zip")
                    count, written = extract_members(fetch, size, prefix, self.config.root, workers=self.config.workers)
                    if not count:
                        raise FileNotFoundError(f"{archive}.zip at {HF_REVISION} holds no member under {prefix}")
                    print(f"{trajectory} {archive}: {count} members, {written / 1e6:.1f} MB fetched")

    def discover(self) -> list[tuple[SequenceIdentity, TrajectorySource]]:
        """Every configured trajectory whose poses, IMU and this set's cameras are on disk."""
        found: list[tuple[SequenceIdentity, TrajectorySource]] = []
        for trajectory in self.config.trajectories:
            directory: Path = self.trajectory_dir(trajectory)
            needed: list[Path] = [directory / "pose_lcam_front.txt", directory / "imu" / "gyro.txt", *(directory / f"image_{c.name}" for c in self.cameras)]
            if all(path.exists() for path in needed):
                found.append((SequenceIdentity(self.config.name, (trajectory.replace("/", "_"),)), TrajectorySource(directory)))
        return found

    def convert(self, identity: SequenceIdentity, source: TrajectorySource, *, force: bool) -> Path:
        """Write the pending base and gt layers of one trajectory, after reading and checking its clocks."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        poses: Float64[ndarray, "n 7"] = np.loadtxt(source.directory / "pose_lcam_front.txt", ndmin=2)
        imu_dir: Path = source.directory / "imu"
        imu_time_s: Float64[ndarray, "m"] = np.loadtxt(imu_dir / "imu_time.txt")
        if not np.allclose(imu_time_s, np.arange(imu_time_s.size) * IMU_PERIOD_NS / 1e9, atol=1e-6):
            raise ValueError(f"{imu_dir / 'imu_time.txt'} is not j * 0.01 s")
        frames, imu = aligned_window(poses.shape[0], imu_time_s.size, self.config.frame_limit)
        frames_ns: Int64[ndarray, "k"] = frames * FRAME_PERIOD_NS
        translations, quaternions = rig_poses(poses[frames])
        gt: GtTrajectory = GtTrajectory(frames_ns, translations, quaternions)
        self.write_layers(
            identity,
            targets,
            pending,
            {
                paths.BASE_LAYER: lambda recording: self.write_base(recording, identity, source, frames, imu),
                paths.GT_LAYER: lambda recording: self.write_gt(recording, gt),
            },
        )
        return targets[paths.BASE_LAYER]

    def write_base(self, recording: rr.RecordingStream, identity: SequenceIdentity, source: TrajectorySource, frames: Int64[ndarray, "k"], imu: slice) -> None:
        """Videos and calibration of this set's cameras, the IMU in the rig frame with its synthetic noise model, the properties."""
        log_rig_node(recording, RIG, reference="imu_00", num_cameras=len(self.cameras), name="tartanair", kind="ego")
        frames_ns: Int64[ndarray, "k"] = frames * FRAME_PERIOD_NS
        with work_dir(f"{identity.recording_id}-") as work:
            for index, camera in enumerate(self.cameras):
                images: list[Path] = [source.directory / f"image_{camera.name}" / f"{k:06d}_{camera.name}{camera.file_suffix}.png" for k in frames]
                clip: Path = work / f"cam_{index:02d}.mp4"
                with self.timer.stage("encode"):
                    encode_frames_to_mp4((path.read_bytes() for path in images), clip, source=FrameSource("png"), fps=10, gop=VIDEO_GOP, cq=VIDEO_CQ)
                node: str = schema.cam_path(RIG, index)
                rr.log(node, rr.AnyValues(name=camera.name, kind="rgb", camera_model=camera.camera_model), static=True, recording=recording)
                cam_R_rig: Float64[ndarray, "3 3"] = camera.rig_R_cam.T
                rr.log(
                    node,
                    rr.Transform3D(translation=-cam_R_rig @ camera.rig_t_cam, mat3x3=cam_R_rig, relation=rr.TransformRelation.ChildFromParent),
                    static=True,
                    recording=recording,
                )
                pinhole: rr.Pinhole = rr.Pinhole(
                    image_from_camera=[[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
                    width=camera.width,
                    height=camera.height,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.1,
                )
                distortion: CameraDistortion = CameraDistortion(camera.model, np.array(camera.coefficients))
                rr.log(schema.pinhole_path(RIG, index), PinholeWithDistortion(pinhole, distortion), static=True, recording=recording)
                log_camera_source(
                    recording, RIG, index, name=camera.name, source_width=camera.width, source_height=camera.height, video_codec="av1", cq=VIDEO_CQ, gop=VIDEO_GOP
                )
                log_video_stream(recording, clip, schema.video_path(RIG, index), times_ns=frames_ns, frame_indices=frames)
        imu_dir: Path = source.directory / "imu"
        imu_ns: Int64[ndarray, "m"] = np.arange(imu.start, imu.stop, dtype=np.int64) * IMU_PERIOD_NS
        gyro: Float64[ndarray, "m 3"] = imu_in_rig(np.loadtxt(imu_dir / "gyro.txt", ndmin=2)[imu])
        accel: Float64[ndarray, "m 3"] = imu_in_rig(np.loadtxt(imu_dir / "acc.txt", ndmin=2)[imu])
        log_imu(recording, RIG, 0, gyro=ImuChannel(imu_ns, gyro), accel=ImuChannel(imu_ns, accel), name="imu")
        imu_node: str = schema.imu_path(RIG, 0)
        recording.log(
            imu_node,
            ImuCalibration(
                gyro_noise_density=1.0e-4,
                accel_noise_density=1.0e-3,
                gyro_bias_random_walk=1.0e-5,
                accel_bias_random_walk=1.0e-4,
                rate_hz=1e9 / IMU_PERIOD_NS,
                source="synthetic placeholder: TartanAir's IMU is noise-free (spline of the GT poses); small values keep an estimator from over-trusting it",
            ),
            static=True,
        )
        recording.log(
            imu_node,
            rr.AnyValues(applied_time_shift_ns=0, time_shift_source="none: frames and IMU are on one synthetic clock", gravity_ms2=GRAVITY_MS2),
            static=True,
        )
        self.timer.capture_s = float(frames_ns[-1] - frames_ns[0]) / 1e9
        environment, difficulty, number = source.directory.parts[-3:]
        writing.send_capture_properties(
            recording,
            identity,
            num_cameras=len(self.cameras),
            num_frames=int(frames.size),
            start_time_ns=0,
            duration_s=self.timer.capture_s,
            clock_source="synthetic: frame k at k * 0.1 s, IMU sample j at j * 0.01 s; frame_index: TartanAir frame number",
            environment=environment,
            difficulty=difficulty.removeprefix("Data_"),
            trajectory=number,
            camera_set=self.config.camera_set,
            hf_revision=HF_REVISION,
        )

    def write_gt(self, recording: rr.RecordingStream, gt: GtTrajectory) -> None:
        """world_T_rig at every kept frame (exact, from ``pose_lcam_front.txt``), the path and its trail, the Z-up world."""
        rr.log("/", WORLD_UP_VIEW_COORDINATES["+z"], static=True, recording=recording)
        log_gt_path(recording, RIG, gt)
        recording.send_property(
            "gt",
            rr.AnyValues(
                num_poses=int(gt.times_ns.size),
                duration_s=gt.duration_s,
                trajectory_len_m=gt.length_m,
                world_up="+z",
                source="TartanAir pose_lcam_front.txt (rendered camera poses), NED world turned Z-up",
            ),
        )

    def default_blueprint(self) -> rrb.Blueprint:
        """Every camera beside the rig in 3D, over the IMU plots."""
        return blueprints.rig_blueprint(
            [blueprints.camera_view(camera.name, RIG, index, contents=[schema.video_path(RIG, index)]) for index, camera in enumerate(self.cameras)],
            rig=RIG,
            run_source=schema.GT_RUN_SOURCE,
            eye_controls=blueprints.follow_eye_controls((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
            plots=blueprints.imu_plots(RIG, 0),
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Preview card: the rig beside the left camera."""
        return blueprints.table_blueprint(
            len(self.cameras),
            rig=RIG,
            run_source=schema.GT_RUN_SOURCE,
            eye_controls=blueprints.follow_eye_controls((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
            front_pane=blueprints.camera_view(self.cameras[0].name, RIG, 0, contents=[schema.video_path(RIG, 0)]),
        )
