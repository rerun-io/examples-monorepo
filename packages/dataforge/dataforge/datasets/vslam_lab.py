"""VSLAM-LAB benchmarks (TUM RGB-D, EuRoC MAV, ...): one converter over VSLAM-LAB's common layout.

VSLAM-LAB fetches each benchmark and rewrites it into one per-sequence layout (``vslam_lab_source``), so this
module reads that layout and nothing benchmark-specific: ``--benchmark`` picks the VSLAM-LAB directory and the
catalog dataset (``vslam-lab-<benchmark>``), and ``BENCHMARKS`` holds the two claims the files do not make, the
world's up axis and where the rig looks. ``download`` only verifies a local tree; VSLAM-LAB's own
``download-sequence`` task fetches it (``set-benchmark-path <dir>`` puts it at ``<dir>/VSLAM-LAB-Benchmark``,
this config's default ``--root`` under ``DATAFORGE_RAW_ROOT=<dir>``).

Each sequence becomes, under one recording id (README#the-layer-rule):

* **base**: one AV1 video per ``rgb_<i>`` at its pinhole (rig_T_cam = calibration.yaml's ``T_BS``), the
  registered depth stream as ``rr.DepthImage`` on the same pinhole, the IMU, the capture properties.
* **gt**: groundtruth.csv's world_T_body on the rig node, its path and trail, the root ``ViewCoordinates``.
* **sensor_metadata**: the IMU noise model as simplecv's ``ImuCalibration``, where calibration.yaml has one.

The derived layers read only their sidecars (groundtruth.csv and calibration.yaml, copied verbatim), so they
rebuild without the frames. Every stream keeps the benchmark's own nanosecond stamps, unshifted, on
``video_time``: a gt row lands on the frame it was measured at, and a trajectory read back from the catalog
compares 1:1 with the upstream file.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64, UInt16
from numpy import ndarray
from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters
from simplecv.imu_calibration import ImuCalibration

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.basalt import FollowFrame
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
from dataforge.datasets.vslam_lab_source import (
    Calibration,
    CameraEntry,
    FrameIndex,
    ImuStream,
    PoseTable,
    read_calibration,
    read_frame_index,
    read_imu,
    read_poses,
)
from dataforge.euroc import GtTrajectory, TimestampedSamples, gt_trajectory, nominal_fps
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    TRAJECTORY_COLOR,
    ImuChannel,
    log_camera_node,
    log_imu,
    log_pose_track,
    log_rig_node,
    log_trail_segments,
    log_video_stream,
)
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4, require_av1_nvenc, resolve_ffmpeg
from dataforge.world_up import WORLD_UP_VIEW_COORDINATES, WorldUpAxis

BenchmarkChoice: TypeAlias = Literal["rgbdtum", "euroc"]
"""``--benchmark``: which VSLAM-LAB dataset, and which catalog dataset."""
RIG: int = 0
"""A VSLAM-LAB sequence is one rig."""
REQUIRED_FILES: tuple[str, ...] = ("calibration.yaml", "rgb.csv", "groundtruth.csv")
"""What makes a sequence directory convertible."""
SIDECARS: tuple[str, ...] = ("groundtruth.csv", "calibration.yaml")
"""Kept verbatim beside the rrds: the gt and sensor_metadata layers rebuild from them alone."""
IMAGE_PLANE_DISTANCE: float = 0.1
"""Frustum length in metres."""
GT_TRAJECTORY_WIDTH_UI_POINTS: float = 1.5
"""Stroke of the whole gt path."""


@dataclass(frozen=True, slots=True)
class Benchmark:
    """What one VSLAM-LAB benchmark looks like, and the two claims its files do not make."""

    directory: str
    """VSLAM-LAB's directory for it under the benchmark root."""
    label: str
    """The rig node's device name."""
    num_cameras: int
    """``rgb_<i>`` streams every sequence carries."""
    depth_cameras: tuple[int, ...]
    """Cameras with a registered ``depth_<i>`` stream."""
    has_imu: bool
    """Whether sequences ship ``imu_0.csv``."""
    world_up: WorldUpAxis
    """The gt world's up axis; the root ``ViewCoordinates`` of every gt layer."""
    follow: FollowFrame
    """Where the rig looks and which way is up, in the body frame; aims the Follow view."""
    gt_source: str
    """What produced groundtruth.csv."""


BENCHMARKS: dict[BenchmarkChoice, Benchmark] = {
    "rgbdtum": Benchmark(
        directory="RGBDTUM",
        label="kinect",
        num_cameras=1,
        depth_cameras=(0,),
        has_imu=False,
        # Measured over all nine freiburg1 sequences: the camera's image-up averages +z in the mocap world
        # (0.65..0.93 of unit length) while it looks down at the desk or floor (forward z -0.26..-0.68).
        world_up="+z",
        # The body is the RGB camera itself (T_BS identity): it looks along +z, image-up is -y.
        follow=FollowFrame(forward=(0.0, 0.0, 1.0), up=(0.0, -1.0, 0.0)),
        gt_source="motion capture (camera optical centre)",
    ),
    "euroc": Benchmark(
        directory="EUROC",
        label="vi-sensor",
        num_cameras=2,
        depth_cameras=(),
        has_imu=True,
        # Measured on MH_01_easy: gravity read through the gt rotation points +z at 0.99 of |g|.
        world_up="+z",
        # The body is imu0: cam0 looks along body +z (0.004, 0.026, 1.0) with image-up along body +x.
        follow=FollowFrame(forward=(0.004, 0.026, 1.0), up=(1.0, -0.015, -0.004)),
        gt_source="state_groundtruth_estimate0 (Vicon / Leica MS50, world_T_imu0)",
    ),
}
"""Every benchmark this converter knows; a new one is one entry once its world up is measured."""


@dataclass
class VslamLabConfig(FrameLimitedConfig):
    """VSLAM-LAB benchmark sequences: video, depth, IMU and ground truth in VSLAM-LAB's layout."""

    command: ClassVar[str] = "vslam-lab"
    """Registry key and CLI subcommand; ``--benchmark`` picks the catalog dataset."""

    _target: type = field(default_factory=lambda: VslamLabDataset)
    """Dataset class instantiated by ``setup()``."""
    benchmark: BenchmarkChoice = "rgbdtum"
    """VSLAM-LAB benchmark to convert; also the catalog dataset (``vslam-lab-<benchmark>``)."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "VSLAM-LAB-Benchmark")
    """The VSLAM-LAB benchmark tree, ``<root>/<BENCHMARK>/<sequence>/``."""

    @property
    def name(self) -> str:
        """Catalog dataset and identity ``dataset`` part: one per benchmark."""
        return f"{self.command}-{self.benchmark}"


def camera_parameters(camera: CameraEntry) -> PinholeParameters:
    """One calibration.yaml camera as simplecv parameters; ``T_BS`` is the pose in the rig (body) frame."""
    body_T_cam: Float64[ndarray, "4 4"] = camera.body_T_sensor
    coefficients: tuple[float, ...] = (*camera.distortion_coefficients, 0.0) if camera.distortion_type == "radtan4" else camera.distortion_coefficients
    return PinholeParameters(
        name=camera.cam_name,
        extrinsics=Extrinsics(world_R_cam=body_T_cam[:3, :3], world_t_cam=body_T_cam[:3, 3]),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF",
            fl_x=camera.focal_length[0],
            fl_y=camera.focal_length[1],
            cx=camera.principal_point[0],
            cy=camera.principal_point[1],
            width=camera.image_dimension[0],
            height=camera.image_dimension[1],
        ),
        distortion=None if camera.distortion_type is None else BrownConradyDistortion(*coefficients),
    )


def rig_reference(calibration: Calibration) -> str | None:
    """The sensor the body frame is, i.e. whose ``T_BS`` is the identity (``cam_00`` on TUM, ``imu_00`` on EuRoC)."""
    for kind, entries in (("cam", calibration.cameras), ("imu", calibration.imus)):
        for index, entry in enumerate(entries):
            if np.allclose(entry.body_T_sensor, np.eye(4)):
                return f"{kind}_{index:02d}"
    return None


def png_bytes(directory: Path, frames: FrameIndex) -> Iterator[bytes]:
    """The frame files of one stream, in order, as the encoder's PNG input."""
    for path in frames.paths:
        yield (directory / path).read_bytes()


def log_depth(recording: rr.RecordingStream, directory: Path, frames: FrameIndex, camera: int, *, meter: float) -> None:
    """Log one depth stream as ``rr.DepthImage`` on its camera's pinhole, frame by frame.

    ``DepthImage`` with ``meter`` set to the benchmark's ``depth_factor`` (native units per metre, 5000 on TUM),
    not metres as floats: the uint16 pixels stay exactly what the sensor wrote, at half the bytes, and the viewer
    and every reader still get metres from ``meter``. The pinhole is the RGB camera's because VSLAM-LAB depth is
    registered to it (``cam_type: rgb+depth``), so a pixel of one is the same ray as the pixel of the other.
    """
    entity: str = schema.depth_path(RIG, camera)
    for index, (stamp, path) in enumerate(zip(frames.times_ns, frames.paths, strict=True)):
        depth: UInt16[ndarray, "h w"] | None = cv2.imread(str(directory / path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"{directory / path} is not a readable 16-bit depth PNG")
        recording.set_time(schema.TIMELINE, duration=np.timedelta64(int(stamp), "ns"))
        recording.set_time(schema.FRAME_INDEX, sequence=index)
        recording.log(entity, rr.DepthImage(depth, meter=meter))
    recording.reset_time()


class VslamLabDataset(DataforgeDataset[VslamLabConfig, Path]):
    """Converts one VSLAM-LAB benchmark's sequences into base, gt and sensor_metadata layers."""

    def __init__(self, config: VslamLabConfig) -> None:
        super().__init__(config)
        self.benchmark: Benchmark = BENCHMARKS[config.benchmark]

    @property
    def benchmark_dir(self) -> Path:
        """``<root>/<BENCHMARK>``."""
        return self.config.root / self.benchmark.directory

    def discover(self) -> list[tuple[SequenceIdentity, Path]]:
        """Every sequence directory of the benchmark holding the three files a conversion needs, sorted."""
        if not self.benchmark_dir.is_dir():
            return []
        return [
            (SequenceIdentity(self.config.name, (sequence.name,)), sequence)
            for sequence in sorted(self.benchmark_dir.iterdir())
            if not transports.local_verify(sequence, required=REQUIRED_FILES)
        ]

    def download(self) -> None:
        """Verify the local VSLAM-LAB tree; VSLAM-LAB's ``download-sequence`` fetches it."""
        incomplete: list[str] = [
            f"{sequence.name} lacks {', '.join(missing)}"
            for sequence in sorted(path for path in self.benchmark_dir.glob("*") if path.is_dir())
            if (missing := transports.local_verify(sequence, required=REQUIRED_FILES))
        ]
        found: int = len(self.discover())
        if incomplete or not found:
            raise FileNotFoundError(f"{self.benchmark_dir}: {'; '.join(incomplete) or 'no sequence'}; fetch it with VSLAM-LAB's download-sequence")
        print(f"{self.config.name}: {found} sequence(s) in {self.benchmark_dir}")

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Write whichever layers are missing; base from the frames, the derived ones from their sidecars."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        sidecars: dict[str, Path] = {name: paths.sidecar_path(paths.output_root(), identity, name) for name in SIDECARS}
        for name, sidecar in sidecars.items():
            if paths.BASE_LAYER in pending or not sidecar.is_file():
                with writing.atomic_write(sidecar) as staged:
                    shutil.copyfile(source / name, staged)
        calibration: Calibration = read_calibration(sidecars["calibration.yaml"])
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            paths.BASE_LAYER: lambda recording: self.write_base(recording, identity, source, calibration),
            paths.GT_LAYER: lambda recording: self.write_gt(recording, read_poses(sidecars["groundtruth.csv"])),
        }
        if any(imu.sigma_g_c is not None for imu in calibration.imus):
            writers[paths.SENSOR_METADATA_LAYER] = lambda recording: write_sensor_metadata(recording, calibration)
        self.write_layers(identity, targets, [layer for layer in pending if layer in writers], writers)
        return targets[paths.BASE_LAYER]

    def write_base(self, recording: rr.RecordingStream, identity: SequenceIdentity, source: Path, calibration: Calibration) -> None:
        """The sensor layer: every camera's video, its depth where registered, the IMU, the capture properties."""
        if len(calibration.cameras) != self.benchmark.num_cameras:
            raise ValueError(f"{source}: {len(calibration.cameras)} cameras, but {self.benchmark.directory} declares {self.benchmark.num_cameras}")
        require_av1_nvenc(resolve_ffmpeg())
        limit: int | None = self.config.frame_limit
        streams: dict[str, FrameIndex] = {
            name: FrameIndex(times_ns=frames.times_ns[:limit], paths=frames.paths[:limit]) for name, frames in read_frame_index(source / "rgb.csv").items()
        }
        log_rig_node(recording, RIG, reference=rig_reference(calibration), num_cameras=len(calibration.cameras), name=self.benchmark.label, kind="ego")
        work_dir: Path = paths.work_root() / identity.recording_id
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            for index, camera in enumerate(calibration.cameras):
                frames: FrameIndex = streams[camera.cam_name]
                clip: Path = work_dir / f"cam_{index:02d}.mp4"
                with self.timer.stage("encode"):
                    encode_frames_to_mp4(png_bytes(source, frames), clip, source=FrameSource("png"), fps=nominal_fps(frames.times_ns))
                log_camera_node(
                    recording,
                    RIG,
                    index,
                    camera_parameters(camera),
                    name=camera.cam_name,
                    kind="grayscale" if camera.cam_type == "gray" else "rgb",
                    image_plane_distance=IMAGE_PLANE_DISTANCE,
                    camera_model=f"pinhole-{camera.distortion_type}" if camera.distortion_type else "pinhole",
                )
                frame_indices: Int64[ndarray, "n_frames"] = np.arange(frames.times_ns.size, dtype=np.int64)
                log_video_stream(recording, clip, schema.video_path(RIG, index), times_ns=frames.times_ns, frame_indices=frame_indices)
                if camera.depth_name is not None and camera.depth_factor is not None:
                    log_depth(recording, source, streams[camera.depth_name], index, meter=camera.depth_factor)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        first_ns: int = int(streams["rgb_0"].times_ns[0])
        last_ns: int = max(int(frames.times_ns[-1]) for frames in streams.values())
        if calibration.imus:
            imu: ImuStream = read_imu(source / "imu_0.csv")
            kept: Int64[ndarray, "n_kept"] = np.flatnonzero(imu.gyro.times_ns <= last_ns) if limit is not None else np.arange(imu.gyro.times_ns.size)
            body_T_imu: Float64[ndarray, "4 4"] = calibration.imus[0].body_T_sensor
            log_imu(
                recording,
                RIG,
                0,
                gyro=ImuChannel(times_ns=imu.gyro.times_ns[kept], values_xyz=imu.gyro.values_xyz[kept]),
                accel=ImuChannel(times_ns=imu.accel.times_ns[kept], values_xyz=imu.accel.values_xyz[kept]),
                name=calibration.imus[0].imu_name,
                rig_T_imu=rr.Transform3D(translation=body_T_imu[:3, 3], mat3x3=body_T_imu[:3, :3]),
            )
        self.timer.capture_s = (last_ns - first_ns) / 1e9
        writing.send_capture_properties(
            recording,
            identity,
            num_cameras=len(calibration.cameras),
            num_frames=max(frames.times_ns.size for frames in streams.values()),
            start_time_ns=first_ns,
            duration_s=self.timer.capture_s,
            clock_source="the benchmark's own rgb.csv / imu_0.csv / groundtruth.csv nanosecond stamps, unshifted; frame_index: row of rgb.csv",
            benchmark=self.benchmark.directory,
            sequence=source.name,
            has_depth=bool(self.benchmark.depth_cameras),
            has_imu=bool(calibration.imus),
        )

    def write_gt(self, recording: rr.RecordingStream, poses: PoseTable) -> None:
        """The ground-truth layer: world_T_rig on the rig node, the whole path and its trail, the world axes.

        A dropout row (a non-unit quaternion) becomes identity rotation with its translation kept, the rule
        ``dataforge.euroc.gt_trajectory`` holds for every EuRoC-style file. A file whose rotation never changes
        is position-only ground truth (VSLAM-LAB's own EuRoC csv is cam0 positions with a constant placeholder
        quaternion): it is still logged, but ``has_rotation`` is false and convert says so.
        """
        trajectory: GtTrajectory = gt_trajectory(
            TimestampedSamples(times_ns=poses.times_ns, values=np.column_stack([poses.translations_xyz, poses.quaternions_xyzw[:, [3, 0, 1, 2]]]))
        )
        has_rotation: bool = bool(np.ptp(trajectory.quaternions_xyzw, axis=0).max() > 0.0)
        if not has_rotation:
            print(f"  warning: {self.benchmark.directory} groundtruth.csv never changes its rotation; logging it as position-only ground truth")
        rr.log("/", WORLD_UP_VIEW_COORDINATES[self.benchmark.world_up], static=True, recording=recording)
        log_pose_track(
            recording,
            schema.rig_path(RIG),
            times_ns=trajectory.times_ns,
            translations_xyz=trajectory.translations_xyz,
            quaternions_xyzw=trajectory.quaternions_xyzw,
        )
        rr.log(
            schema.trajectory_path(schema.GT_RUN_SOURCE),
            rr.LineStrips3D([trajectory.translations_xyz], colors=TRAJECTORY_COLOR, radii=rr.Radius.ui_points(GT_TRAJECTORY_WIDTH_UI_POINTS)),
            static=True,
            recording=recording,
        )
        log_trail_segments(recording, schema.trail_path(schema.GT_RUN_SOURCE), times_ns=trajectory.times_ns, translations_xyz=trajectory.translations_xyz)
        steps_m: Float64[ndarray, "n_steps"] = np.linalg.norm(np.diff(trajectory.translations_xyz, axis=0), axis=1)
        recording.send_property(
            "gt",
            rr.AnyValues(
                num_poses=int(trajectory.times_ns.size),
                duration_s=(int(trajectory.times_ns[-1]) - int(trajectory.times_ns[0])) / 1e9,
                trajectory_len_m=float(steps_m.sum()),
                num_sanitized=trajectory.num_sanitized,
                has_rotation=has_rotation,
                world_up=self.benchmark.world_up,
                source=self.benchmark.gt_source,
            ),
        )

    def default_blueprint(self) -> rrb.Blueprint:
        """Benchmark-wide layout: the rig in 3D beside every camera (and its depth), over the IMU plots."""
        panes: list[rrb.Spatial2DView] = []
        for index in range(self.benchmark.num_cameras):
            panes.append(blueprints.camera_view(f"rgb_{index}", RIG, index, contents=[schema.video_path(RIG, index)]))
            if index in self.benchmark.depth_cameras:
                panes.append(blueprints.camera_view(f"depth_{index}", RIG, index, contents=[schema.depth_path(RIG, index)]))
        plots: list[rrb.TimeSeriesView] = (
            [
                blueprints.sensor_plot("Gyroscope", schema.imu_path(RIG, 0), schema.gyro_path(RIG, 0)),
                blueprints.sensor_plot("Accelerometer", schema.imu_path(RIG, 0), schema.accel_path(RIG, 0)),
            ]
            if self.benchmark.has_imu
            else []
        )
        follow: FollowFrame = self.benchmark.follow
        return blueprints.rig_blueprint(
            panes, rig=RIG, run_source=schema.GT_RUN_SOURCE, eye_controls=blueprints.follow_eye_controls(follow.forward, follow.up), plots=plots
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Preview card: the rig with no video or depth beside ``rgb_0``."""
        follow: FollowFrame = self.benchmark.follow
        return blueprints.table_blueprint(
            self.benchmark.num_cameras,
            rig=RIG,
            run_source=schema.GT_RUN_SOURCE,
            eye_controls=blueprints.follow_eye_controls(follow.forward, follow.up),
            front_pane=blueprints.camera_view("rgb_0", RIG, 0, contents=[schema.video_path(RIG, 0)]),
            depth_cameras=self.benchmark.depth_cameras,
        )

    def table_fields(self) -> writing.TableFields:
        """Sequence, length and ground-truth size on every card and table row."""
        fields: tuple[writing.TableField, ...] = (
            writing.TableField("property:capture:sequence", "sequence"),
            writing.TableField("property:capture:num_frames", "frames"),
            writing.TableField("property:capture:duration_s", "duration_s"),
            writing.TableField("property:gt:trajectory_len_m", "gt path m"),
        )
        return writing.TableFields(cards=fields, table=fields)


def write_sensor_metadata(recording: rr.RecordingStream, calibration: Calibration) -> None:
    """The IMU noise model calibration.yaml hands VIO baselines, as ``ImuCalibration`` on each IMU node.

    VSLAM-LAB writes OKVIS-style values, deliberately looser than a datasheet (EuRoC's sigma_g_c is 2e-3 where
    the sensor sheet says 1.7e-4); ``source`` says so, so a consumer does not mistake them for measured densities.
    """
    for index, imu in enumerate(calibration.imus):
        recording.log(
            schema.imu_path(RIG, index),
            ImuCalibration(
                gyro_noise_density=imu.sigma_g_c,
                accel_noise_density=imu.sigma_a_c,
                gyro_bias_random_walk=imu.sigma_gw_c,
                accel_bias_random_walk=imu.sigma_aw_c,
                rate_hz=imu.fps,
                source=f"VSLAM-LAB calibration.yaml imus[{index}] (OKVIS-style sigma_*_c, not datasheet densities)",
            ),
            static=True,
        )
