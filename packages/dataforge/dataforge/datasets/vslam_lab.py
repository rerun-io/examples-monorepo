"""VSLAM-LAB benchmarks (TUM RGB-D, EuRoC MAV, ...): one converter over VSLAM-LAB's common layout.

VSLAM-LAB fetches each benchmark and rewrites it into one per-sequence layout (``vslam_lab_source``), so this
module reads that layout and nothing benchmark-specific: ``--benchmark`` picks the VSLAM-LAB directory and the
catalog dataset (``vslam-lab-<benchmark>``), and ``BENCHMARKS`` declares each one's sensor layout (every
calibration.yaml is checked against it), its ground truth, and the two claims the files do not make, the world's up
axis and where the rig looks. VSLAM-LAB's own ``download-sequence`` task fetches the tree (``set-benchmark-path
<dir>`` puts it at ``<dir>/VSLAM-LAB-Benchmark``, this config's default ``--root`` under ``DATAFORGE_RAW_ROOT=<dir>``);
``download`` verifies it and extracts EuRoC's own ground truth from the kept sequence zip.

Each sequence becomes, under one recording id (README#the-layer-rule):

* **base**: one AV1 video per ``rgb_<i>`` at its pinhole (rig_T_cam = calibration.yaml's ``T_BS``), the
  registered depth stream as ``rr.EncodedDepthImage`` on the same pinhole, the IMU, the capture properties.
* **gt**: the benchmark's ground truth as world_T_body on the rig node, its path and trail, the root ``ViewCoordinates``.
* **sensor_metadata**: the IMU noise model as simplecv's ``ImuCalibration``, where calibration.yaml has one.

The derived layers read the sequence's own ground-truth csv and calibration.yaml in place, so they rebuild without
the frames; no sidecar copy is kept. Every stream keeps the benchmark's own nanosecond stamps, unshifted, on
``video_time``: a gt row lands on the frame it was measured at, and a trajectory read back from the catalog
compares 1:1 with the upstream file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64
from numpy import ndarray
from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters
from simplecv.imu_calibration import ImuCalibration

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.archives import ZipMemberReader
from dataforge.basalt import FollowFrame
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.vslam_lab_source import (
    EUROC_GT_FILE,
    EUROC_GT_MEMBER,
    Calibration,
    CameraEntry,
    FrameIndex,
    ImuStream,
    StreamName,
    read_calibration,
    read_euroc_gt,
    read_frame_index,
    read_imu,
    read_poses,
)
from dataforge.euroc import GtTrajectory, nominal_fps
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import frame_index_column, log_camera_node, log_gt_path, log_imu, log_rig_node, log_video_stream, time_column
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4, require_av1_nvenc, resolve_ffmpeg, work_dir
from dataforge.world_up import WORLD_UP_VIEW_COORDINATES, WorldUpAxis

BenchmarkChoice: TypeAlias = Literal["rgbdtum", "euroc"]
"""``--benchmark``: which VSLAM-LAB dataset, and which catalog dataset."""
RIG: int = 0
"""A VSLAM-LAB sequence is one rig."""
REQUIRED_FILES: tuple[str, ...] = ("calibration.yaml", "rgb.csv")
"""What makes a sequence directory discoverable; the ground truth is checked by ``convert``, the frames read only while
base is pending."""
IMAGE_PLANE_DISTANCE: float = 0.1
"""Frustum length in metres."""
GT_TRAJECTORY_WIDTH_UI_POINTS: float = 1.5
"""Stroke of the whole gt path."""
DEPTH_FRAMES_PER_CHUNK: int = 4
"""Depth frames per sent chunk: about 1 MB of PNG, the chunk size Rerun's own batcher chose for these frames."""
PNG_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"
"""First eight bytes of every PNG file."""


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Where a benchmark's world_T_body comes from: one named file and the adapter for its format."""

    file: str
    """The csv in the sequence directory the gt layer reads."""
    read: Callable[[Path], GtTrajectory]
    """Its reader, which returns world_T_body with the quaternion scalar last."""
    source: str
    """What produced it; the gt layer's ``source`` property."""
    zip_member: str | None = None
    """Where ``download`` extracts ``file`` from, verbatim, in VSLAM-LAB's kept ``<room>/<seq>/<seq>.zip``; ``None``
    when VSLAM-LAB writes ``file`` itself."""


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
    gt: GroundTruth
    """The ground truth the gt layer reads."""


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
        gt=GroundTruth(file="groundtruth.csv", read=read_poses, source="motion capture (camera optical centre)"),
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
        # VSLAM-LAB's groundtruth.csv is cam0 positions from TUM's supp_v2 with a placeholder rotation, not world_T_body.
        gt=GroundTruth(
            file=EUROC_GT_FILE,
            read=read_euroc_gt,
            source="state_groundtruth_estimate0 (Vicon / Leica MS50, world_T_imu0)",
            zip_member=EUROC_GT_MEMBER,
        ),
    ),
}
"""Every benchmark this converter knows; a new one is one entry once its world up is measured."""


@dataclass
class VslamLabConfig(DataforgeDatasetConfig):
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


def log_depth(recording: rr.RecordingStream, directory: Path, frames: FrameIndex, camera: int, *, meter: float) -> None:
    """Send one depth stream as ``rr.EncodedDepthImage`` on its camera's pinhole: the sensor's own 16-bit PNGs.

    The PNG bytes go in as they are, with ``meter`` set to the benchmark's ``depth_factor`` (native units per metre,
    5000 on TUM): the pixels stay exactly what the sensor wrote, nothing is decoded here, and the viewer and every
    reader still get metres from ``meter``. The pinhole is the RGB camera's because VSLAM-LAB depth is registered to
    it (``cam_type: rgb+depth``), so a pixel of one is the same ray as the pixel of the other. Sent as columns on the
    two capture clocks, ``DEPTH_FRAMES_PER_CHUNK`` frames at a time, so no wall clock enters the recording.

    Raises:
        ValueError: A frame is not a 16-bit grayscale PNG, the one encoding ``meter`` describes.
    """
    for start in range(0, frames.times_ns.size, DEPTH_FRAMES_PER_CHUNK):
        batch: slice = slice(start, start + DEPTH_FRAMES_PER_CHUNK)
        blobs: list[bytes] = [(directory / path).read_bytes() for path in frames.paths[batch]]
        for path, blob in zip(frames.paths[batch], blobs, strict=True):
            if blob[:8] != PNG_SIGNATURE or blob[24:26] != b"\x10\x00":  # IHDR bit depth 16, colour type 0 (gray)
                raise ValueError(f"{directory / path} is not a 16-bit grayscale PNG")
        rr.send_columns(
            schema.depth_path(RIG, camera),
            indexes=[time_column(frames.times_ns[batch]), frame_index_column(np.arange(start, start + len(blobs), dtype=np.int64))],
            columns=rr.EncodedDepthImage.columns(blob=blobs, media_type=["image/png"] * len(blobs), meter=[meter] * len(blobs)),
            recording=recording,
        )


class VslamLabDataset(DataforgeDataset[VslamLabConfig, Path]):
    """Converts one VSLAM-LAB benchmark's sequences into base, gt and sensor_metadata layers."""

    def __init__(self, config: VslamLabConfig) -> None:
        super().__init__(config)
        self.benchmark: Benchmark = BENCHMARKS[config.benchmark]
        self.layers: tuple[str, ...] = (
            (paths.BASE_LAYER, paths.GT_LAYER, paths.SENSOR_METADATA_LAYER) if self.benchmark.has_imu else (paths.BASE_LAYER, paths.GT_LAYER)
        )

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
        """Verify the local VSLAM-LAB tree, and put a ground truth VSLAM-LAB does not write beside each sequence.

        VSLAM-LAB's ``download-sequence`` fetches the tree. Directories without the sequence files are not sequences
        (VSLAM-LAB keeps EuRoC's ``supp_v2/`` and room archives beside them), so only an empty benchmark directory is
        an error. Where the benchmark's ground truth is a member of the sequence zip (EuRoC), that member is copied
        out verbatim once, so the gt layer reads a small file in the sequence directory and the zip may go.

        Raises:
            FileNotFoundError: No sequence, or a sequence whose ground truth is neither extracted nor in its zip.
        """
        sequences: list[tuple[SequenceIdentity, Path]] = self.discover()
        if not sequences:
            raise FileNotFoundError(f"{self.benchmark_dir} holds no sequence with {', '.join(REQUIRED_FILES)}; fetch one with VSLAM-LAB's download-sequence")
        member: str | None = self.benchmark.gt.zip_member
        for _, source in sequences:
            target: Path = source / self.benchmark.gt.file
            if member is None or target.is_file():
                continue
            pattern: str = f"*/{source.name}/{source.name}.zip"
            kept: list[Path] = sorted(self.benchmark_dir.glob(pattern))
            if not kept:
                raise FileNotFoundError(f"{target} is missing and no {self.benchmark_dir / pattern} holds {member}; VSLAM-LAB keeps it unless its retention is MINIMAL")
            with ZipMemberReader(kept[0]) as reader, writing.atomic_write(target) as staged:
                staged.write_bytes(reader.read_member(member))
            print(f"{source.name}: {member} of {kept[0]} -> {target}")
        print(f"{self.config.name}: {len(sequences)} sequence(s) in {self.benchmark_dir}")

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Write whichever layers are missing, each from this sequence's own VSLAM-LAB files.

        Everything but the frames, calibration.yaml and (when gt is pending) the ground truth, is read and checked
        before the first layer is written, so a refused sequence leaves every published layer as it was, even under
        ``--force``. The derived layers read those files in place rather than from a sidecar copy: the converter never
        deletes the VSLAM-LAB tree, so there is one copy and no layer can come from a stale one.
        """
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        calibration: Calibration = self.checked_calibration(source)
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            paths.BASE_LAYER: lambda recording: self.write_base(recording, identity, source, calibration),
        }
        if paths.GT_LAYER in pending:
            poses: GtTrajectory = self.read_gt(source)
            writers[paths.GT_LAYER] = lambda recording: self.write_gt(recording, poses)
        if any(imu.sigma_g_c is not None for imu in calibration.imus):
            writers[paths.SENSOR_METADATA_LAYER] = lambda recording: write_sensor_metadata(recording, calibration)
        self.write_layers(identity, targets, pending, writers)
        return targets[paths.BASE_LAYER]

    def write_base(self, recording: rr.RecordingStream, identity: SequenceIdentity, source: Path, calibration: Calibration) -> None:
        """The sensor layer: every camera's video, its depth where registered, the IMU, the capture properties."""
        require_av1_nvenc(resolve_ffmpeg())
        streams: dict[StreamName, FrameIndex] = read_frame_index(source / "rgb.csv")
        log_rig_node(recording, RIG, reference=rig_reference(calibration), num_cameras=len(calibration.cameras), name=self.benchmark.label, kind="ego")
        with work_dir(f"{identity.recording_id}-") as work:
            for index, camera in enumerate(calibration.cameras):
                frames: FrameIndex = streams[camera.cam_name]
                clip: Path = work / f"cam_{index:02d}.mp4"
                with self.timer.stage("encode"):
                    encode_frames_to_mp4(((source / path).read_bytes() for path in frames.paths), clip, source=FrameSource("png"), fps=nominal_fps(frames.times_ns))
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
        first_ns: int = int(streams["rgb_0"].times_ns[0])
        last_ns: int = max(int(frames.times_ns[-1]) for frames in streams.values())
        if self.benchmark.has_imu:
            imu: ImuStream = read_imu(source / "imu_0.csv")
            body_T_imu: Float64[ndarray, "4 4"] = calibration.imus[0].body_T_sensor
            log_imu(
                recording,
                RIG,
                0,
                gyro=imu.gyro,
                accel=imu.accel,
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
            has_imu=self.benchmark.has_imu,
        )

    def checked_calibration(self, source: Path) -> Calibration:
        """calibration.yaml, refused unless it states the sensor layout ``BENCHMARKS`` declares, the one source the
        layers, the properties and the blueprints then use.

        Raises:
            ValueError: Its camera count, depth cameras or IMU presence differ from the benchmark's.
        """
        path: Path = source / "calibration.yaml"
        calibration: Calibration = read_calibration(path)
        depth_cameras: tuple[int, ...] = tuple(index for index, camera in enumerate(calibration.cameras) if camera.depth_name is not None)
        found: tuple[int, tuple[int, ...], bool] = (len(calibration.cameras), depth_cameras, bool(calibration.imus))
        declared: tuple[int, tuple[int, ...], bool] = (self.benchmark.num_cameras, self.benchmark.depth_cameras, self.benchmark.has_imu)
        if found != declared:
            raise ValueError(f"{path}: (cameras, depth cameras, IMU) are {found}, but {self.benchmark.directory} declares {declared}")
        return calibration

    def read_gt(self, source: Path) -> GtTrajectory:
        """The benchmark's world_T_body, read from the file its ``GroundTruth`` names by that file's own reader.

        The reader has already turned a dropout row (a non-unit quaternion) into identity rotation with its
        translation kept, the rule ``dataforge.euroc.sanitized`` holds for every EuRoC-style file.

        Raises:
            FileNotFoundError: The file is not there; for EuRoC, ``download`` extracts it from the kept zip.
            ValueError: The file holds no pose.
        """
        path: Path = source / self.benchmark.gt.file
        if not path.is_file():
            writer: str = f"dataforge-download {self.config.command} --benchmark {self.config.benchmark}" if self.benchmark.gt.zip_member else "VSLAM-LAB"
            raise FileNotFoundError(f"{path} is missing; {writer} writes it")
        poses: GtTrajectory = self.benchmark.gt.read(path)
        if not poses.times_ns.size:
            raise ValueError(f"{path} holds no pose")
        return poses

    def write_gt(self, recording: rr.RecordingStream, poses: GtTrajectory) -> None:
        """The ground-truth layer: world_T_rig on the rig node, the whole path and its trail, the world axes."""
        rr.log("/", WORLD_UP_VIEW_COORDINATES[self.benchmark.world_up], static=True, recording=recording)
        log_gt_path(
            recording,
            RIG,
            times_ns=poses.times_ns,
            translations_xyz=poses.translations_xyz,
            quaternions_xyzw=poses.quaternions_xyzw,
            radius=GT_TRAJECTORY_WIDTH_UI_POINTS,
            radius_unit="ui_points",
        )
        steps_m: Float64[ndarray, "n_steps"] = np.linalg.norm(np.diff(poses.translations_xyz, axis=0), axis=1)
        recording.send_property(
            "gt",
            rr.AnyValues(
                num_poses=int(poses.times_ns.size),
                duration_s=(int(poses.times_ns[-1]) - int(poses.times_ns[0])) / 1e9,
                trajectory_len_m=float(steps_m.sum()),
                num_sanitized=poses.num_sanitized,
                world_up=self.benchmark.world_up,
                source=self.benchmark.gt.source,
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
