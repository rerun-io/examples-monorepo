"""TUM-VI (the TUM Visual-Inertial benchmark, 512x512 16-bit release): the raw ASL tree → base, gt and sensor_metadata.

Each local ``dataset-<name>_512_16`` directory (``tumvi_source`` describes the layout) becomes, under the recording id
``tum-vi__<name>`` (README#the-layer-rule):

* **base**: the stereo pair as one AV1 video per camera at its KB4 (equidistant) pinhole, ``rig_T_cam`` = Kalibr's
  ``T_cam_imu`` inverted; the 200 Hz IMU; the capture properties. Only frames both cameras took at the same stamp are
  kept (the cameras are hardware-synchronized; a frame only one camera has is dropped and counted in the properties).
  The 16-bit PNGs go to the encoder as they are, so the video holds their upper 8 bits (as an 8-bit release would).
* **gt**: mocap0, world_T_imu0, every sample, on ``rig_00``, plus its path and the root ``ViewCoordinates``. Outside
  the motion-capture room there is none, so the corridor, magistrale and slides runs have ground truth at their start
  and end only; the layer's properties list those spans.
* **sensor_metadata**: imu_config.yaml's noise model as simplecv's ``ImuCalibration`` on the IMU node, with
  ``applied_time_shift_ns`` 0: TUM-VI's camera and IMU stamps are already on one clock, and nothing is shifted here.

Every stream keeps the dataset's own nanosecond stamps, unshifted, on ``video_time``. The derived layers read the
sequence's own small files in place (mocap0/data.csv, dso/imu_config.yaml), so they rebuild without the frames; no
sidecar copy is kept, as for vslam-lab: the converter never deletes the raw tree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int64
from numpy import ndarray
from simplecv.imu_calibration import ImuCalibration

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.basalt import CalibratedCamera, FollowFrame, camera_parameters
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.tumvi_source import (
    ImuNoise,
    MocapSpan,
    StereoFrames,
    mocap_spans,
    pair_stereo,
    read_camchain,
    read_imu_noise,
)
from dataforge.euroc import GT_VALUE_COLUMNS, IMU_VALUE_COLUMNS, TimestampedSamples, gt_trajectory, nominal_fps, read_camera_index, read_numeric_csv
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import GtTrajectory, ImuChannel, log_camera_node, log_gt_path, log_imu, log_rig_node, log_video_stream
from dataforge.video_encoding import FrameSource, encode_frames_to_mp4, require_av1_nvenc, resolve_ffmpeg, work_dir
from dataforge.world_up import WORLD_UP_VIEW_COORDINATES, WorldUpAxis

RIG: int = 0
"""A TUM-VI sequence is one rig."""
IMU: int = 0
"""One IMU, ``imu0`` → ``imu_00``; the rig frame is its frame."""
RIG_REFERENCE: str = "imu_00"
"""Every extrinsic is stated against the IMU (Kalibr's ``T_cam_imu``), so the rig frame is the IMU frame."""
SEQUENCE_GLOB: str = "dataset-*_512_16"
"""Sequence directories of the 512x512 16-bit release."""
REQUIRED_FILES: tuple[str, ...] = (
    "mav0/cam0/data.csv",
    "mav0/cam1/data.csv",
    "mav0/imu0/data.csv",
    "mav0/mocap0/data.csv",
    "dso/camchain.yaml",
    "dso/imu_config.yaml",
)
"""What makes a sequence directory convertible."""
IMAGE_PLANE_DISTANCE: float = 0.1
"""Frustum length in metres; the stereo baseline is ~10 cm."""
WORLD_UP: WorldUpAxis = "+z"
"""The mocap world's up axis. Measured on room1: gravity read through the mocap0 rotation points +z at 0.997 of |g|."""
FOLLOW: FollowFrame = FollowFrame(forward=(-0.030, -0.999, 0.011), up=(-0.027, 0.012, 1.0))
"""Where the rig looks, in the IMU frame: ``basalt.follow_frame`` of camchain.yaml (identical in all 20 sequences).
The cameras look along IMU -y with image-up along IMU +z, so no camera needs a quarter turn."""
GT_SOURCE: str = "mocap0 (motion capture, world_T_imu0; start and end only outside the room)"
"""The gt layer's ``source`` property."""


def sequence_name(directory: Path) -> str:
    """``dataset-magistrale1_512_16`` → ``magistrale1``: the identity part and the segment id's suffix."""
    return directory.name.removeprefix("dataset-").removesuffix("_512_16")


@dataclass
class TumViConfig(DataforgeDatasetConfig):
    """TUM-VI: hardware-synchronized fisheye stereo, a 200 Hz IMU and motion-capture ground truth (raw ASL tree)."""

    command: ClassVar[str] = "tum-vi"
    """Registry key, CLI subcommand and catalog dataset."""

    _target: type = field(default_factory=lambda: TumViDataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "tum-vi")
    """The raw tree holding the ``dataset-<name>_512_16`` directories; only read."""


class TumViDataset(DataforgeDataset[TumViConfig, Path]):
    """Converts TUM-VI's sequences into base, gt and sensor_metadata layers."""

    def discover(self) -> list[tuple[SequenceIdentity, Path]]:
        """Every complete sequence directory under ``root``, sorted by name."""
        if not self.config.root.is_dir():
            return []
        return [
            (SequenceIdentity(self.config.name, (sequence_name(sequence),)), sequence)
            for sequence in sorted(self.config.root.glob(SEQUENCE_GLOB))
            if sequence.is_dir() and not transports.local_verify(sequence, required=REQUIRED_FILES)
        ]

    def download(self) -> None:
        """Verify the local tree: TUM-VI is fetched by hand (the ``_512_16`` tarballs, extracted under ``root``).

        Raises:
            FileNotFoundError: No complete sequence under ``root``.
        """
        sequences: list[tuple[SequenceIdentity, Path]] = self.discover()
        if not sequences:
            raise FileNotFoundError(f"{self.config.root} holds no {SEQUENCE_GLOB} directory with {', '.join(REQUIRED_FILES)}")
        print(f"{self.config.name}: {len(sequences)} sequence(s) in {self.config.root}")

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Write whichever layers are missing, each from this sequence's own files.

        The calibration, the noise model and (when gt is pending) mocap0 are read and checked before the first layer is
        written, so a refused sequence leaves every published layer as it was, even under ``--force``.
        """
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        cameras: tuple[CalibratedCamera, ...] = read_camchain(source / "dso/camchain.yaml")
        noise: ImuNoise = read_imu_noise(source / "dso/imu_config.yaml")
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            paths.BASE_LAYER: lambda recording: self.write_base(recording, identity, source, cameras),
            paths.SENSOR_METADATA_LAYER: lambda recording: write_sensor_metadata(recording, noise),
        }
        if paths.GT_LAYER in pending:
            poses: GtTrajectory = read_mocap(source / "mav0/mocap0/data.csv")
            writers[paths.GT_LAYER] = lambda recording: write_gt(recording, poses)
        self.write_layers(identity, targets, pending, writers)
        return targets[paths.BASE_LAYER]

    def write_base(self, recording: rr.RecordingStream, identity: SequenceIdentity, source: Path, cameras: tuple[CalibratedCamera, ...]) -> None:
        """The sensor layer: both cameras' videos over their stereo pairs, the IMU, the capture properties."""
        require_av1_nvenc(resolve_ffmpeg())
        mav0: Path = source / "mav0"
        stereo: StereoFrames = pair_stereo(
            read_camera_index((mav0 / "cam0/data.csv").read_bytes()), read_camera_index((mav0 / "cam1/data.csv").read_bytes())
        )
        log_rig_node(recording, RIG, reference=RIG_REFERENCE, num_cameras=len(cameras), name="tum-vi", kind="ego")
        frame_indices: Int64[ndarray, "n_pairs"] = np.arange(stereo.times_ns.size, dtype=np.int64)
        with work_dir(f"{identity.recording_id}-") as work:
            for camera in cameras:
                index: int = camera.index
                clip: Path = work / f"cam_{index:02d}.mp4"
                frames: Path = mav0 / f"cam{index}/data"
                with self.timer.stage("encode"):
                    encode_frames_to_mp4(
                        ((frames / name).read_bytes() for name in stereo.filenames[index]),
                        clip,
                        source=FrameSource("png"),
                        fps=nominal_fps(stereo.times_ns),
                    )
                log_camera_node(
                    recording,
                    RIG,
                    index,
                    camera_parameters(camera, name=f"cam{index}"),
                    name=f"cam{index}",
                    kind="grayscale",
                    image_plane_distance=IMAGE_PLANE_DISTANCE,
                    camera_model=camera.camera_model,
                )
                log_video_stream(recording, clip, schema.video_path(RIG, index), times_ns=stereo.times_ns, frame_indices=frame_indices)
        inertial: TimestampedSamples = read_numeric_csv((mav0 / "imu0/data.csv").read_bytes(), num_values=IMU_VALUE_COLUMNS)
        if inertial.times_ns.size == 0:
            raise ValueError(f"{mav0 / 'imu0/data.csv'} has no data rows")
        log_imu(
            recording,
            RIG,
            IMU,
            gyro=ImuChannel(times_ns=inertial.times_ns, values_xyz=inertial.values[:, :3]),
            accel=ImuChannel(times_ns=inertial.times_ns, values_xyz=inertial.values[:, 3:6]),
            name="imu0",
        )
        first_ns: int = int(stereo.times_ns[0])
        self.timer.capture_s = (int(stereo.times_ns[-1]) - first_ns) / 1e9
        writing.send_capture_properties(
            recording,
            identity,
            num_cameras=len(cameras),
            num_frames=int(stereo.times_ns.size),
            start_time_ns=first_ns,
            duration_s=self.timer.capture_s,
            clock_source="the dataset's own cam<N>/imu0/mocap0 nanosecond stamps, unshifted; frame_index: stereo pair number",
            sequence=source.name,
            unpaired_cam0=stereo.unpaired[0],
            unpaired_cam1=stereo.unpaired[1],
        )

    def default_blueprint(self) -> rrb.Blueprint:
        """The rig in 3D beside both cameras, over the IMU plots."""
        return blueprints.rig_blueprint(
            [blueprints.camera_view(f"cam{index}", RIG, index) for index in range(2)],
            rig=RIG,
            run_source=schema.GT_RUN_SOURCE,
            eye_controls=blueprints.follow_eye_controls(FOLLOW.forward, FOLLOW.up),
            plots=blueprints.imu_plots(RIG, IMU),
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Preview card: the rig with no video beside ``cam0``."""
        return blueprints.table_blueprint(
            2,
            rig=RIG,
            run_source=schema.GT_RUN_SOURCE,
            eye_controls=blueprints.follow_eye_controls(FOLLOW.forward, FOLLOW.up),
            front_pane=blueprints.camera_view("cam0", RIG, 0),
        )

    def table_fields(self) -> writing.TableFields:
        """Sequence, length and ground-truth coverage on every card and table row."""
        fields: tuple[writing.TableField, ...] = (
            writing.TableField("property:capture:sequence", "sequence"),
            writing.TableField("property:capture:num_frames", "frames"),
            writing.TableField("property:capture:duration_s", "duration_s"),
            writing.TableField("property:gt:num_spans", "gt spans"),
        )
        return writing.TableFields(cards=fields, table=fields)


def read_mocap(path: Path) -> GtTrajectory:
    """mocap0/data.csv as world_T_imu0, every sample kept (a dropout row's rotation becomes identity, counted).

    Raises:
        ValueError: The file holds no pose.
    """
    return gt_trajectory(read_numeric_csv(path.read_bytes(), num_values=GT_VALUE_COLUMNS))


def write_gt(recording: rr.RecordingStream, poses: GtTrajectory) -> None:
    """The ground-truth layer: world_T_imu0 on the rig node, the path, the world axes, and where mocap covers the run."""
    spans: tuple[MocapSpan, ...] = mocap_spans(poses.times_ns)
    rr.log("/", WORLD_UP_VIEW_COORDINATES[WORLD_UP], static=True, recording=recording)
    log_gt_path(recording, RIG, poses)
    recording.send_property(
        "gt",
        rr.AnyValues(
            num_poses=int(poses.times_ns.size),
            duration_s=poses.duration_s,
            num_sanitized=poses.num_sanitized,
            world_up=WORLD_UP,
            source=GT_SOURCE,
            num_spans=len(spans),
            span_start_ns=[span.start_ns for span in spans],
            span_stop_ns=[span.stop_ns for span in spans],
            span_num_poses=[span.num_poses for span in spans],
        ),
    )


def write_sensor_metadata(recording: rr.RecordingStream, noise: ImuNoise) -> None:
    """imu_config.yaml's noise model on the IMU node, with the time shift the feed reads (0: one clock, nothing shifted)."""
    recording.log(
        schema.imu_path(RIG, IMU),
        ImuCalibration(
            gyro_noise_density=noise.gyroscope_noise_density,
            accel_noise_density=noise.accelerometer_noise_density,
            gyro_bias_random_walk=noise.gyroscope_random_walk,
            accel_bias_random_walk=noise.accelerometer_random_walk,
            rate_hz=noise.update_rate,
            source="TUM-VI dso/imu_config.yaml (Kalibr; white noise x2 and bias random walk x10 over the Allan values, as the file states)",
        ),
        rr.AnyValues(applied_time_shift_ns=0, time_shift_source="TUM-VI camera and IMU stamps share one clock; no shift applied at conversion"),
        static=True,
    )
