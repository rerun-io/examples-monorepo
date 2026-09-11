"""RoboCap: 6-camera + 3-IMU capture rig → one exoego:v2 base-layer rrd per session.

Raw layout on disk (already local; ``download`` only verifies it):

    <root>/0factory-calibration-<device>/imus_cam_*_extrinsic/*-camchain-imucam.yaml
    <root>/<device>_session_<S>/video_dev<D>_session<S>_segment<G>_<camname>.mp4
    <root>/<device>_session_<S>/IMUWriter_dev<0|1|2>_session<S>_segment<G>.db

Sessions vs segments. The recorder rolls every stream to new files every
~10 minutes; a *segment* is that storage artifact, not a semantic unit (the
gap at a roll boundary is ~30 ms). The sequence unit is therefore the
**session**, and ``convert`` merges all of its segments into one recording —
a pure remux per segment, no re-encoding, joined on the shared device clock.
Each segment's video starts with its own IDR frame, so the viewer's decoder
resets cleanly at every boundary. Converted rrds span 16 MB to 22.7 GB. Both
``should_skip`` and mid-write failure are all-or-nothing at session granularity;
that tradeoff is accepted for v1 and should be revisited if resumability matters.

Clocks. Both sensors share one nanosecond device clock (boot-relative, NOT a
Unix epoch — values sit around tens of seconds), offset by a fixed constant:
basalt's RoboCap loader (``src/io/dataset_io_robocap.cpp``) *adds*
``kCameraToImuOffsetNs`` to camera timestamps to land on the IMU clock. We pick
the **raw camera clock** for ``video_time`` — video times are
``comment_us * 1000 + pts_ns`` (the MP4 format-level ``comment`` tag is the
capture start on that device clock, in microseconds), and IMU sample times are
therefore ``t_imu_ns - CAMERA_TO_IMU_OFFSET_NS``. Nothing is resampled: raw
samples land at their native rate. ``video_time`` is deliberately a *duration*
timeline — retagging it as ``timestamp`` would render the boot clock as
1970-01-01T00:00:25. Segments recorded within one boot share the axis, which
is exactly what makes the merged session line up without any retiming.

Calibration comes from simplecv's RoboCap ego loader; the private helpers are
imported deliberately (the donor is scheduled for deletion once the dataset
lives here).
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import av
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from beartype.roar import BeartypeException
from jaxtyping import Float64, Int64
from numpy import ndarray
from simplecv.camera_parameters import Fisheye62Parameters
from simplecv.data.ego.robocap_ego import (
    CAM_TO_CALIB_INFO,
    CAMERA_DISPLAY_ORDER,
    VIDEO_SUFFIX_TO_CAM_NAME,
    KalibrCamWithExtrinsic,
    _kalibr_cam_to_fisheye62,
    _load_kalibr_camchain_imucam,
)

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import ImuChannel, log_camera_node, log_imu, log_rig_node, log_video_stream

GYRO_SCALE: float = 0.000266316
"""Raw gyro LSB → rad/s; measured in the basalt fork (``dataset_io_robocap.cpp``)."""
ACCEL_SCALE: float = 0.001197101
"""Raw accel LSB → m/s^2; measured in the basalt fork (``dataset_io_robocap.cpp``)."""
CAMERA_TO_IMU_OFFSET_NS: int = 14_902_432
"""basalt's ``kCameraToImuOffsetNs``: camera_ns + offset = imu_ns, so imu_ns - offset = camera_ns."""
IMAGE_PLANE_DISTANCE: float = 0.025
"""Frustum length in metres, matching ``RobocapEgoSequence.image_plane_distance``."""
RIG: int = 0
"""RoboCap is one rig; the whole device is ``rig_00``."""
RIG_REFERENCE: str = "imu_00"
"""The rig frame *is* dev0's IMU frame: every camera's Kalibr extrinsic is a ``T_imu_cam``,
so the rig's reference sensor is the IMU node, not a camera."""
IMU_DEVICE: int = 0
"""v1 logs the middle IMU (``dev0``) only. TODO(dataforge): also emit dev1/dev2."""
RUN_SOURCE: str = "basalt"
"""Processing source of the pose layer the blueprints already lay out; nothing writes it yet."""
SESSION_DIR_RE: re.Pattern[str] = re.compile(r"^(?P<device>[0-9a-f]+)_session_(?P<session>\d+)$")
"""Session directory names; the ``-old`` duplicates deliberately do not match."""
MESH_RELATIVE_PATH: str = "robocap-mesh/3DModel.glb"
"""Cap scan location under the corpus root; the mesh is optional (skipped when unreadable)."""
MESH_TRANSLATION: list[float] = [0.0, -0.15, 0.025]
"""Hand-tuned alignment of the Y-up photogrammetry scan into the dev0 IMU (rig) frame,
carried over from robocap-slam's visualization; eyeballed, not a CAD extrinsic."""
MESH_MAT3X3: list[list[float]] = [
    [0.17364818, 0.0, 0.98480775],
    [-0.98480775, 0.0, 0.17364818],
    [0.0, -1.0, 0.0],
]
"""Rz(-80deg) @ Rx(-90deg), the rotation half of the same hand-tuned alignment."""
VIDEO_NAME_RE: re.Pattern[str] = re.compile(r"^video_dev(?P<device>\d+)_session(?P<session>\d+)_segment(?P<segment>\d+)_(?P<camname>[a-z-]+)$")
"""Per-camera MP4 stem, e.g. ``video_dev0_session1_segment1_right-eye``."""


@dataclass(frozen=True, slots=True)
class RobocapSource:
    """One session as discovery found it; ``convert`` needs nothing else from the tree."""

    session_dir: Path
    """``<root>/<device>_session_<S>`` directory holding the videos and IMU dbs."""
    device: str
    """Capture-device id, which also names the factory calibration directory."""
    session: int
    """Session number ``S`` (part of every filename in the directory)."""
    segments: tuple[int, ...]
    """Segment numbers present on disk, ascending; ``convert`` merges them all."""


@dataclass
class RobocapConfig(DataforgeDatasetConfig):
    """RoboCap capture-rig recordings (6 fisheye cameras, 3 IMUs, no labels)."""

    command: ClassVar[str] = "robocap"
    """Registry key, catalog dataset name, and identity ``dataset`` part."""

    _target: type = field(default_factory=lambda: RobocapDataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = Path("/mnt/nas/datasets/robocap")
    """Corpus root holding ``<device>_session_<N>/`` and the factory calibration."""
    mesh_path: Path | None = None
    """Cap mesh glb; defaults to ``<root>/robocap-mesh/3DModel.glb`` when None."""


def read_imu_channel(database: sqlite3.Connection, table: str, scale: float) -> ImuChannel:
    """Read one raw IMU table and map it onto the camera clock.

    Args:
        database: Read-only connection to an ``IMUWriter_dev*.db``.
        table: ``gyro_data`` or ``acc_data``; columns are ``(id, imuid_, x, y, z, timestamp)``.
        scale: LSB scale factor (``GYRO_SCALE`` or ``ACCEL_SCALE``).

    Returns:
        The channel's samples, sorted by time, on the ``video_time`` clock.
    """
    rows: list[tuple[int, int, int, int]] = database.execute(f"SELECT x, y, z, timestamp FROM {table} ORDER BY timestamp").fetchall()
    if not rows:
        return ImuChannel(times_ns=np.zeros(0, dtype=np.int64), values_xyz=np.zeros((0, 3), dtype=np.float64))
    raw: Int64[ndarray, "n_samples 4"] = np.asarray(rows, dtype=np.int64)
    times_ns: Int64[ndarray, "n_samples"] = raw[:, 3] - CAMERA_TO_IMU_OFFSET_NS
    values_xyz: Float64[ndarray, "n_samples 3"] = raw[:, :3].astype(np.float64) * scale
    return ImuChannel(times_ns=times_ns, values_xyz=values_xyz)


def read_imu_database(db_path: Path) -> tuple[ImuChannel, ImuChannel] | None:
    """Read ``(gyro, accel)`` from one IMU db, or ``None`` when the file is unusable.

    Some segments ship a malformed db (session_1's ``dev2``), which must not
    abort a conversion — the stream is simply absent from the recording.

    Args:
        db_path: Path to an ``IMUWriter_dev<N>_session<S>_segment<G>.db``.

    Returns:
        The gyro and accel channels, or ``None`` if SQLite refused the file.
    """
    try:
        # contextlib.closing because sqlite3's own context manager only ends the
        # transaction; it leaves the connection (and its fd) open until GC.
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as database:
            gyro: ImuChannel = read_imu_channel(database, "gyro_data", GYRO_SCALE)
            accel: ImuChannel = read_imu_channel(database, "acc_data", ACCEL_SCALE)
    except BeartypeException:
        raise
    except sqlite3.DatabaseError as error:
        print(f"  warning: skipping IMU {db_path.name}: {error}")
        return None
    return gyro, accel


def video_epoch_ns(video_path: Path) -> int:
    """Start time of an MP4 on the device clock, from its format-level ``comment`` tag.

    RoboCap writes the capture start (boot-relative device clock, microseconds)
    into the container ``comment`` metadata; sample PTS are offsets from it.

    Args:
        video_path: Path to a RoboCap MP4.

    Returns:
        The capture start in nanoseconds on the camera clock.
    """
    with av.open(str(video_path)) as container:
        comment: str | None = container.metadata.get("comment")
        if comment is None:
            comment = next((stream.metadata.get("comment") for stream in container.streams.video if stream.metadata.get("comment")), None)
    if comment is None:
        raise ValueError(f"Missing absolute timestamp comment metadata in {video_path}")
    return int(comment) * 1_000


def follow_eye_controls() -> rrb.EyeControls3D:
    """First-person eye controls aligned with RoboCap's rig frame.

    Hand-placed, not derived: RoboCap's six cameras do not make a front pair
    whose baseline names an up axis, so the numbers below come from the cap mesh
    alignment instead.
    """
    return blueprints.eye_controls_from_pose(
        # Rig-frame coordinates: the dev0 IMU frame's -Z is the wearer's up
        # (the cap mesh alignment maps scan-up there).
        position=(0.8, -0.8, -0.6),
        # The rig origin coincides with the front-stereo cameras (left_front is
        # ~4 cm away), so this lines the view up with where the wearer faces.
        look_target=(0.0, 0.0, 0.0),
        eye_up=(0.0, 0.0, -1.0),
    )


def build_blueprint(camera_names: list[str]) -> rrb.Blueprint:
    """Default layout: 3D rig + a grid of camera panes over gyro/accel plots.

    Mirrors basalt's ``basalt_vio_blueprint.py`` layout, on exoego:v2 paths.

    Args:
        camera_names: Full canonical camera labels in ``cam_00..cam_NN`` order;
            callers pass ``list(CAMERA_DISPLAY_ORDER)`` so panes stay stable.

    Returns:
        The blueprint embedded in every RoboCap base-layer rrd.
    """
    # TODO(dataforge): once dev1/dev2 are emitted, fan the plots out to one pane pair per IMU.
    return blueprints.rig_blueprint(
        [blueprints.camera_view(name, RIG, index) for index, name in enumerate(camera_names)],
        rig=RIG,
        run_source=RUN_SOURCE,
        eye_controls=follow_eye_controls(),
        plots=[
            blueprints.sensor_plot(name, schema.imu_path(RIG, IMU_DEVICE), contents)
            for name, contents in (
                ("Gyroscope", schema.gyro_path(RIG, IMU_DEVICE)),
                ("Accelerometer", schema.accel_path(RIG, IMU_DEVICE)),
            )
        ],
    )


def build_table_blueprint(camera_names: list[str]) -> rrb.Blueprint:
    """Segment-table preview card: follow-framed 3D (full trajectory, all frusta,
    NO video textures) beside the single front-stereo video pane.

    Every visible table row renders through this at once (ARKitScenes profiled
    ~15 cards saturating 12 cores when they decoded all).

    Args:
        camera_names: Full canonical camera labels in ``cam_00..cam_NN`` order.
    """
    return blueprints.table_blueprint(
        len(camera_names),
        rig=RIG,
        run_source=RUN_SOURCE,
        eye_controls=follow_eye_controls(),
        front_pane=blueprints.camera_view(CAMERA_DISPLAY_ORDER[0], RIG, 0),
    )


class RobocapDataset(DataforgeDataset[RobocapConfig, RobocapSource]):
    """Converts RoboCap sessions into exoego:v2 base-layer recordings."""

    def default_blueprint(self) -> rrb.Blueprint:
        """Corpus-wide layout: every RoboCap rig has the same six cameras."""
        return build_blueprint(list(CAMERA_DISPLAY_ORDER))

    def table_blueprint(self) -> rrb.Blueprint:
        """Cheap preview card for the dataset's segment table."""
        return build_table_blueprint(list(CAMERA_DISPLAY_ORDER))

    # ── raw-tree discovery ────────────────────────────────────────────────
    def session_dirs(self) -> list[Path]:
        """Every ``<device>_session_<N>`` directory, ``-old`` duplicates excluded."""
        return sorted(path for path in self.config.root.glob("*_session_*") if path.is_dir() and SESSION_DIR_RE.match(path.name))

    def download(self) -> None:
        """Verify the local corpus; RoboCap has no upstream fetch."""
        missing: list[str] = transports.local_verify(self.config.root, required=["*_session_*", "0factory-calibration-*"])
        if missing:
            raise FileNotFoundError(f"RoboCap corpus at {self.config.root} is missing: {', '.join(missing)}")
        session_dirs: list[Path] = self.session_dirs()
        discovered: list[tuple[SequenceIdentity, RobocapSource]] = self.discover()
        num_segments: int = sum(len(source.segments) for _, source in discovered)
        print(
            f"robocap: {self.config.root} — {len(session_dirs)} session dirs, {len(discovered)} with videos "
            f"({num_segments} segments), calibration present"
        )
        discovered_dirs: set[Path] = {source.session_dir for _, source in discovered}
        video_less_session_dirs: list[Path] = sorted(set(session_dirs) - discovered_dirs)
        if video_less_session_dirs:
            print(f"  warning: session directories contain no videos: {', '.join(path.name for path in video_less_session_dirs)}")

    def discover(self) -> list[tuple[SequenceIdentity, RobocapSource]]:
        """Pair every ``(device, session)`` on disk with its directory and the segments inside."""
        found: dict[tuple[str, int], tuple[Path, set[int]]] = {}
        for session_dir in self.session_dirs():
            match: re.Match[str] | None = SESSION_DIR_RE.match(session_dir.name)
            if match is None:
                continue
            device: str = match["device"]
            session: int = int(match["session"])
            for video_path in session_dir.glob(f"video_dev*_session{session}_segment*_*.mp4"):
                video_match: re.Match[str] | None = VIDEO_NAME_RE.match(video_path.stem)
                if video_match is not None:
                    found.setdefault((device, session), (session_dir, set()))[1].add(int(video_match["segment"]))
        return [
            (
                # Zero-padded so lexicographic ordering (catalog tables, file
                # listings) matches capture order for any conceivable corpus size.
                SequenceIdentity(dataset=self.config.name, parts=(device, f"s{session:08d}")),
                RobocapSource(session_dir=session_dir, device=device, session=session, segments=tuple(sorted(segments))),
            )
            for (device, session), (session_dir, segments) in sorted(found.items())
        ]

    def _videos(self, session_dir: Path, session: int, segment: int) -> dict[str, Path]:
        """Map canonical camera name → MP4 path, ordered by ``CAMERA_DISPLAY_ORDER``."""
        by_name: dict[str, Path] = {}
        for video_path in session_dir.glob(f"video_dev*_session{session}_segment{segment}_*.mp4"):
            match: re.Match[str] | None = VIDEO_NAME_RE.match(video_path.stem)
            if match is None:
                continue
            cam_name: str | None = VIDEO_SUFFIX_TO_CAM_NAME.get(match["camname"])
            if cam_name is not None:
                by_name[cam_name] = video_path
        return {name: by_name[name] for name in CAMERA_DISPLAY_ORDER if name in by_name}

    def _calibration(self, device: str) -> dict[str, Fisheye62Parameters]:
        """Load the factory Kalibr calibration, keyed by canonical camera name."""
        calib_dir: Path = self.config.root / f"0factory-calibration-{device}"
        if not calib_dir.is_dir():
            raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")
        cameras: dict[str, Fisheye62Parameters] = {}
        for cam_name, (calib_folder, cam_index) in CAM_TO_CALIB_INFO.items():
            imucam_files: list[Path] = sorted((calib_dir / calib_folder).glob("*-camchain-imucam.yaml"))
            if not imucam_files:
                continue
            cam_dict: dict[str, KalibrCamWithExtrinsic] = _load_kalibr_camchain_imucam(imucam_files[0])
            cam_data: KalibrCamWithExtrinsic | None = cam_dict.get(f"cam{cam_index}")
            if cam_data is not None:
                cameras[cam_name] = _kalibr_cam_to_fisheye62(cam_data, name=cam_name)
        return cameras

    # ── conversion ────────────────────────────────────────────────────────
    def convert(self, identity: SequenceIdentity, source: RobocapSource, *, force: bool) -> Path:
        """Write one session's base-layer rrd: every segment merged on the shared device clock.

        A stream that cannot be placed on the clock (a missing ``comment`` epoch
        tag, a defect in the corpus) degrades to a warning and a gap in that
        camera's stream — one bad file must not lose the rest of a session.
        """
        target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
        if writing.should_skip(target, force=force):
            print(f"skip {identity.sequence_key} → {target}")
            return target

        cameras: dict[str, Fisheye62Parameters] = self._calibration(source.device)
        # Every MP4's container comment is parsed before the recording opens: each
        # stream is retimed by its own epoch, and the earliest doubles as start_time_ns.
        segment_videos: dict[int, dict[str, Path]] = {}
        segment_epochs: dict[int, dict[str, int]] = {}
        for segment in source.segments:
            videos: dict[str, Path] = self._videos(source.session_dir, source.session, segment)
            epochs: dict[str, int] = {}
            for cam_name, video_path in videos.items():
                try:
                    epochs[cam_name] = video_epoch_ns(video_path)
                except ValueError as error:
                    print(f"  warning: segment {segment} {cam_name}: {error}; stream gap")
            segment_videos[segment] = videos
            segment_epochs[segment] = epochs
        camera_names: list[str] = [
            name for name in CAMERA_DISPLAY_ORDER if name in cameras and any(name in epochs for epochs in segment_epochs.values())
        ]
        if not camera_names:
            raise ValueError(
                f"no camera stream in {source.session_dir} carries a comment epoch tag "
                f"({len(cameras)} calibrated cameras, {len(source.segments)} segments scanned)"
            )
        segment_starts_ns: dict[int, int] = {segment: min(epochs.values()) for segment, epochs in segment_epochs.items() if epochs}
        ordered_segment_starts_ns: list[int] = [segment_starts_ns[segment] for segment in sorted(segment_starts_ns)]

        with writing.atomic_recording(
            target,
            recording_id=identity.recording_id,
            default_blueprint=self.default_blueprint(),
        ) as recording:
            # Deliberately NO ViewCoordinates at "/": the pose layer owns the root
            # ViewCoordinates (its world is gravity-aligned Z-up).
            log_rig_node(recording, RIG, reference=RIG_REFERENCE, num_cameras=len(camera_names), name="robocap", kind="ego")
            self._log_mesh(recording)

            frames_per_camera: dict[str, int] = dict.fromkeys(camera_names, 0)
            for cam_name in camera_names:
                index: int = CAMERA_DISPLAY_ORDER.index(cam_name)
                log_camera_node(
                    recording,
                    RIG,
                    index,
                    cameras[cam_name],
                    name=cam_name,
                    kind="grayscale",
                    image_plane_distance=IMAGE_PLANE_DISTANCE,
                )
                for segment in source.segments:
                    if cam_name not in segment_epochs[segment]:
                        continue
                    frames_per_camera[cam_name] += log_video_stream(
                        recording,
                        segment_videos[segment][cam_name],
                        schema.video_path(RIG, index),
                        shift_ns=segment_epochs[segment][cam_name],
                    )

            self._log_imu(recording, source)

            # start_time_ns makes the device-clock origin queryable without scanning chunks
            # (e.g. for a consumer that wants a zero-based axis or cross-segment alignment).
            writing.send_capture_properties(
                recording,
                identity,
                num_cameras=len(camera_names),
                num_frames=max(frames_per_camera.values()),
                start_time_ns=min(segment_starts_ns.values()),
                num_segments=len(source.segments),
                num_timed_segments=len(segment_starts_ns),
                segment_starts_ns=ordered_segment_starts_ns,
            )

        print(f"done {identity.sequence_key} → {target} ({len(camera_names)} cameras, {len(source.segments)} segments, {max(frames_per_camera.values())} frames)")
        return target

    def _log_mesh(self, recording: rr.RecordingStream) -> None:
        """Log the textured cap scan as a static child of the rig, if the asset is readable.

        The mesh makes the rig legible as a physical object (the pinholes attach to
        its brim) and rides any temporal ``world_T_rig`` a pose layer adds later.
        """
        mesh_path: Path = self.config.mesh_path if self.config.mesh_path is not None else self.config.root / MESH_RELATIVE_PATH
        if not os.access(mesh_path, os.R_OK):
            print(f"  warning: cap mesh not readable, skipping: {mesh_path}")
            return
        mesh_entity: str = f"{schema.rig_path(RIG)}/mesh"
        rr.log(mesh_entity, rr.Transform3D(translation=MESH_TRANSLATION, mat3x3=MESH_MAT3X3), static=True, recording=recording)
        rr.log(mesh_entity, rr.Asset3D(path=mesh_path), static=True, recording=recording)

    def _log_imu(self, recording: rr.RecordingStream, source: RobocapSource) -> None:
        """Log the dev0 IMU's raw gyro/accel samples columnar on ``video_time``.

        Segments' dbs concatenate directly: their timestamps are already on the
        shared device clock, so the merged channels stay sorted.
        """
        gyro_parts: list[ImuChannel] = []
        accel_parts: list[ImuChannel] = []
        for segment in source.segments:
            db_path: Path = source.session_dir / f"IMUWriter_dev{IMU_DEVICE}_session{source.session}_segment{segment}.db"
            if not db_path.is_file():
                print(f"  warning: no IMU database at {db_path}")
                continue
            channels: tuple[ImuChannel, ImuChannel] | None = read_imu_database(db_path)
            if channels is not None:
                gyro_parts.append(channels[0])
                accel_parts.append(channels[1])
        if not gyro_parts:
            return
        gyro: ImuChannel = ImuChannel(
            times_ns=np.concatenate([part.times_ns for part in gyro_parts]),
            values_xyz=np.concatenate([part.values_xyz for part in gyro_parts]),
        )
        accel: ImuChannel = ImuChannel(
            times_ns=np.concatenate([part.times_ns for part in accel_parts]),
            values_xyz=np.concatenate([part.values_xyz for part in accel_parts]),
        )
        log_imu(recording, RIG, IMU_DEVICE, gyro=gyro, accel=accel, name=f"dev{IMU_DEVICE}")
