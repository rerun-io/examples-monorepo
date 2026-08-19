"""RoboCap: 6-camera + 3-IMU capture rig → one exoego:v2 base-layer rrd per segment.

Raw layout on disk (already local; ``download`` only verifies it):

    <root>/0factory-calibration-<device>/imus_cam_*_extrinsic/*-camchain-imucam.yaml
    <root>/<device>_session_<S>/video_dev<D>_session<S>_segment<G>_<camname>.mp4
    <root>/<device>_session_<S>/IMUWriter_dev<0|1|2>_session<S>_segment<G>.db

Clocks. Both sensors share one nanosecond device clock, offset by a fixed
constant: basalt's RoboCap loader (``src/io/dataset_io_robocap.cpp``) *adds*
``kCameraToImuOffsetNs`` to camera timestamps to land on the IMU clock. We pick
the **raw camera clock** for ``video_time`` — video times are
``comment_us * 1000 + pts_ns`` (the MP4 format-level ``comment`` tag is an
absolute epoch in microseconds), and IMU sample times are therefore
``t_imu_ns - CAMERA_TO_IMU_OFFSET_NS``. Nothing is resampled: raw samples land
at their native rate.

Calibration comes from simplecv's RoboCap ego loader; the private helpers are
imported deliberately (the donor is scheduled for deletion once the dataset
lives here).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
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
from simplecv.rerun_log_utils import log_pinhole

from dataforge import paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity

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
IMU_DEVICE: int = 0
"""v1 logs the middle IMU (``dev0``) only. TODO(dataforge): also emit dev1/dev2."""
SESSION_DIR_RE: re.Pattern[str] = re.compile(r"^(?P<device>[0-9a-f]+)_session_(?P<session>\d+)$")
"""Session directory names; the ``-old`` duplicates deliberately do not match."""
IDENTITY_TRANSFORM: rr.Transform3D = rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3, dtype=np.float32))
"""Explicit identity pose; an argument-less ``Transform3D`` logs no components at all."""
VIDEO_NAME_RE: re.Pattern[str] = re.compile(r"^video_dev(?P<device>\d+)_session(?P<session>\d+)_segment(?P<segment>\d+)_(?P<camname>[a-z-]+)$")
"""Per-camera MP4 stem, e.g. ``video_dev0_session1_segment1_right-eye``."""


@dataclass(frozen=True, slots=True)
class ImuSamples:
    """One raw IMU channel (gyro or accel) at its native sample rate."""

    times_ns: Int64[ndarray, "n_samples"]
    """Sample times on the ``video_time`` camera clock, in nanoseconds."""
    values_xyz: Float64[ndarray, "n_samples 3"]
    """Scaled samples (rad/s for gyro, m/s^2 for accel)."""


@dataclass
class RobocapConfig(DataforgeDatasetConfig):
    """RoboCap capture-rig recordings (6 fisheye cameras, 3 IMUs, no labels)."""

    _target: type = field(default_factory=lambda: RobocapDataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = Path("/mnt/nas/datasets/robocap")
    """Corpus root holding ``<device>_session_<N>/`` and the factory calibration."""


def read_imu_channel(database: sqlite3.Connection, table: str, scale: float) -> ImuSamples:
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
        return ImuSamples(times_ns=np.zeros(0, dtype=np.int64), values_xyz=np.zeros((0, 3), dtype=np.float64))
    raw: Int64[ndarray, "n_samples 4"] = np.asarray(rows, dtype=np.int64)
    times_ns: Int64[ndarray, "n_samples"] = raw[:, 3] - CAMERA_TO_IMU_OFFSET_NS
    values_xyz: Float64[ndarray, "n_samples 3"] = raw[:, :3].astype(np.float64) * scale
    return ImuSamples(times_ns=times_ns, values_xyz=values_xyz)


def read_imu_database(db_path: Path) -> tuple[ImuSamples, ImuSamples] | None:
    """Read ``(gyro, accel)`` from one IMU db, or ``None`` when the file is unusable.

    Some segments ship a malformed db (session_1's ``dev2``), which must not
    abort a conversion — the stream is simply absent from the recording.

    Args:
        db_path: Path to an ``IMUWriter_dev<N>_session<S>_segment<G>.db``.

    Returns:
        The gyro and accel channels, or ``None`` if SQLite refused the file.
    """
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as database:
            gyro: ImuSamples = read_imu_channel(database, "gyro_data", GYRO_SCALE)
            accel: ImuSamples = read_imu_channel(database, "acc_data", ACCEL_SCALE)
    except BeartypeException:
        raise
    except sqlite3.DatabaseError as error:
        print(f"  warning: skipping IMU {db_path.name}: {error}")
        return None
    return gyro, accel


def video_epoch_ns(video_path: Path) -> int:
    """Absolute start time of an MP4 from its format-level ``comment`` tag.

    RoboCap writes the capture epoch in microseconds into the container
    ``comment`` metadata; sample PTS are offsets from it.

    Args:
        video_path: Path to a RoboCap MP4.

    Returns:
        The capture epoch in nanoseconds on the camera clock.
    """
    with av.open(str(video_path)) as container:
        comment: str | None = container.metadata.get("comment")
        if comment is None:
            comment = next((stream.metadata.get("comment") for stream in container.streams.video if stream.metadata.get("comment")), None)
    if comment is None:
        raise ValueError(f"Missing absolute timestamp comment metadata in {video_path}")
    return int(comment) * 1_000


def build_blueprint(camera_names: list[str]) -> rrb.Blueprint:
    """Default layout: 3D rig + a grid of camera panes over gyro/accel plots.

    Mirrors basalt's ``basalt_vio_blueprint.py`` layout, on exoego:v2 paths.

    Args:
        camera_names: Human camera labels in ``cam_00..cam_NN`` order.

    Returns:
        The blueprint embedded in every RoboCap base-layer rrd.
    """
    camera_views: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(name=name, origin=schema.pinhole_path(RIG, index), contents=f"{schema.pinhole_path(RIG, index)}/**")
        for index, name in enumerate(camera_names)
    ]
    # TODO(dataforge): once dev1/dev2 are emitted, fan the plots out to one pane pair per IMU.
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Rig", origin="/", line_grid=True),
                rrb.Grid(*camera_views, grid_columns=3, name="Synchronized cameras"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Gyroscope",
                    origin=schema.imu_path(RIG, IMU_DEVICE),
                    contents=schema.gyro_path(RIG, IMU_DEVICE),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    name="Accelerometer",
                    origin=schema.imu_path(RIG, IMU_DEVICE),
                    contents=schema.accel_path(RIG, IMU_DEVICE),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
            ),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )


class RobocapDataset(DataforgeDataset[RobocapConfig]):
    """Converts RoboCap segments into exoego:v2 base-layer recordings."""

    def default_blueprint(self) -> rrb.Blueprint:
        """Corpus-wide layout: every RoboCap rig has the same six cameras."""
        return build_blueprint(list(CAMERA_DISPLAY_ORDER))

    # ── raw-tree discovery ────────────────────────────────────────────────
    def session_dirs(self) -> list[Path]:
        """Every ``<device>_session_<N>`` directory, ``-old`` duplicates excluded."""
        return sorted(path for path in self.config.root.glob("*_session_*") if path.is_dir() and SESSION_DIR_RE.match(path.name))

    def download(self) -> None:
        """Verify the local corpus; RoboCap has no upstream fetch."""
        missing: list[str] = transports.local_verify(self.config.root, required=["*_session_*", "0factory-calibration-*"])
        if missing:
            raise FileNotFoundError(f"RoboCap corpus at {self.config.root} is missing: {', '.join(missing)}")
        sessions: list[Path] = self.session_dirs()
        print(f"robocap: {self.config.root} — {len(sessions)} sessions, {len(self.sequences())} segments, calibration present")

    def sequences(self) -> list[SequenceIdentity]:
        """Discover every ``(device, session, segment)`` triple on disk."""
        triples: set[tuple[str, int, int]] = set()
        for session_dir in self.session_dirs():
            match: re.Match[str] | None = SESSION_DIR_RE.match(session_dir.name)
            if match is None:
                continue
            device: str = match["device"]
            session: int = int(match["session"])
            for video_path in session_dir.glob(f"video_dev*_session{session}_segment*_*.mp4"):
                video_match: re.Match[str] | None = VIDEO_NAME_RE.match(video_path.stem)
                if video_match is not None:
                    triples.add((device, session, int(video_match["segment"])))
        return [
            SequenceIdentity(dataset="robocap", parts=(device, f"s{session}", f"seg{segment}")) for device, session, segment in sorted(triples)
        ]

    def _locate(self, identity: SequenceIdentity) -> tuple[Path, str, int, int]:
        """Resolve an identity back to ``(session_dir, device, session, segment)``."""
        if len(identity.parts) != 3:
            raise ValueError(f"RoboCap identities have three parts, got {identity.sequence_key!r}")
        device: str = identity.parts[0]
        session: int = int(identity.parts[1].removeprefix("s"))
        segment: int = int(identity.parts[2].removeprefix("seg"))
        session_dir: Path = self.config.root / f"{device}_session_{session}"
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        return session_dir, device, session, segment

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
    def convert(self, identity: SequenceIdentity, *, force: bool) -> Path:
        """Write one segment's base-layer rrd (cameras + video + dev0 IMU)."""
        target: Path = paths.rrd_path(paths.output_root(), layer="base", identity=identity)
        if writing.should_skip(target, force=force):
            print(f"skip {identity.sequence_key} → {target}")
            return target

        session_dir, device, session, segment = self._locate(identity)
        videos: dict[str, Path] = self._videos(session_dir, session, segment)
        if not videos:
            raise FileNotFoundError(f"No videos for {identity.sequence_key} in {session_dir}")
        cameras: dict[str, Fisheye62Parameters] = self._calibration(device)
        camera_names: list[str] = [name for name in videos if name in cameras]
        if not camera_names:
            raise FileNotFoundError(f"No calibrated cameras for {identity.sequence_key}")

        with writing.atomic_recording(
            target,
            application_id="dataforge",
            recording_id=identity.recording_id,
            default_blueprint=build_blueprint(camera_names),
        ) as recording:
            rr.log("/", rr.ViewCoordinates.RDF, static=True, recording=recording)
            # simplecv's RoboCap loader treats the dev0 IMU frame as world, so world_T_rig is
            # identity and static for v1. TODO(dataforge): log the VIO trajectory as a temporal
            # world_T_rig once a pose layer exists.
            rr.log(schema.rig_path(RIG), IDENTITY_TRANSFORM, static=True, recording=recording)
            rr.log(
                schema.rig_path(RIG),
                rr.AnyValues(schema_version=schema.EXOEGO_SCHEMA_VERSION, reference="cam_00", num_cameras=len(camera_names)),
                static=True,
                recording=recording,
            )

            num_frames: int = 0
            for index, cam_name in enumerate(camera_names):
                rr.log(
                    schema.cam_path(RIG, index),
                    rr.AnyValues(name=cam_name, kind="grayscale"),
                    static=True,
                    recording=recording,
                )
                log_pinhole(
                    cameras[cam_name],
                    cam_log_path=Path(schema.cam_path(RIG, index)),
                    image_plane_distance=IMAGE_PLANE_DISTANCE,
                    static=True,
                    recording=recording,
                )
                num_frames = max(num_frames, self._log_video(recording, videos[cam_name], schema.video_path(RIG, index)))

            self._log_imu(recording, session_dir, session, segment)

            recording.send_recording_name(identity.recording_id)
            recording.send_property(
                "capture",
                rr.AnyValues(schema=schema.DATAFORGE_SCHEMA_VERSION, num_frames=num_frames, num_cameras=len(camera_names)),
            )
            recording.send_property("convert", rr.AnyValues(version="1"))

        print(f"done {identity.sequence_key} → {target} ({len(camera_names)} cameras, {num_frames} frames)")
        return target

    def _log_video(self, recording: rr.RecordingStream, video_path: Path, entity_path: str) -> int:
        """Remux one MP4 as a ``VideoStream``, retimed onto the camera clock.

        ``Mp4Reader`` emits chunk timestamps as raw PTS (nanoseconds from the
        start of the file), so every indexed chunk is shifted by the file's
        absolute epoch before it reaches the recording.

        Args:
            recording: Destination recording stream.
            video_path: RoboCap MP4 to remux (no decode, no re-encode).
            entity_path: ``.../pinhole/video`` entity to log under.

        Returns:
            Number of video samples written.
        """
        epoch_ns: int = video_epoch_ns(video_path)
        sample_count: int = 0

        def retime(chunk: rr.experimental.Chunk) -> list[rr.experimental.Chunk]:
            nonlocal sample_count
            if schema.TIMELINE not in chunk.timeline_names:
                return [chunk]  # the static codec chunk carries no index
            record_batch: pa.RecordBatch = chunk.to_record_batch()
            index: int = record_batch.schema.get_field_index(schema.TIMELINE)
            shifted: pa.Array = pa.array(np.asarray(record_batch.column(index).cast(pa.int64())) + epoch_ns, type=pa.duration("ns"))
            sample_count += record_batch.num_rows
            retimed: pa.RecordBatch = record_batch.set_column(index, record_batch.schema.field(index), shifted)
            return rr.experimental.Chunk.from_record_batch(retimed, index=schema.TIMELINE, entity_path=chunk.entity_path)

        reader: rr.experimental.Mp4Reader = rr.experimental.Mp4Reader(
            video_path,
            mode="stream",
            entity_path=entity_path,
            timeline_name=schema.TIMELINE,
        )
        recording.send_chunks(reader.stream().flat_map(retime))
        return sample_count

    def _log_imu(self, recording: rr.RecordingStream, session_dir: Path, session: int, segment: int) -> None:
        """Log the dev0 IMU's raw gyro/accel samples columnar on ``video_time``."""
        db_path: Path = session_dir / f"IMUWriter_dev{IMU_DEVICE}_session{session}_segment{segment}.db"
        if not db_path.is_file():
            print(f"  warning: no IMU database at {db_path}")
            return
        channels: tuple[ImuSamples, ImuSamples] | None = read_imu_database(db_path)
        if channels is None:
            return
        for samples, entity_path in ((channels[0], schema.gyro_path(RIG, IMU_DEVICE)), (channels[1], schema.accel_path(RIG, IMU_DEVICE))):
            if samples.times_ns.size == 0:
                continue
            rr.send_columns(
                entity_path,
                indexes=[rr.TimeColumn(schema.TIMELINE, duration=samples.times_ns.astype("timedelta64[ns]"))],
                columns=rr.Scalars.columns(scalars=samples.values_xyz),
                recording=recording,
            )
        rr.log(schema.imu_path(RIG, IMU_DEVICE), IDENTITY_TRANSFORM, static=True, recording=recording)
        rr.log(
            schema.imu_path(RIG, IMU_DEVICE),
            rr.AnyValues(name=f"dev{IMU_DEVICE}", kind="imu"),
            static=True,
            recording=recording,
        )
