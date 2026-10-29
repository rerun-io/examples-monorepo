"""Gen-HumanWholeBody base layer and sidecar: one MCAP's cameras, rig, IMU and frames, read through lenses.

The base layer is the only one that reads the MCAP. Everything a derived layer needs besides it is the
sidecar: the MCAP's non-camera topics, still protobuf-decoded and on the MCAP clock, in an rrd that
nothing registers (``paths.sidecar_path``). Each later layer reads the sidecar through the same
``EpisodeFile`` the base reads the MCAP through.
"""

from __future__ import annotations

import threading
from _thread import LockType
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from jaxtyping import Float64, Int64
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters

from dataforge import schema, writing
from dataforge.datasets.genrobot_source import (
    DOMAIN_LEVELS,
    DUPLICATE_TOPICS,
    EEF_POSE_TOPIC,
    FRAME_VALIDITY_TOPIC,
    IMU_TOPIC,
    NUM_CAMERAS,
    REFERENCE_CAMERA,
    ROBOT_INFO_TOPIC,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    STANDARD_GRAVITY_M_S2,
    Annotation,
    CameraCalibration,
    DoubleSphere,
    Episode,
    EpisodeClock,
    EpisodeFile,
    InvalidRange,
    RobotInfo,
    SystemInfo,
    camera_topic,
    clear_sps_constraint_set3,
    is_keyframe,
    messages,
    optional_message,
    resize_calibration_topic,
    rig_T_cam,
    rig_T_imu,
    struct_rows,
    topic_pattern,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    FRAME_INDEX_FIELD,
    ChunkBatcher,
    ImuChannel,
    annotation_context,
    log_camera_node,
    log_camera_source,
    log_imu,
    log_rig_node,
    time_column,
)

RIG: int = 0
"""The DAS-Ego is the only device, so it is ``rig_00``."""
IMU: int = 0
"""Its one IMU."""
VIDEO_CODEC: str = "h264"
"""The cameras' own bitstream, remuxed with one SPS flag cleared (``clear_sps_constraint_set3``)."""
IMAGE_PLANE_DISTANCE_M: float = 0.05
"""Frustum length in the 3D view; the rig is a head-sized device."""
CLOCK_SOURCE: str = "mcap log_time (= publish_time = header.timestamp) minus the file's first message; frame_index = camera2 frame"
"""``property:capture:clock_source``."""
SIDECAR_NAME: str = "source.rrd"
"""The sidecar's file name under ``sidecars/<recording_id>/``."""
SIDECAR_DECODERS: tuple[str, ...] = ("protobuf", "schema", "stats", "metadata", "attachments", "recording_info")
"""The messages, each topic's protobuf schema, and the file's statistics (its first stamp is ``video_time``'s origin)."""
SIDECAR_APPLICATION_ID: str = "dataforge-genrobot-sidecar"
"""Its own application id, so a viewer that opens it never merges its raw topics into the segment."""

VIDEO_TIME_FIELD: pa.Field = (
    rrc.Chunk.from_columns("/", indexes=[time_column(np.array([0], dtype=np.int64))], columns=rr.Scalars.columns(scalars=[0.0]))
    .to_record_batch()
    .schema.field(schema.TIMELINE)
)
"""Rerun-derived index field of the ``video_time`` duration timeline."""


def exposure_path(cam: int) -> str:
    """Per-frame exposure and white balance the camera ships with every image."""
    return f"{schema.cam_path(RIG, cam)}/exposure"


RETIMED_ENTITIES: dict[str, tuple[str, ...]] = {
    **{camera_topic(cam): (schema.video_path(RIG, cam), exposure_path(cam)) for cam in range(NUM_CAMERAS)},
    EEF_POSE_TOPIC: (schema.rig_path(RIG),),
    IMU_TOPIC: (schema.gyro_path(RIG, IMU), schema.accel_path(RIG, IMU)),
    FRAME_VALIDITY_TOPIC: ("/frames",),
}
"""Output entities counted from each source topic before lenses fan out its columns."""


def binary_to_blob(data: pa.BinaryArray) -> pa.ListArray:
    """``binary`` rows as ``list<uint8>`` (the ``Blob`` layout) over the same bytes, honouring a sliced array's offset."""
    offsets: pa.Array = pa.Array.from_buffers(pa.int32(), len(data) + 1, [None, data.buffers()[1]], offset=data.offset)
    values: pa.Array = pa.Array.from_buffers(pa.uint8(), data.buffers()[2].size, [None, data.buffers()[2]])
    return pa.ListArray.from_arrays(offsets, values)


def fixed_size_rows(values: pa.StructArray, fields: str) -> pa.FixedSizeListArray:
    """A struct's float fields as one fixed-size list per row, the layout of a ``Transform3D`` component."""
    return pa.FixedSizeListArray.from_arrays(pa.array(struct_rows(values, fields).reshape(-1)), len(fields))


def video_lens(cam: int) -> rrc.DeriveLens:
    """H.264 access units as ``VideoStream`` samples, with the toggling SPS flag cleared."""
    return rrc.DeriveLens("foxglove.CompressedImage:message", output_entity=schema.video_path(RIG, cam)).to_component(
        rr.VideoStream.descriptor_sample(), rrc.Selector(".data").pipe(lambda data: clear_sps_constraint_set3(binary_to_blob(data))), cast_to="auto"
    )


EXPOSURE_FIELDS: tuple[str, ...] = ("integration_time", "analog_gain", "aperture", "wb_rgain", "wb_bgain", "wb_ct", "metadata")
"""Every per-image field besides the pixels, the stamp and the constant format and frame id."""


def exposure_lens(cam: int) -> rrc.DeriveLens:
    """The per-image exposure fields as they are shipped."""
    lens: rrc.DeriveLens = rrc.DeriveLens("foxglove.CompressedImage:message", output_entity=exposure_path(cam))
    for name in EXPOSURE_FIELDS:
        lens = lens.to_component(name, rrc.Selector(f".{name}"))
    return lens


def rig_pose_lens() -> rrc.DeriveLens:
    """``world_T_rig`` from the VIO pose: translation and rotation from ONE lens.

    The viewer resolves a ``Transform3D`` from its latest row as a whole; two lenses would write the
    translation and the rotation on separate rows, and the rotation-only row resets the translation.
    """
    return (
        rrc.DeriveLens("foxglove.PoseInFrame:message", output_entity=schema.rig_path(RIG))
        .to_component(rr.Transform3D.descriptor_translation(), rrc.Selector(".pose.position").pipe(lambda p: fixed_size_rows(p, "xyz")), cast_to="auto")
        .to_component(rr.Transform3D.descriptor_quaternion(), rrc.Selector(".pose.orientation").pipe(lambda q: fixed_size_rows(q, "xyzw")), cast_to="auto")
    )


def vector_instances(vectors: pa.StructArray, *, scale: float = 1.0) -> pa.ListArray:
    """Each ``{x, y, z}`` times ``scale`` as a three-value list, which ``Selector("[]")`` then spreads into three scalar instances."""
    values: Float64[ndarray, "n 3"] = struct_rows(vectors) * scale
    return pa.ListArray.from_arrays(pa.array(np.arange(0, values.size + 1, 3, dtype=np.int32)), pa.array(values.reshape(-1)))


def imu_lenses() -> list[rrc.DeriveLens]:
    """Gyro in rad/s as shipped, accelerometer converted from g to m/s^2: three ``Scalars`` per sample, x y z."""
    gyro: rrc.DeriveLens = rrc.DeriveLens("foxglove.IMUMeasurement:message", output_entity=schema.gyro_path(RIG, IMU)).to_component(
        rr.Scalars.descriptor_scalars(), rrc.Selector(".angular_velocity").pipe(vector_instances).pipe(rrc.Selector("[]")), cast_to="auto"
    )
    accel: rrc.DeriveLens = rrc.DeriveLens("foxglove.IMUMeasurement:message", output_entity=schema.accel_path(RIG, IMU)).to_component(
        rr.Scalars.descriptor_scalars(),
        rrc.Selector(".linear_acceleration").pipe(lambda vectors: vector_instances(vectors, scale=STANDARD_GRAVITY_M_S2)).pipe(rrc.Selector("[]")),
        cast_to="auto",
    )
    return [gyro, accel]


def frames_lens() -> rrc.DeriveLens:
    """``/frames`` (exoego §14) from ``frame_validity``: the source's frame number, stamp, validity and failures.

    ``source_timestamp_s`` is the schema's column; a float64 of epoch seconds rounds the stamp to about
    0.2 us, so the exact stamp rides beside it as ``source_timestamp_ns``.
    """
    return (
        rrc.DeriveLens("foxglove.FrameValidity:message", output_entity="/frames")
        .to_component("source_frame_id", rrc.Selector(".frame_index").pipe(lambda index: index.cast(pa.int64())))
        .to_component("source_timestamp_s", rrc.Selector(".timestamp_ns").pipe(lambda stamp: pa.array(stamp.to_numpy(zero_copy_only=False) / 1e9)))
        .to_component("source_timestamp_ns", rrc.Selector(".timestamp_ns").pipe(lambda stamp: stamp.cast(pa.int64())))
        .to_component("is_valid", rrc.Selector(".is_valid"))
        .to_component("failures", rrc.Selector(".failures"))
    )


@dataclass
class Retimer:
    """Moves each source chunk before lenses from the MCAP's ``message_log_time`` onto ``video_time`` and ``frame_index``.

    Chunks keep their row ids. Native-rate sensors (the IMU) get ``video_time`` only, as exoego §14
    asks; every frame-aligned stream also gets the reference camera's frame. Counts each entity's rows
    on the way through, so the caller can hold them against the MCAP's own message counts.
    """

    clock: EpisodeClock
    """The episode clock."""
    native_rate: frozenset[str]
    """Source topics that keep their own rate and get no ``frame_index``."""
    starts: dict[str, int] = field(default_factory=dict)
    """Each camera's first decodable stamp; leading P-frames are dropped before deriving columns."""
    rows: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    """Rows seen per entity."""
    frames: dict[str, list[Int64[ndarray, "n"]]] = field(default_factory=lambda: defaultdict(list))
    """``frame_index`` values seen per entity."""
    lock: LockType = field(default_factory=threading.Lock)
    """The stream may call back from several threads; the tallies are shared."""

    def __call__(self, chunk: rrc.Chunk) -> list[rrc.Chunk]:
        batch: pa.RecordBatch = chunk.to_record_batch()
        if "message_log_time" not in batch.schema.names:
            return [chunk]
        stamps: Int64[ndarray, "n"] = np.asarray(batch.column("message_log_time").cast(pa.int64()))
        start_ns = self.starts.get(chunk.entity_path)
        if start_ns is not None and np.any(stamps < start_ns):
            keep = stamps >= start_ns
            if not np.any(keep):
                return []
            batch = batch.filter(pa.array(keep))
            stamps = stamps[keep]
        index: int = batch.schema.get_field_index("message_log_time")
        batch = batch.set_column(index, VIDEO_TIME_FIELD, pa.array(self.clock.video_time_ns(stamps), type=pa.duration("ns")))
        batch = batch.remove_column(batch.schema.get_field_index("message_publish_time"))
        frames: Int64[ndarray, "n"] = self.clock.frame_indices(stamps)
        with self.lock:
            entities = RETIMED_ENTITIES.get(chunk.entity_path, (chunk.entity_path,))
            for entity in entities:
                self.rows[entity] += batch.num_rows
                if chunk.entity_path not in self.native_rate:
                    self.frames[entity].append(frames)
        if chunk.entity_path not in self.native_rate:
            batch = batch.add_column(index + 1, FRAME_INDEX_FIELD, pa.array(frames, type=pa.int64()))
        return rrc.Chunk.from_record_batch(batch)

    def frame_indices(self, entity: str) -> Int64[ndarray, "n"]:
        """Every ``frame_index`` an entity received, sorted."""
        parts: list[Int64[ndarray, "n"]] = self.frames.get(entity, [])
        return np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)


KEYFRAME_SEARCH_NS: int = 5_000_000_000
"""Window the keyframe search reads at a time; the encoder's GoP is 30 frames (1 s)."""


@dataclass(frozen=True, slots=True)
class VideoStart:
    """Where one camera's decodable video begins."""

    stamp_ns: int
    """Stamp of its first keyframe."""
    skipped: int
    """Samples before it: P-frames whose reference frame was never recorded, which no decoder can show."""


def video_start(file: EpisodeFile, cam: int, *, t0_ns: int, end_ns: int) -> VideoStart:
    """Find a camera's first keyframe; some streams start mid-GoP, and a camera may start late.

    The stream is read in ``KEYFRAME_SEARCH_NS`` windows from the recording start, so the search stops a
    few seconds into the stream however late it begins.

    Raises:
        ValueError: The camera has no keyframe at all; no decoder could show any of its frames.
    """
    skipped: int = 0
    for window_ns in range(t0_ns, end_ns + 1, KEYFRAME_SEARCH_NS):
        for stamps, values in messages(file.stream([camera_topic(cam)], start_time_ns=window_ns, end_time_ns=window_ns + KEYFRAME_SEARCH_NS)):
            if np.any(np.diff(stamps) <= 0):
                raise ValueError(f"{file.path}: camera{cam} samples are not in stamp order")
            for stamp, sample in zip(stamps.tolist(), values.field("data").to_pylist(), strict=True):
                if is_keyframe(sample):
                    return VideoStart(stamp_ns=stamp, skipped=skipped)
                skipped += 1
    raise ValueError(f"{file.path}: camera{cam} has no keyframe in {skipped} samples")


def base_stream(
    file: EpisodeFile, episode: Episode, retimer: Retimer, starts: Sequence[VideoStart],
    *, codecs: Sequence[rrc.Chunk], batchers: list[ChunkBatcher],
) -> rrc.LazyChunkStream:
    """Cameras from their first keyframe, rig pose, IMU and frames of one MCAP, shaped by lenses and retimed."""
    # Decode/decompress each MCAP block once. Split branches share this reader;
    # six independent camera readers repeatedly decompressed the same source blocks.
    retimer.starts = {camera_topic(cam): start.stamp_ns for cam, start in enumerate(starts)}
    stream = file.stream([*(camera_topic(cam) for cam in range(NUM_CAMERAS)), EEF_POSE_TOPIC, IMU_TOPIC, FRAME_VALIDITY_TOPIC]).flat_map(retimer)
    cameras: list[rrc.LazyChunkStream] = []
    for cam in range(NUM_CAMERAS):
        camera, stream = stream.split(content=camera_topic(cam))
        batcher = ChunkBatcher(static_chunks=(codecs[cam],))
        batchers.append(batcher)
        cameras.append(camera.lenses([video_lens(cam), exposure_lens(cam)]).flat_map(batcher))
    stream = stream.lenses(rig_pose_lens(), content=EEF_POSE_TOPIC)
    stream = stream.lenses(imu_lenses(), content=IMU_TOPIC)
    stream = stream.lenses(frames_lens(), content=FRAME_VALIDITY_TOPIC)
    batcher = ChunkBatcher()
    batchers.append(batcher)
    return rrc.LazyChunkStream.merge(*cameras, stream.flat_map(batcher))


def camera_parameters(cam: int, calibration: CameraCalibration) -> PinholeParameters:
    """The camera's frustum: the double-sphere focal lengths and principal point, at its ``rig_T_cam``.

    Only the frustum: a Rerun ``Pinhole`` cannot project through the double-sphere lens, so 3D content
    is not overlaid on these panes (the ``projections`` layer is).
    """
    transform: Float64[ndarray, "4 4"] = rig_T_cam(calibration.T_b_c)
    lens: DoubleSphere = calibration.lens
    return PinholeParameters(
        name=f"camera{cam}",
        extrinsics=Extrinsics(world_R_cam=transform[:3, :3], world_t_cam=transform[:3, 3]),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF", fl_x=lens.fx, fl_y=lens.fy, cx=lens.cx, cy=lens.cy, height=calibration.height, width=calibration.width
        ),
    )


@dataclass(frozen=True, slots=True)
class BaseCensus:
    """What the base layer wrote, held against the MCAP's own counts."""

    camera_frames: tuple[int, ...]
    """Video samples written per camera."""
    rig_poses: int
    """``world_T_rig`` rows."""
    imu_samples: int
    """IMU rows."""
    frames: int
    """``/frames`` rows."""


def write_sidecar(mcap: Path, target: Path, *, recording_id: str) -> None:
    """Keep the MCAP's non-camera topics, protobuf-decoded, plus its statistics, in an rrd beside the layers.

    Left out: the camera packets (they are the base layer's video), the 960x780 calibration of an image
    the file does not contain, and the two topics that duplicate others (``DUPLICATE_TOPICS``).

    The audio (16 kHz stereo PCM16) lives here and nowhere else: exoego:v2 has no audio slot. Newer
    Rerun releases can log audio, so a later layer can add it from this file without the MCAP.
    """
    excluded: list[str] = [
        *(topic_pattern(camera_topic(cam)) for cam in range(NUM_CAMERAS)),
        *(topic_pattern(resize_calibration_topic(cam)) for cam in range(NUM_CAMERAS)),
        *(topic_pattern(topic) for topic in DUPLICATE_TOPICS),
    ]
    stream: rrc.LazyChunkStream = rrc.McapReader(mcap, decoders=list(SIDECAR_DECODERS), exclude_topic_regex=excluded).stream()
    with writing.atomic_write(target) as temp:
        stream.write_rrd(temp, application_id=SIDECAR_APPLICATION_ID, recording_id=recording_id)


def write_base_layer(
    identity: SequenceIdentity,
    mcap: Path,
    target: Path,
    *,
    episode: Episode,
    domain: tuple[str, ...],
    hf_revision: str,
    default_blueprint: rrb.Blueprint | None,
) -> BaseCensus:
    """Publish one episode's base layer atomically and check it against the MCAP's message counts.

    Raises:
        ValueError: A stream wrote a different number of rows than the MCAP holds for its topic.
    """
    file: EpisodeFile = EpisodeFile(mcap)
    counts: dict[str, int] = shipped_counts(mcap)
    robot: RobotInfo | None = optional_message(file, ROBOT_INFO_TOPIC, RobotInfo)
    robot_name: str | None = None if robot is None else robot.robot_name
    retimer: Retimer = Retimer(episode.clock, native_rate=frozenset({IMU_TOPIC}))
    end_ns: int = rrc.McapReader(mcap).time_bounds()[1]
    starts: list[VideoStart] = [video_start(file, cam, t0_ns=episode.clock.t0_ns, end_ns=end_ns) for cam in range(NUM_CAMERAS)]
    resolutions: list[str] = []
    with writing.atomic_recording(target, recording_id=identity.recording_id, default_blueprint=default_blueprint) as recording:
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
        rr.log("/", annotation_context(), static=True, recording=recording)
        log_rig_node(recording, RIG, reference=schema.imu_path(RIG, IMU).rsplit("/", 1)[1], num_cameras=NUM_CAMERAS, name=robot_name, kind="ego")
        for cam, calibration in enumerate(episode.calibrations):
            log_camera_node(
                recording, RIG, cam, camera_parameters(cam, calibration),
                name=f"camera{cam}", kind="rgb", image_plane_distance=IMAGE_PLANE_DISTANCE_M, camera_model="ds",
            )  # fmt: skip
            rr.log(
                schema.cam_path(RIG, cam),
                rr.AnyValues(
                    ds_coefficients=pa.array([calibration.D], type=pa.list_(pa.float64())),
                    calib_T_cam=pa.array([calibration.T_b_c], type=pa.list_(pa.float64())),
                    calibration_frame_id=calibration.frame_id,
                ),
                static=True,
                recording=recording,
            )
            rr.log(schema.cam_path(RIG, cam), rr.AnyValues(skipped_leading_samples=pa.array([starts[cam].skipped], type=pa.int64())), static=True, recording=recording)
            resolutions.append(
                log_camera_source(
                    recording, RIG, cam, name=f"camera{cam}", source_width=SOURCE_WIDTH, source_height=SOURCE_HEIGHT,
                    video_codec=VIDEO_CODEC, source_num_frames=counts[camera_topic(cam)],
                )  # fmt: skip
            )
        empty: ImuChannel = ImuChannel(np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float64))
        transform: Float64[ndarray, "4 4"] = rig_T_imu()
        log_imu(recording, RIG, IMU, gyro=empty, accel=empty, name="imu", rig_T_imu=rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]))
        if episode.annotation is not None:
            rr.log(schema.instruction_path(), rr.TextDocument(episode.annotation.bold_mark, media_type=rr.MediaType.MARKDOWN), static=True, recording=recording)
        # The static codecs ride in the optimized stream: GoP rebatching needs each video's codec beside its samples.
        codecs = [rrc.Chunk.from_columns(schema.video_path(RIG, cam), indexes=[], columns=rr.VideoStream.columns(codec=[rr.VideoCodec.H264])) for cam in range(NUM_CAMERAS)]
        recording.send_chunks(codecs)
        batchers: list[ChunkBatcher] = []
        stream = base_stream(file, episode, retimer, starts, codecs=codecs, batchers=batchers)
        recording.send_chunks(stream)
        for batcher in batchers:
            recording.send_chunks(batcher.flush())
        census: BaseCensus = BaseCensus(
            camera_frames=tuple(retimer.rows[schema.video_path(RIG, cam)] for cam in range(NUM_CAMERAS)),
            rig_poses=retimer.rows[schema.rig_path(RIG)],
            imu_samples=retimer.rows[schema.gyro_path(RIG, IMU)],
            frames=retimer.rows["/frames"],
        )
        expected: dict[str, int] = {
            **{camera_topic(cam): census.camera_frames[cam] + starts[cam].skipped for cam in range(NUM_CAMERAS)},
            EEF_POSE_TOPIC: census.rig_poses,
            IMU_TOPIC: census.imu_samples,
            FRAME_VALIDITY_TOPIC: census.frames,
        }
        short: dict[str, tuple[int, int]] = {topic: (rows, counts.get(topic, 0)) for topic, rows in expected.items() if rows != counts.get(topic, 0)}
        if short:
            raise ValueError(f"{identity.sequence_key}: written rows differ from the MCAP's message counts (written, shipped): {short}")
        log_missing_cameras(recording, episode.clock, retimer)
        send_properties(recording, identity, episode, census, domain=domain, hf_revision=hf_revision, robot_name=robot_name, resolutions=resolutions)
    return census


def shipped_counts(mcap: Path) -> dict[str, int]:
    """Messages per topic, from the MCAP's summary.

    Raises:
        ValueError: The file has no statistics, so a census cannot be held against it.
    """
    counts: dict[str, int] = {}
    for channel in rrc.McapReader(mcap).info().channels:
        if channel.message_count is None:
            raise ValueError(f"{mcap}: no message count for {channel.topic}; the MCAP summary lacks statistics")
        counts[channel.topic] = channel.message_count
    return counts


def log_missing_cameras(recording: rr.RecordingStream, clock: EpisodeClock, retimer: Retimer) -> None:
    """``/frames:missing_cameras`` (exoego §14): per reference frame, the cameras with no image at it."""
    present: list[set[int]] = [set(retimer.frame_indices(schema.video_path(RIG, cam)).tolist()) for cam in range(NUM_CAMERAS)]
    missing: list[list[str]] = [[f"camera{cam}" for cam in range(NUM_CAMERAS) if frame not in present[cam]] for frame in clock.frame_numbers.tolist()]
    rr.send_columns(
        "/frames",
        indexes=[time_column(clock.video_time_ns(clock.frame_times_ns)), rr.TimeColumn(schema.FRAME_INDEX, sequence=clock.frame_numbers)],
        columns=rr.AnyValues.columns(missing_cameras=pa.array(missing, type=pa.list_(pa.string()))),
        recording=recording,
    )


def send_properties(
    recording: rr.RecordingStream,
    identity: SequenceIdentity,
    episode: Episode,
    census: BaseCensus,
    *,
    domain: tuple[str, ...],
    hf_revision: str,
    robot_name: str | None,
    resolutions: list[str],
) -> None:
    """``capture`` and ``episode`` properties; every value has the same Arrow type in every episode."""
    writing.send_capture_properties(
        recording,
        identity,
        num_cameras=NUM_CAMERAS,
        num_frames=max(census.camera_frames),
        hf_revision=hf_revision,
        clock_source=CLOCK_SOURCE,
        source_resolution=pa.array(resolutions),
        source_start_time_ns=pa.array([episode.clock.t0_ns], type=pa.int64()),
        reference_camera=f"camera{REFERENCE_CAMERA}",
        num_reference_frames=pa.array([episode.clock.frame_times_ns.size], type=pa.int64()),
        duration_s=pa.array([episode.clock.duration_s], type=pa.float64()),
        sps_constraint_set3_cleared=pa.array([True]),
        accelerometer_source_unit="g",
    )
    # An absent annotation, system info or validity message leaves its properties null, each in its one Arrow type.
    annotation: Annotation | None = episode.annotation
    system: SystemInfo = episode.system or SystemInfo()
    sub_success: list[bool] | None = None if annotation is None else [sub.is_success for sub in annotation.sub_segments]
    invalid: list[InvalidRange] | None = None if episode.validity is None else episode.validity.invalid_ranges or []
    levels: list[str | None] = [*domain, *([None] * (DOMAIN_LEVELS - len(domain)))]
    recording.send_property(
        "episode",
        rr.AnyValues(
            task=pa.array([None if annotation is None else annotation.bold_mark], type=pa.string()),
            sst=pa.array([None if annotation is None else annotation.sst], type=pa.string()),
            **{f"domain_{level}": pa.array([value], type=pa.string()) for level, value in enumerate(levels)},
            video_id=pa.array([None if annotation is None else annotation.video_id], type=pa.string()),
            num_segments=pa.array([None if annotation is None else len(annotation.segments_info or [])], type=pa.int64()),
            num_sub_segments=pa.array([None if sub_success is None else len(sub_success)], type=pa.int64()),
            is_success=pa.array([all(sub_success) if sub_success else None], type=pa.bool_()),
            firmware=pa.array([system.version], type=pa.string()),
            hardware_version=pa.array([system.HardwareVersion], type=pa.string()),
            device=pa.array([system.VersionName], type=pa.string()),
            device_name=pa.array([robot_name], type=pa.string()),
            product_model_id=pa.array([system.product_model_id], type=pa.int64()),
            recording_start_trigger=pa.array([None if system.recording_start_trigger is None else system.recording_start_trigger.name], type=pa.string()),
            abnormal_reason=pa.array([system.abnormal_reason], type=pa.int64()),
            calibration=pa.array([episode.calibrations[REFERENCE_CAMERA].frame_id], type=pa.string()),
            invalid_codes=pa.array([None if invalid is None else [r.invalid_code for r in invalid]], type=pa.list_(pa.string())),
            invalid_start_s=pa.array([None if invalid is None else [r.start_time_s for r in invalid]], type=pa.list_(pa.float64())),
            invalid_end_s=pa.array([None if invalid is None else [r.end_time_s for r in invalid]], type=pa.list_(pa.float64())),
            invalid_messages=pa.array([None if invalid is None else [r.invalid_message for r in invalid]], type=pa.list_(pa.string())),
        ),
    )
