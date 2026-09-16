"""exoego:v2 writers + video remux: the taps every dataforge converter reuses.

These taps live together because every dataset needs them *identically* and each
one hides an invariant that is easy to break silently: the rig node's honest key
set, the ``Mp4Reader`` → ``send_chunks`` pass-through (which must not mint fresh
row ids), and the IMU/magnetometer nodes' mandatory static ``rig_T_sensor``.

The encoder that produces the mp4 these writers remux lives next door in
``dataforge.video_encoding`` and is re-exported here, so a converter that
encodes an image sequence and then logs it has one import to make.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters
from simplecv.rerun_log_utils import log_pinhole
from simplecv.rig import CameraKind, PeerSensorKind

from dataforge import schema
from dataforge.video_encoding import (
    RAW_PIXEL_FORMATS as RAW_PIXEL_FORMATS,
)
from dataforge.video_encoding import (
    FrameKind as FrameKind,
)
from dataforge.video_encoding import (
    FrameSource as FrameSource,
)
from dataforge.video_encoding import (
    encode_frames_to_mp4 as encode_frames_to_mp4,
)
from dataforge.video_encoding import (
    encode_image_files_to_mp4 as encode_image_files_to_mp4,
)
from dataforge.video_encoding import (
    mp4_frame_count as mp4_frame_count,
)
from dataforge.video_encoding import (
    require_av1_nvenc as require_av1_nvenc,
)
from dataforge.video_encoding import (
    resolve_ffmpeg as resolve_ffmpeg,
)

VIDEO_SAMPLE_COMPONENT: str = "VideoStream:sample"
"""Component that marks an ``Mp4Reader`` chunk as carrying samples, not keyframe flags."""

VIDEO_KEYFRAME_COMPONENT: str = "VideoStream:is_keyframe"
"""Component of the trailing chunk that flags which samples are keyframes."""

VideoChunkKind: TypeAlias = Literal["codec", "sample", "keyframe"]
"""What one chunk out of an ``Mp4Reader`` stream is, by the components it carries."""

IDENTITY_TRANSFORM: rr.Transform3D = rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3, dtype=np.float32))
"""Explicit identity pose; an argument-less ``Transform3D`` logs no components at all."""


def classify_video_chunk(record_batch: pa.RecordBatch) -> VideoChunkKind:
    """Name one ``Mp4Reader`` chunk by its component set, or refuse to guess.

    ``Mp4Reader`` emits three shapes and the retiming tap treats each one
    differently, so which is which is decided here rather than by an ``else``
    that would silently absorb a fourth shape a later reader adds: a static
    ``codec`` chunk carrying no index at all, the per-GOP ``sample`` chunks whose
    index *is* the presentation clock, and one trailing ``keyframe`` chunk
    indexed on the same timeline but holding one row per keyframe rather than per
    sample.

    Args:
        record_batch: One chunk's batch, whose ``rerun:*`` schema metadata still
            carries the entity path an unrecognized shape is reported against.

    Returns:
        Which of the three shapes this batch is.

    Raises:
        ValueError: The batch is indexed on ``video_time`` but carries neither a
            sample nor a keyframe column, so retiming it would be a guess.
    """
    names: list[str] = list(record_batch.schema.names)
    if schema.TIMELINE not in names:
        return "codec"
    if VIDEO_SAMPLE_COMPONENT in names:
        return "sample"
    if VIDEO_KEYFRAME_COMPONENT in names:
        return "keyframe"
    metadata: dict[bytes, bytes] = record_batch.schema.metadata or {}
    entity_path: str = metadata.get(b"rerun:entity_path", b"<unknown entity>").decode(errors="replace")
    raise ValueError(
        f"{entity_path}: an Mp4Reader chunk indexed on {schema.TIMELINE} carries neither {VIDEO_SAMPLE_COMPONENT} "
        f"nor {VIDEO_KEYFRAME_COMPONENT}, only {names}; the reader's chunk shapes changed and retiming it would be a guess"
    )


def time_column(times_ns: Int64[ndarray, "n_samples"]) -> rr.TimeColumn:
    """The ``video_time`` index column for one stream's sample times.

    ``view`` rather than ``astype``: ``timedelta64[ns]`` and ``int64`` share a
    layout, so the column reinterprets the caller's array instead of copying a
    1 kHz stream's worth of stamps. That only holds for ``int64``, hence the
    dtype check — a float clock would be reinterpreted as nonsense nanoseconds.

    Args:
        times_ns: Sample times on the ``video_time`` clock, in nanoseconds.

    Returns:
        The index column every ``send_columns`` call in this package passes.
    """
    assert times_ns.dtype == np.int64, f"video_time is an int64 nanosecond clock, got {times_ns.dtype}"
    return rr.TimeColumn(schema.TIMELINE, duration=times_ns.view("timedelta64[ns]"))


def log_rig_node(
    recording: rr.RecordingStream,
    rig: int,
    *,
    reference: str,
    num_cameras: int,
    name: str | None = None,
    kind: str | None = None,
) -> None:
    """Tag one ``/world/rig_NN`` node with the static metadata exoego:v2 requires.

    Deliberately NO static ``Transform3D`` on the rig node: per exoego:v2 a
    world-anchored rig carries no transform, and a static one would permanently
    shadow the temporal ``world_T_rig`` a slam/pose layer stacks on the same
    entity (verified: static components shadow temporal ones per component,
    silently).

    Args:
        recording: Destination recording stream.
        rig: Rig index; the node is ``schema.rig_path(rig)``.
        reference: Id of the sensor child whose frame the rig frame coincides
            with — usually ``"cam_00"``, but an inertially-referenced rig names
            its ``imu_MM`` instead.
        num_cameras: Number of cameras actually logged under this rig.
        name: Optional human device label (``"robocap"``, ``"oak"``, an iPhone name).
        kind: Optional device role (``"exo"`` / ``"ego"`` / ``"quest"``).
    """
    # AnyValues omits a None-valued kwarg while its key is still untyped, so name/kind
    # simply stay off the node when a dataset has nothing meaningful to say. The registry
    # is process-global: once any recording types the key, later Nones arrive as nulls.
    rr.log(
        schema.rig_path(rig),
        rr.AnyValues(schema_version=schema.EXOEGO_SCHEMA_VERSION, reference=reference, num_cameras=num_cameras, name=name, kind=kind),
        static=True,
        recording=recording,
    )


def log_video_stream(
    recording: rr.RecordingStream,
    video_path: Path,
    entity_path: str,
    *,
    shift_ns: int = 0,
    times_ns: Int64[ndarray, "n_samples"] | None = None,
) -> int:
    """Remux one mp4 as a ``VideoStream``, optionally retimed, and count its samples.

    Five invariants ride along, in the order they bite:

    1. ``Mp4Reader`` emits one **static** codec chunk that carries no index; it
       must pass through untouched (its ``timeline_names`` is empty).
    2. ``send_chunks`` drives the lazy stream to completion, so ``sample_count``
       is final on return — anything short of a fully-consuming terminal would
       silently leave it at 0.
    3. ``Chunk.from_record_batch`` is called **bare**: the batch's ``rerun:*``
       metadata already carries index and entity path, and passing either
       override forces fresh row/chunk ids.
    4. Raw sample PTS is wall clock only when ``shift_ns == 0``. A non-zero
       shift means the file's PTS are offsets from a capture epoch, and the
       ``video_time`` column is rewritten to the absolute clock.
    5. Not every indexed chunk is a sample: after the per-GOP ``VideoStream:sample``
       chunks the reader emits one trailing ``VideoStream:is_keyframe`` chunk,
       indexed on the same timeline, with one row per keyframe. Only the sample
       chunks count toward ``sample_count`` and consume ``times_ns``; the keyframe
       chunk is retimed by looking its PTS up among the samples already seen.
       ``classify_video_chunk`` names each shape from its components and refuses
       an unrecognized one, so a reader that grows a fourth shape fails loudly
       instead of having it silently retimed as a keyframe chunk.

    ``shift_ns`` and ``times_ns`` answer different questions and cannot be
    combined: a shift means "the file's own PTS are right, the origin is not",
    while ``times_ns`` means "the file's PTS are a nominal-rate fiction" — which
    is the case for any mp4 this package encoded from an image sequence, where
    the real capture times live in a separate timestamp file.

    Args:
        recording: Destination recording stream.
        video_path: mp4 to remux (no decode, no re-encode).
        entity_path: ``.../pinhole/video`` entity to log under.
        shift_ns: Nanoseconds added to every indexed chunk's ``video_time``.
        times_ns: Exact ``video_time`` per sample, in presentation order; must
            hold one value per sample in the file.

    Returns:
        Number of video samples written.

    Raises:
        ValueError: If both retiming modes are given, or if ``times_ns`` does not
            have exactly one value per sample in the mp4.
    """
    if times_ns is not None and shift_ns != 0:
        raise ValueError("shift_ns and times_ns are mutually exclusive: a per-sample clock is not an offset from the file's own")
    sample_count: int = 0
    # Original PTS and replacement time of every sample chunk seen so far. The
    # trailing keyframe chunk concatenates them once, rather than paying for a
    # per-sample dict on a stream that can run to millions of frames.
    seen_pts_ns: list[Int64[ndarray, "n_rows"]] = []
    seen_times_ns: list[Int64[ndarray, "n_rows"]] = []

    def retimed(record_batch: pa.RecordBatch, index: int, values_ns: Int64[ndarray, "n_rows"]) -> list[rr.experimental.Chunk]:
        """Same batch, same row ids, new index values (still a ``duration("ns")``)."""
        column: pa.Array = pa.array(values_ns, type=pa.duration("ns"))
        return rr.experimental.Chunk.from_record_batch(record_batch.set_column(index, record_batch.schema.field(index), column))  # invariant 3

    def tap(chunk: rr.experimental.Chunk) -> list[rr.experimental.Chunk]:
        nonlocal sample_count
        record_batch: pa.RecordBatch = chunk.to_record_batch()
        kind: VideoChunkKind = classify_video_chunk(record_batch)  # invariant 5
        if kind == "codec":
            return [chunk]  # invariant 1: the static codec chunk carries no index
        index: int = record_batch.schema.get_field_index(schema.TIMELINE)
        if kind == "sample":
            sample_count += record_batch.num_rows
        if times_ns is None:
            # invariant 4: the file's own PTS are the clock, so a plain remux never
            # reads the index out at all and a shift only adds a constant to it.
            if shift_ns == 0:
                return [chunk]
            return retimed(record_batch, index, np.asarray(record_batch.column(index).cast(pa.int64())) + shift_ns)

        original_ns: Int64[ndarray, "n_rows"] = np.asarray(record_batch.column(index).cast(pa.int64()))
        if kind == "sample":
            if sample_count > times_ns.size:
                raise ValueError(f"{video_path.name} has more samples than the {times_ns.size} timestamps given")
            replacement: Int64[ndarray, "n_rows"] = times_ns[sample_count - record_batch.num_rows : sample_count]
            seen_pts_ns.append(original_ns)
            seen_times_ns.append(replacement)
            return retimed(record_batch, index, replacement)

        # The trailing keyframe chunk. ``-bf 0`` forbids reordering, so the samples'
        # PTS are one ascending array and a single searchsorted places every keyframe
        # among them; comparing what it landed on is what catches a reader that
        # started emitting the keyframes before their samples.
        if not seen_pts_ns:
            raise ValueError(f"{video_path.name}: a keyframe chunk arrived before any sample; the reader's chunk order changed")
        sample_pts_ns: Int64[ndarray, "n_samples"] = np.concatenate(seen_pts_ns)
        found: Int64[ndarray, "n_rows"] = np.searchsorted(sample_pts_ns, original_ns)
        if int(found.max(initial=-1)) >= sample_pts_ns.size or not np.array_equal(sample_pts_ns[found], original_ns):
            raise ValueError(f"{video_path.name}: keyframe PTS {original_ns[:4].tolist()} precede their samples; the reader's chunk order changed")
        return retimed(record_batch, index, np.concatenate(seen_times_ns)[found])

    # A B-frame source (iPhone/insta360 HEVC) forces Mp4Reader into an FFmpeg
    # re-encode; everything else passes through untouched, and then these options
    # are documented no-ops. try_gpu turns that unavoidable re-encode into NVENC
    # when the ffmpeg build has it, which not every build does, so DATAFORGE_FFMPEG
    # points at an NVENC-capable binary (e.g. the fleet's ~/.pixi/bin/ffmpeg).
    ffmpeg_override: str | None = os.environ.get("DATAFORGE_FFMPEG")
    reader: rr.experimental.Mp4Reader = rr.experimental.Mp4Reader(
        video_path,
        mode="stream",
        entity_path=entity_path,
        timeline_name=schema.TIMELINE,
        transcode=rr.experimental.Mp4TranscodeOptions(try_gpu=True, ffmpeg_override=ffmpeg_override),
    )
    recording.send_chunks(reader.stream().flat_map(tap))  # invariant 2
    if times_ns is not None and sample_count != times_ns.size:
        raise ValueError(f"{video_path.name} holds {sample_count} samples but {times_ns.size} timestamps were given")
    return sample_count


@dataclass(frozen=True, slots=True)
class ImuChannel:
    """One raw IMU channel (gyro or accel) at its native sample rate."""

    times_ns: Int64[ndarray, "n_samples"]
    """Sample times on the ``video_time`` clock, in nanoseconds."""
    values_xyz: Float64[ndarray, "n_samples 3"]
    """Scaled samples (rad/s for gyro, m/s^2 for accel)."""


def _log_scalar_channel(recording: rr.RecordingStream, entity_path: str, channel: ImuChannel) -> None:
    """Send one sensor channel's samples columnar on ``video_time``; an empty channel logs nothing."""
    if channel.times_ns.size == 0:
        return
    rr.send_columns(
        entity_path,
        indexes=[time_column(channel.times_ns)],
        columns=rr.Scalars.columns(scalars=channel.values_xyz),
        recording=recording,
    )


def _log_sensor_node(recording: rr.RecordingStream, node: str, *, name: str, kind: PeerSensorKind, **extra: object) -> None:
    """Tag one non-camera peer sensor node with the static pair exoego:v2 §6 requires.

    The identity ``rig_T_sensor`` is **not** optional: a reader that cannot place
    a sensor's samples in the rig frame has to special-case the writer instead.
    ``**extra`` carries a sensor's own optional keys (the magnetometer's ``unit``).
    ``kind`` is a ``PeerSensorKind`` and not the wider ``SensorKind``: this writer
    logs no ``Pinhole``, so a camera word here would produce a camera node with
    no calibration.
    """
    rr.log(node, IDENTITY_TRANSFORM, static=True, recording=recording)
    rr.log(node, rr.AnyValues(drop_untyped_nones=True, name=name, kind=kind, **extra), static=True, recording=recording)


def log_camera_node(
    recording: rr.RecordingStream,
    rig: int,
    cam: int,
    camera: PinholeParameters | Fisheye62Parameters,
    *,
    name: str,
    kind: CameraKind,
    image_plane_distance: float,
    camera_model: str | None = None,
    distortion_valid_radius: float | None = None,
    image_rotation_cw_deg: int | None = None,
) -> None:
    """Tag one ``/world/rig_NN/cam_MM`` node and log its calibration under it.

    The node's metadata and its ``rig_T_cam`` + ``Pinhole`` belong together: a
    camera whose calibration lands without its ``name``/``kind`` reads as an
    unlabelled frustum, and one whose metadata lands without its calibration
    cannot be projected at all.

    The optional keys are **named** rather than taken as ``**extra``: they are a
    closed set that consumers read off the node, and a kwargs bag turns a
    misspelt one into a silently different AnyValue key.

    Args:
        recording: Destination recording stream.
        rig: Rig index owning the camera.
        cam: Camera index within the rig.
        camera: The camera's simplecv parameters (intrinsics, distortion, ``rig_T_cam``).
        name: Human stream label (``"cam0"``, ``"left-eye"``, …).
        kind: Image content; ``"grayscale"``, ``"rgb"`` or ``"depth"``.
        image_plane_distance: Frustum length in metres.
        camera_model: Projection the coefficients belong to (``"kb4"``,
            ``"pinhole-radtan8"``), so a consumer need not infer it from the
            distortion component; ``None`` leaves the key off.
        distortion_valid_radius: Normalized image radius past which the model
            stops holding, for the models that declare one; ``None`` leaves the
            key off, which is what a model without such a limit means.
        image_rotation_cw_deg: Clockwise rotation already applied to this
            camera's encoded frames — and to the ``camera`` parameters above,
            which describe the rotated image — so a consumer relating the video
            to the raw sensor readout knows how far it was turned. ``None``
            leaves the key off, which is what an unturned camera means: a
            logged ``0`` would state a decision where none was needed.
    """
    # drop_untyped_nones is AnyValues' default, but it is stated because callers
    # rely on it: a kb4 camera passes distortion_valid_radius=None to mean "this
    # model has no such radius", and the key must be absent rather than logged as
    # an untyped null.
    rr.log(
        schema.cam_path(rig, cam),
        rr.AnyValues(
            drop_untyped_nones=True,
            name=name,
            kind=kind,
            camera_model=camera_model,
            distortion_valid_radius=distortion_valid_radius,
            image_rotation_cw_deg=image_rotation_cw_deg,
        ),
        static=True,
        recording=recording,
    )
    log_pinhole(
        camera,
        cam_log_path=Path(schema.cam_path(rig, cam)),
        image_plane_distance=image_plane_distance,
        static=True,
        recording=recording,
    )


def log_pose_track(
    recording: rr.RecordingStream,
    entity_path: str,
    *,
    times_ns: Int64[ndarray, "n_poses"],
    translations_xyz: Float64[ndarray, "n_poses 3"],
    quaternions_xyzw: Float64[ndarray, "n_poses 4"],
) -> None:
    """Send a temporal pose track columnar: one ``Transform3D`` per sample on ``video_time``.

    The rig node's ``world_T_rig``, and every other track that animates an
    entity, go through here so the quaternion layout stays one decision: Rerun
    wants the scalar **last**, whatever order the source file wrote.

    Args:
        recording: Destination recording stream.
        entity_path: Entity to animate, usually ``schema.rig_path(rig)``.
        times_ns: Pose times on the ``video_time`` clock, in nanoseconds.
        translations_xyz: Positions in metres.
        quaternions_xyzw: Orientations, scalar last.
    """
    rr.send_columns(
        entity_path,
        indexes=[time_column(times_ns)],
        columns=rr.Transform3D.columns(translation=translations_xyz, quaternion=quaternions_xyzw),
        recording=recording,
    )


def log_trail_segments(
    recording: rr.RecordingStream,
    entity_path: str,
    *,
    times_ns: Int64[ndarray, "n_poses"],
    translations_xyz: Float64[ndarray, "n_poses 3"],
    color: tuple[int, int, int],
    radius_ui_points: float,
) -> None:
    """Send a motion trail columnar: one two-point ``LineStrips3D`` per pose, the step that reached it.

    A trail is what a blueprint shows through a cursor-relative window, so it has
    to be **per pose** rather than one growing strip — but a per-pose
    ``Points3D`` draws it as dots, and at a 1 kHz sample rate a dot wide enough
    to see is wider than the gap between samples, so the trail reads as a string
    of scattered balls. One segment per pose, from the previous position to this
    one, draws the same rows as a stroke the eye follows.

    Two decisions the caller does not get to vary per row:

    * **The first pose gets a zero-length segment**, so the trail has exactly as
      many rows as the pose track it trails. One fewer would put the window a
      sample out of step with the rig, which is worse than a strip the viewer
      draws as nothing.
    * **The tint and the width are static.** One trail is one quantity, not a
      per-row class. The width is in ui points because it is a stroke on screen:
      a metric radius would have to be re-picked for every device whose motion
      is on a different scale.

    Args:
        recording: Destination recording stream.
        entity_path: Entity to draw the trail on, usually ``schema.trail_path(source)``.
        times_ns: Pose times on the ``video_time`` clock, in nanoseconds.
        translations_xyz: Positions in metres, in the same order as ``times_ns``.
        color: Trail tint, RGB.
        radius_ui_points: Stroke width in ui points; Rerun carries it as a
            negative radius, which is what keeps it screen-space.
    """
    rr.log(
        entity_path,
        rr.LineStrips3D.from_fields(colors=color, radii=rr.Radius.ui_points(radius_ui_points)),
        static=True,
        recording=recording,
    )
    previous: Int64[ndarray, "n_poses"] = np.maximum(np.arange(times_ns.size, dtype=np.int64) - 1, 0)
    segments_xyz: Float64[ndarray, "n_poses 2 3"] = np.stack([translations_xyz[previous], translations_xyz], axis=1)
    rr.send_columns(
        entity_path,
        indexes=[time_column(times_ns)],
        columns=rr.LineStrips3D.columns(strips=list(segments_xyz)),
        recording=recording,
    )


def log_imu(recording: rr.RecordingStream, rig: int, imu: int, *, gyro: ImuChannel, accel: ImuChannel, name: str) -> None:
    """Log one IMU node: both channels columnar, plus the static node metadata.

    The static identity ``rig_T_imu`` is **not** optional — exoego:v2 §8 makes
    ``Transform3D`` part of the IMU node, so a reader can resolve the sensor's
    place in the rig frame without special-casing the writer.

    Args:
        recording: Destination recording stream.
        rig: Rig index owning the IMU.
        imu: IMU index within the rig.
        gyro: Angular-velocity samples in rad/s; an empty channel is skipped.
        accel: Linear-acceleration samples in m/s^2; an empty channel is skipped.
        name: Human label for the device (e.g. ``"dev0"``, ``"oak-imu"``).
    """
    _log_scalar_channel(recording, schema.gyro_path(rig, imu), gyro)
    _log_scalar_channel(recording, schema.accel_path(rig, imu), accel)
    _log_sensor_node(recording, schema.imu_path(rig, imu), name=name, kind="imu")


HEADING_COLOR: tuple[int, int, int] = (255, 128, 0)
"""Fixed arrow tint for every magnetometer heading; the field is one quantity, not a per-row class."""


def log_magnetometer(
    recording: rr.RecordingStream,
    rig: int,
    mag: int,
    *,
    field: ImuChannel,
    name: str,
    unit: str | None = None,
    heading_length_m: float = 0.15,
) -> None:
    """Log one magnetometer node: the raw field, a heading arrow, and the node metadata.

    The field is logged in the **sensor's own units** — consumer hardware
    ships unlabelled counts, and inventing a calibration would be worse than
    saying so — with the optional ``unit`` AnyValue recording what they are.
    ``heading`` is a derived convenience: the same samples normalized to a fixed
    length so the field direction is visible in the 3D view, riding the rig like
    every other child. Zero-norm rows (a dropout) get no arrow rather than a NaN.

    The static identity ``rig_T_mag`` is **not** optional, for the same reason as
    the IMU's: a reader resolves the sensor's place in the rig frame without
    special-casing the writer.

    Args:
        recording: Destination recording stream.
        rig: Rig index owning the magnetometer.
        mag: Magnetometer index within the rig.
        field: Timestamped 3-axis field samples; an empty channel logs only the
            static node.
        name: Human label for the device (e.g. ``"reverb-g2"``).
        unit: Physical unit of the samples when it is known (e.g. ``"mG"``);
            ``None`` leaves the key off rather than guessing.
        heading_length_m: Length of the heading arrows, in metres.
    """
    _log_scalar_channel(recording, schema.field_path(rig, mag), field)
    if field.times_ns.size:
        norms: Float64[ndarray, "n_samples"] = np.linalg.norm(field.values_xyz, axis=1)
        measured: Bool[ndarray, "n_samples"] = norms > 0.0
        if measured.any():
            headings: Float64[ndarray, "n_headings 3"] = field.values_xyz[measured] / norms[measured, None] * heading_length_m
            rr.log(schema.heading_path(rig, mag), rr.Arrows3D.from_fields(colors=HEADING_COLOR), static=True, recording=recording)
            rr.send_columns(
                schema.heading_path(rig, mag),
                indexes=[time_column(field.times_ns[measured])],
                columns=rr.Arrows3D.columns(vectors=headings),
                recording=recording,
            )
    _log_sensor_node(recording, schema.mag_path(rig, mag), name=name, kind="mag", unit=unit)
