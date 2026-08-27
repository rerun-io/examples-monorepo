"""Shared exoego:v2 writers every dataforge converter reuses: rig node, video, IMU.

These three taps live together because both datasets need them *identically* and
each one hides an invariant that is easy to break silently: the rig node's honest
key set, the ``Mp4Reader`` → ``send_chunks`` pass-through (which must not mint
fresh row ids), and the IMU node's mandatory static ``rig_T_imu``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float64, Int64
from numpy import ndarray

from dataforge import schema

IDENTITY_TRANSFORM: rr.Transform3D = rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3, dtype=np.float32))
"""Explicit identity pose; an argument-less ``Transform3D`` logs no components at all."""


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
    # AnyValues drops None-valued kwargs, so name/kind simply stay off the node
    # when a dataset has nothing meaningful to say.
    rr.log(
        schema.rig_path(rig),
        rr.AnyValues(schema_version=schema.EXOEGO_SCHEMA_VERSION, reference=reference, num_cameras=num_cameras, name=name, kind=kind),
        static=True,
        recording=recording,
    )


def log_video_stream(recording: rr.RecordingStream, video_path: Path, entity_path: str, *, shift_ns: int = 0) -> int:
    """Remux one mp4 as a ``VideoStream``, optionally retimed, and count its samples.

    Four invariants ride along, in the order they bite:

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

    Args:
        recording: Destination recording stream.
        video_path: mp4 to remux (no decode, no re-encode).
        entity_path: ``.../pinhole/video`` entity to log under.
        shift_ns: Nanoseconds added to every indexed chunk's ``video_time``.

    Returns:
        Number of video samples written.
    """
    sample_count: int = 0

    def tap(chunk: rr.experimental.Chunk) -> list[rr.experimental.Chunk]:
        nonlocal sample_count
        if schema.TIMELINE not in chunk.timeline_names:
            return [chunk]  # invariant 1: the static codec chunk carries no index
        if shift_ns == 0:
            sample_count += chunk.num_rows
            return [chunk]
        record_batch: pa.RecordBatch = chunk.to_record_batch()
        index: int = record_batch.schema.get_field_index(schema.TIMELINE)
        shifted: pa.Array = pa.array(np.asarray(record_batch.column(index).cast(pa.int64())) + shift_ns, type=pa.duration("ns"))
        sample_count += record_batch.num_rows
        retimed: pa.RecordBatch = record_batch.set_column(index, record_batch.schema.field(index), shifted)
        return rr.experimental.Chunk.from_record_batch(retimed)  # invariant 3

    reader: rr.experimental.Mp4Reader = rr.experimental.Mp4Reader(
        video_path,
        mode="stream",
        entity_path=entity_path,
        timeline_name=schema.TIMELINE,
    )
    recording.send_chunks(reader.stream().flat_map(tap))  # invariant 2
    return sample_count


@dataclass(frozen=True, slots=True)
class ImuChannel:
    """One raw IMU channel (gyro or accel) at its native sample rate."""

    times_ns: Int64[ndarray, "n_samples"]
    """Sample times on the ``video_time`` clock, in nanoseconds."""
    values_xyz: Float64[ndarray, "n_samples 3"]
    """Scaled samples (rad/s for gyro, m/s^2 for accel)."""


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
    for channel, entity_path in ((gyro, schema.gyro_path(rig, imu)), (accel, schema.accel_path(rig, imu))):
        if channel.times_ns.size == 0:
            continue
        rr.send_columns(
            entity_path,
            indexes=[rr.TimeColumn(schema.TIMELINE, duration=channel.times_ns.astype("timedelta64[ns]"))],
            columns=rr.Scalars.columns(scalars=channel.values_xyz),
            recording=recording,
        )
    rr.log(schema.imu_path(rig, imu), IDENTITY_TRANSFORM, static=True, recording=recording)
    rr.log(schema.imu_path(rig, imu), rr.AnyValues(name=name, kind="imu"), static=True, recording=recording)
