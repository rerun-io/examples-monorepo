"""Shared exoego:v2 writers every dataforge converter reuses, plus the video encoder.

These taps live together because every dataset needs them *identically* and each
one hides an invariant that is easy to break silently: the rig node's honest key
set, the ``Mp4Reader`` → ``send_chunks`` pass-through (which must not mint fresh
row ids), the IMU/magnetometer nodes' mandatory static ``rig_T_sensor``, and the
encoder's ban on B-frames.

Datasets that ship image sequences instead of video get here through
``encode_frames_to_mp4``: frames are piped straight into ffmpeg's stdin, so a
converter never materializes a decoded frame tree on disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

from dataforge import schema

FrameKind: TypeAlias = Literal["png", "gray8", "rgb24"]
"""How one element of an encoder frame iterable is laid out."""

RAW_PIXEL_FORMATS: dict[FrameKind, str] = {"gray8": "gray", "rgb24": "rgb24"}
"""ffmpeg ``-pix_fmt`` name for each rawvideo frame kind."""

VIDEO_SAMPLE_COMPONENT: str = "VideoStream:sample"
"""Component that marks an ``Mp4Reader`` chunk as carrying samples, not keyframe flags."""

IDENTITY_TRANSFORM: rr.Transform3D = rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3, dtype=np.float32))
"""Explicit identity pose; an argument-less ``Transform3D`` logs no components at all."""


@dataclass(frozen=True, slots=True)
class FrameSource:
    """How the caller's frame iterable is laid out for ffmpeg's stdin."""

    kind: FrameKind
    """``"png"`` feeds encoded PNG bytes through ``image2pipe``; the raw kinds feed ``rawvideo`` planes."""
    width: int | None = None
    """Frame width in pixels; required for the raw kinds, which carry no header."""
    height: int | None = None
    """Frame height in pixels; required for the raw kinds, which carry no header."""

    def __post_init__(self) -> None:
        if self.kind == "png":
            return
        if self.width is None:
            raise ValueError(f"a {self.kind} source needs an explicit width: rawvideo frames carry no header")
        if self.height is None:
            raise ValueError(f"a {self.kind} source needs an explicit height: rawvideo frames carry no header")

    def input_args(self, *, fps: int) -> list[str]:
        """ffmpeg input-side arguments that describe this layout on ``pipe:0``."""
        if self.kind == "png":
            return ["-f", "image2pipe", "-framerate", str(fps), "-c:v", "png", "-i", "pipe:0"]
        return [
            "-f",
            "rawvideo",
            "-pix_fmt",
            RAW_PIXEL_FORMATS[self.kind],
            "-s",
            f"{self.width}x{self.height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
        ]


def resolve_ffmpeg() -> Path:
    """Locate the ffmpeg to encode with: ``DATAFORGE_FFMPEG`` first, then ``PATH``."""
    override: str | None = os.environ.get("DATAFORGE_FFMPEG")
    if override:
        return Path(override)
    found: str | None = shutil.which("ffmpeg")
    if found is None:
        raise FileNotFoundError("no ffmpeg on PATH; set DATAFORGE_FFMPEG to an NVENC-capable binary")
    return Path(found)


def require_av1_nvenc(ffmpeg: Path) -> None:
    """Refuse an ffmpeg that cannot encode AV1 on the GPU, before any frame is read.

    Checked up front because the alternative failure is a software AV1 encode
    that takes hours on a full sequence and looks like a hang.

    Args:
        ffmpeg: Binary to interrogate with ``-encoders``.
    """
    listed: subprocess.CompletedProcess[str] = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    )
    if "av1_nvenc" in listed.stdout:
        return
    raise RuntimeError(
        f"{ffmpeg} lists no av1_nvenc encoder; many conda-forge ffmpeg builds ship without NVENC. "
        "Point DATAFORGE_FFMPEG at a binary that has it (e.g. the fleet's ~/.pixi/bin/ffmpeg)."
    )


def encode_frames_to_mp4(
    frames: Iterable[bytes],
    output: Path,
    *,
    source: FrameSource,
    fps: int,
    gop: int = 30,
    cq: int = 32,
    ffmpeg: Path | None = None,
) -> int:
    """Encode an iterable of frames into an AV1 mp4 by piping them through ffmpeg.

    Nothing is written to disk but the mp4: a dataset that ships PNG or raw
    frames streams straight from its archive into ffmpeg's stdin. Two properties
    are load-bearing for the Rerun side:

    * **No B-frames** (``-bf 0``). ``rr.VideoStream`` rejects reordered samples,
      and ``Mp4Reader`` would otherwise have to re-encode the file it was just
      handed.
    * **Sample count is verified** against the mp4 after ffmpeg exits, so a
      short pipe (a truncated archive, a dead encoder) fails here rather than as
      a silent timestamp/sample misalignment in ``log_video_stream``.

    ffmpeg's stderr is drained by a thread while frames go into its stdin: both
    pipes are finite, so writing a large frame while stderr sits full deadlocks.

    Args:
        frames: One encoded PNG (``kind="png"``) or one raw plane per frame.
        output: mp4 to write; its parent directory must exist.
        source: Layout of the ``frames`` elements.
        fps: Nominal frame rate stamped into the container. Real per-sample
            timestamps are applied later by ``log_video_stream(times_ns=...)``.
        gop: Keyframe interval in frames.
        cq: NVENC constant-quality target; lower is bigger and better.
        ffmpeg: Binary to use; ``None`` resolves via ``resolve_ffmpeg()``.

    Returns:
        Number of frames fed into the encoder.
    """
    binary: Path = resolve_ffmpeg() if ffmpeg is None else ffmpeg
    require_av1_nvenc(binary)
    command: list[str] = [
        str(binary),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *source.input_args(fps=fps),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
        "-c:v",
        "av1_nvenc",
        "-preset",
        "p4",
        "-rc",
        "vbr",
        "-cq",
        str(cq),
        "-bf",
        "0",
        "-g",
        str(gop),
        "-movflags",
        "+faststart",
        str(output),
    ]
    complaints: list[bytes] = []
    fed: int = 0
    process: subprocess.Popen[bytes] = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stderr is not None
    drain: threading.Thread = threading.Thread(target=lambda: complaints.append(process.stderr.read()), daemon=True)  # pyrefly: ignore
    drain.start()
    try:
        for frame in frames:
            process.stdin.write(frame)
            fed += 1
    except BrokenPipeError:
        pass  # ffmpeg already died; its stderr below says why
    finally:
        process.stdin.close()
        returncode: int = process.wait()
        drain.join()
    if returncode != 0:
        stderr_text: str = b"".join(complaints).decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited {returncode} while encoding {output.name} after {fed} frames:\n{stderr_text}")

    written: int = mp4_frame_count(output)
    if written != fed:
        raise ValueError(f"{output} holds {written} samples but {fed} frames were fed; the pipe lost data")
    return fed


def mp4_frame_count(path: Path) -> int:
    """Number of video samples in an mp4, from the container index."""
    with av.open(str(path)) as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        if stream.frames:
            return stream.frames
        return sum(1 for packet in container.demux(stream) if packet.pts is not None)


def encode_image_files_to_mp4(paths: Sequence[Path], output: Path, *, fps: int, gop: int = 30, cq: int = 32, ffmpeg: Path | None = None) -> int:
    """Encode a PNG sequence already on disk, reading one file at a time.

    Args:
        paths: PNG files in presentation order.
        output: mp4 to write.
        fps: Nominal frame rate; see ``encode_frames_to_mp4``.
        gop: Keyframe interval in frames.
        cq: NVENC constant-quality target.
        ffmpeg: Binary to use; ``None`` resolves via ``resolve_ffmpeg()``.

    Returns:
        Number of frames fed into the encoder.
    """
    return encode_frames_to_mp4(
        (path.read_bytes() for path in paths), output, source=FrameSource("png"), fps=fps, gop=gop, cq=cq, ffmpeg=ffmpeg
    )


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
    consumed: int = 0
    new_time_by_pts: dict[int, int] = {}

    def retimed(record_batch: pa.RecordBatch, index: int, values_ns: Int64[ndarray, "n_rows"]) -> list[rr.experimental.Chunk]:
        """Same batch, same row ids, new index values (still a ``duration("ns")``)."""
        column: pa.Array = pa.array(values_ns, type=pa.duration("ns"))
        return rr.experimental.Chunk.from_record_batch(record_batch.set_column(index, record_batch.schema.field(index), column))  # invariant 3

    def tap(chunk: rr.experimental.Chunk) -> list[rr.experimental.Chunk]:
        nonlocal sample_count, consumed
        if schema.TIMELINE not in chunk.timeline_names:
            return [chunk]  # invariant 1: the static codec chunk carries no index
        record_batch: pa.RecordBatch = chunk.to_record_batch()
        index: int = record_batch.schema.get_field_index(schema.TIMELINE)
        original_ns: Int64[ndarray, "n_rows"] = np.asarray(record_batch.column(index).cast(pa.int64()))
        is_sample_chunk: bool = VIDEO_SAMPLE_COMPONENT in record_batch.schema.names  # invariant 5
        if is_sample_chunk:
            sample_count += record_batch.num_rows
        if times_ns is None:
            return [chunk] if shift_ns == 0 else retimed(record_batch, index, original_ns + shift_ns)
        if is_sample_chunk:
            if consumed + record_batch.num_rows > times_ns.size:
                raise ValueError(f"{video_path.name} has more samples than the {times_ns.size} timestamps given")
            replacement: Int64[ndarray, "n_rows"] = times_ns[consumed : consumed + record_batch.num_rows]
            consumed += record_batch.num_rows
            new_time_by_pts.update(zip(original_ns.tolist(), replacement.tolist(), strict=True))
            return retimed(record_batch, index, replacement)
        unseen: list[int] = [pts for pts in original_ns.tolist() if pts not in new_time_by_pts]
        if unseen:
            raise ValueError(f"{video_path.name}: keyframe PTS {unseen[:4]} precede their samples; the reader's chunk order changed")
        return retimed(record_batch, index, np.array([new_time_by_pts[pts] for pts in original_ns.tolist()], dtype=np.int64))

    # A B-frame source (iPhone/insta360 HEVC) forces Mp4Reader into an FFmpeg
    # re-encode; everything else passes through untouched, and then these options
    # are documented no-ops. try_gpu turns that unavoidable re-encode into NVENC
    # when the ffmpeg build has it — conda-forge's does not, so DATAFORGE_FFMPEG
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
    if times_ns is not None and consumed != times_ns.size:
        raise ValueError(f"{video_path.name} holds {consumed} samples but {times_ns.size} timestamps were given")
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
    node: str = schema.mag_path(rig, mag)
    if field.times_ns.size:
        rr.send_columns(
            schema.field_path(rig, mag),
            indexes=[rr.TimeColumn(schema.TIMELINE, duration=field.times_ns.astype("timedelta64[ns]"))],
            columns=rr.Scalars.columns(scalars=field.values_xyz),
            recording=recording,
        )
        norms: Float64[ndarray, "n_samples"] = np.linalg.norm(field.values_xyz, axis=1)
        measured: Bool[ndarray, "n_samples"] = norms > 0.0
        if measured.any():
            headings: Float64[ndarray, "n_headings 3"] = field.values_xyz[measured] / norms[measured, None] * heading_length_m
            rr.log(node + "/heading", rr.Arrows3D.from_fields(colors=HEADING_COLOR), static=True, recording=recording)
            rr.send_columns(
                schema.heading_path(rig, mag),
                indexes=[rr.TimeColumn(schema.TIMELINE, duration=field.times_ns[measured].astype("timedelta64[ns]"))],
                columns=rr.Arrows3D.columns(vectors=headings),
                recording=recording,
            )
    rr.log(node, IDENTITY_TRANSFORM, static=True, recording=recording)
    rr.log(node, rr.AnyValues(name=name, kind="mag", unit=unit), static=True, recording=recording)
