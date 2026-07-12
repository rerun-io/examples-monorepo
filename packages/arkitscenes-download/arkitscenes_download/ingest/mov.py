"""MOV track remuxing and QuickTime timed-metadata framing."""

import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
from jaxtyping import Bool, Float64

TRANSCODE_DIR: Path = Path("/tmp/arkit_spike")
VIDEO_BATCH_SIZE: int = 500


@dataclass(frozen=True, slots=True)
class VideoSamples:
    """One prepared single-track MP4 video asset."""

    path: Path
    """Path to a prepared single-track MP4 container."""
    delete_after_use: bool
    """Whether the consumer must remove the prepared file."""
    encoder: str
    """Encoder or remux mode used to prepare the track."""
    settings: str
    """Exact encoder settings used for provenance and cache identity."""
    ffmpeg_version: str
    """Preparing ffmpeg version."""
    transcode_seconds: float
    """Wall time spent preparing and verifying this track."""


def resolve_ffmpeg() -> Path:
    """Resolve ffmpeg from an explicit override or the active environment."""
    configured: str | None = os.environ.get("FFMPEG_PATH")
    executable: str | None = configured or shutil.which("ffmpeg")
    if executable is None:
        raise FileNotFoundError("ffmpeg is required; run in the Pixi viz environment or set FFMPEG_PATH")
    return Path(executable)


@dataclass(frozen=True, slots=True)
class VideoPacketSamples:
    """Frame-aligned raw AV1 packet samples for a Rerun VideoStream."""

    timestamps: Float64[np.ndarray, "n"]
    """Presentation timestamps in seconds on the prepared track clock."""
    payloads: list[bytes]
    """One raw AV1 packet per frame."""
    is_keyframes: Bool[np.ndarray, "n"]
    """Whether each access unit starts an independently decodable frame."""


@dataclass(frozen=True, slots=True)
class MetadataPacket:
    """One demuxed timed-metadata packet."""

    pts_seconds: float
    """Packet PTS in MOV time."""
    items: list[bytes]
    """Metadata payloads after the QuickTime item header."""


def split_metadata_items(packet_data: bytes) -> list[bytes]:
    """Split concatenated mebx items, whose big-endian size includes the header."""
    items: list[bytes] = []
    position: int = 0
    while position + 8 <= len(packet_data):
        item_size: int = struct.unpack_from(">I", packet_data, position)[0]
        if item_size < 8 or position + item_size > len(packet_data):
            break
        items.append(packet_data[position + 8 : position + item_size])
        position += item_size
    return items


def demux_metadata(path: Path, stream_index: int) -> list[MetadataPacket]:
    """Demux all nonempty packets from one mebx stream."""
    return demux_metadata_streams(path, (stream_index,))[stream_index]


def demux_metadata_streams(path: Path, indices: tuple[int, ...]) -> dict[int, list[MetadataPacket]]:
    """Demux selected timed-metadata streams in one container pass."""
    packets: dict[int, list[MetadataPacket]] = {index: [] for index in indices}
    with av.open(str(path)) as container:
        streams: list[av.stream.Stream] = [container.streams[index] for index in indices]
        for packet in container.demux(*streams):
            payload: bytes = bytes(packet)
            if packet.pts is None or not payload:
                continue
            pts_seconds: float = float(packet.pts * packet.time_base)
            packets[packet.stream.index].append(MetadataPacket(pts_seconds, split_metadata_items(payload)))
    return packets


def iter_video_samples(video: VideoSamples, batch_size: int = VIDEO_BATCH_SIZE):
    """Yield bounded batches of ordered frame-sized raw AV1 packets."""
    timestamps: list[float] = []
    payloads: list[bytes] = []
    is_keyframes: list[bool] = []
    # Disable MOV edit-list application so a negative pre-roll sample remains a
    # distinct access unit instead of being hidden from the packet stream.
    sample_count: int = 0
    with av.open(str(video.path), options={"advanced_editlist": "0"}) as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        expected_sample_count: int = stream.frames
        for packet in container.demux(stream):
            if packet.size == 0:
                continue
            if packet.pts is None or packet.dts is None or packet.time_base is None:
                raise ValueError("prepared video packet has no usable timestamp")
            if packet.pts != packet.dts:
                raise ValueError("prepared VideoStream packet contains a B-frame")
            timestamps.append(float(packet.pts * packet.time_base))
            payloads.append(bytes(packet))
            is_keyframes.append(bool(packet.is_keyframe))
            sample_count += 1
            if len(timestamps) == batch_size:
                yield VideoPacketSamples(np.asarray(timestamps, dtype=np.float64), payloads, np.asarray(is_keyframes, dtype=np.bool_))
                timestamps, payloads, is_keyframes = [], [], []
        if timestamps:
            yield VideoPacketSamples(np.asarray(timestamps, dtype=np.float64), payloads, np.asarray(is_keyframes, dtype=np.bool_))
        if expected_sample_count and sample_count != expected_sample_count:
            raise ValueError(f"prepared track declared {expected_sample_count} frames but demuxed {sample_count} samples")


def track_packet_times(path: Path, stream_index: int) -> Float64[np.ndarray, "n"]:
    """Sorted presentation times in seconds for one source video track."""
    return np.asarray([float(timestamp) for timestamp in _packet_pts(path, stream_index)], dtype=np.float64)


def _packet_pts(path: Path, stream_index: int = 0) -> list[Fraction]:
    """Read the sorted presentation timestamps for all nonempty video packets."""
    timestamps: list[Fraction] = []
    with av.open(str(path)) as container:
        stream: av.stream.Stream = container.streams[stream_index]
        for packet in container.demux(stream):
            if packet.size != 0 and packet.pts is not None:
                timestamps.append(packet.pts * packet.time_base)
    return sorted(timestamps)


def _verify_transcode(source_pts: list[Fraction], output_path: Path) -> None:
    """Reject a transcode unless its codec, frames, timing, and order match."""
    output_pts: list[Fraction] = _packet_pts(output_path)
    with av.open(str(output_path)) as container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        if stream.frames != len(source_pts):
            raise ValueError(f"transcode frame count changed: {len(source_pts)} -> {stream.frames}")
        # A negative pre-roll sample may be hidden by the MP4 edit list while still
        # contributing to the declared frame count; tracks starting at zero retain all PTS.
        expected_pts: list[Fraction] = source_pts[1:] if source_pts[0] < 0 else source_pts
        if output_pts != expected_pts:
            raise ValueError("transcode did not preserve the source PTS sequence exactly")
        if stream.codec_context.codec.id != av.Codec("av1", "r").id:
            raise ValueError(f"transcode codec is {stream.codec_context.name!r}, expected AV1")
        for packet in container.demux(stream):
            if packet.size != 0 and packet.pts != packet.dts:
                raise ValueError("transcode still contains reordered packets")


def _restore_source_pts(encoded_path: Path, source_pts: list[Fraction], output_path: Path) -> None:
    """Remux encoded frames onto the source track's exact presentation grid."""
    with av.open(str(encoded_path)) as input_container, av.open(str(output_path), mode="w", format="mp4") as output_container:
        input_stream: av.video.stream.VideoStream = input_container.streams.video[0]
        # PyAV exposes the AV1 decoder as libdav1d, which cannot be used as an
        # output template. Select the AV1 encoder context only to construct the
        # mux stream; packets and codec configuration remain stream copies.
        output_stream: av.video.stream.VideoStream = cast(av.video.stream.VideoStream, output_container.add_stream("av1"))
        output_stream.codec_context.extradata = input_stream.codec_context.extradata
        output_stream.codec_context.width = input_stream.codec_context.width
        output_stream.codec_context.height = input_stream.codec_context.height
        output_stream.time_base = Fraction(1, 16_800)
        packet_index: int = 0
        for packet in input_container.demux(input_stream):
            if packet.size == 0:
                continue
            if packet_index == 0 and input_stream.frames == len(source_pts) - 1:
                duplicate: av.Packet = av.Packet(bytes(packet))
                first_pts_ticks: Fraction = source_pts[0] / output_stream.time_base
                if first_pts_ticks.denominator != 1:
                    raise ValueError("source PTS is not representable on the MOV time base")
                duplicate.pts = int(first_pts_ticks)
                duplicate.dts = int(first_pts_ticks)
                duplicate.time_base = output_stream.time_base
                duplicate.stream = output_stream
                output_container.mux(duplicate)
                packet_index += 1
            if packet_index >= len(source_pts):
                raise ValueError("transcode produced more frames than the source")
            pts_ticks: Fraction = source_pts[packet_index] / output_stream.time_base
            if pts_ticks.denominator != 1:
                raise ValueError("source PTS is not representable on the MOV time base")
            packet.pts = int(pts_ticks)
            packet.dts = int(pts_ticks)
            packet.time_base = output_stream.time_base
            packet.stream = output_stream
            output_container.mux(packet)
            packet_index += 1
        if packet_index != len(source_pts):
            raise ValueError(f"transcode frame count changed: {len(source_pts)} -> {packet_index}")


def _transcode_av1(path: Path, stream_index: int, quarter_turns: int = 0, drop_leading: int = 0) -> tuple[Path, str, str, str]:
    """Transcode one video track to calibrated AV1 settings, dropping leading frames."""
    TRANSCODE_DIR.mkdir(parents=True, exist_ok=True)
    source_pts: list[Fraction] = _packet_pts(path, stream_index)[drop_leading:]
    if not source_pts:
        raise ValueError(f"dropping {drop_leading} leading frames leaves track {stream_index} empty")
    encoders: list[tuple[str, list[str]]] = [
        ("av1_nvenc", ["-c:v", "av1_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "30"]),
        ("svt_av1", ["-c:v", "libsvtav1", "-preset", "6", "-crf", "32"]),
    ]
    ffmpeg_path: Path = resolve_ffmpeg()
    version: str = subprocess.run([str(ffmpeg_path), "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    stat = path.stat()
    errors: list[str] = []
    filters: dict[int, str] = {
        0: "setpts=PTS-STARTPTS",
        1: "transpose=2,setpts=PTS-STARTPTS",
        2: "hflip,vflip,setpts=PTS-STARTPTS",
        3: "transpose=1,setpts=PTS-STARTPTS",
    }
    # Trimming happens post-decode in presentation order, so the encoder opens
    # the kept range with a fresh keyframe and every retained frame decodes.
    trim: str = f"select='gte(n\\,{drop_leading})'," if drop_leading else ""
    for encoder_name, encoder in encoders:
        settings: str = " ".join(encoder)
        identity: str = f"{stat.st_size}:{stat.st_mtime_ns}:{stream_index}:{quarter_turns % 4}:{drop_leading}:{settings}"
        fingerprint: str = hashlib.sha256(identity.encode()).hexdigest()[:16]
        output_path: Path = TRANSCODE_DIR / f"{path.stem}_track_{stream_index}_rot{quarter_turns % 4}_{fingerprint}.mp4"
        if output_path.exists():
            try:
                _verify_transcode(source_pts, output_path)
                return output_path, encoder_name, settings, version
            except (ValueError, av.FFmpegError):
                output_path.unlink()
        with tempfile.NamedTemporaryFile(prefix=f"{output_path.stem}_encoded_", suffix=".mp4", dir=TRANSCODE_DIR, delete=False) as encoded_file:
            encoded_path: Path = Path(encoded_file.name)
        with tempfile.NamedTemporaryFile(prefix=f"{output_path.stem}_restored_", suffix=".mp4", dir=TRANSCODE_DIR, delete=False) as restored_file:
            restored_path: Path = Path(restored_file.name)
        command: list[str] = [
            str(ffmpeg_path),
            "-y",
            "-v",
            "warning",
            "-copyts",
            "-noautorotate",
            "-i",
            str(path),
            "-map",
            f"0:{stream_index}",
            "-an",
            "-vf",
            trim + filters[quarter_turns % 4],
            *encoder,
            "-fps_mode",
            "passthrough",
            str(encoded_path),
        ]
        try:
            try:
                result: subprocess.CompletedProcess[str] = subprocess.run(command, check=False, capture_output=True, text=True)
            except FileNotFoundError as error:
                errors.append(str(error))
                continue
            if result.returncode == 0:
                _restore_source_pts(encoded_path, source_pts, restored_path)
                _verify_transcode(source_pts, restored_path)
                restored_path.replace(output_path)
                return output_path, encoder_name, settings, version
            errors.append(result.stderr[-2000:])
        finally:
            encoded_path.unlink(missing_ok=True)
            restored_path.unlink(missing_ok=True)
    raise RuntimeError("both AV1 transcoders failed:\n" + "\n".join(errors))


def prepare_video_track(
    path: Path, stream_index: int, quarter_turns: int = 0, keep_transcode_cache: bool = False, drop_leading: int = 0
) -> VideoSamples:
    """Prepare one track by always transcoding it to AV1."""
    started: float = time.perf_counter()
    transcode_path, encoder, settings, version = _transcode_av1(path, stream_index, quarter_turns, drop_leading)
    elapsed: float = time.perf_counter() - started
    return VideoSamples(transcode_path, not keep_transcode_cache, encoder, settings, version, elapsed)
