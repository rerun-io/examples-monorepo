import hashlib
import io
import json
import os
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import av
import numpy as np
import rerun as rr
from jaxtyping import Int
from numpy import ndarray
from pyarrow import ChunkedArray, LargeListArray, ListArray

from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters
from simplecv.rerun_custom_types import (
    PinholeWithDistortion,
)
from simplecv.rerun_custom_types import (
    confidence_scores_to_rgb as confidence_scores_to_rgb,
)
from simplecv.rrd_query_utils import RRDQuerySession, first_valid_value, unwrap_singleton_lists


def _default_cache_root() -> Path:
    env_override: str | None = os.environ.get("SIMPLECV_VIDEO_CACHE")
    if env_override:
        return Path(env_override).expanduser()
    return Path.home() / ".cache" / "simplecv" / "exoego_videos"


@dataclass(slots=True)
class _VideoCacheMetadata:
    rrd_mtime_ns: int
    rrd_size: int


class VideoCache:
    """Filesystem-backed cache for remuxed AssetVideo blobs."""

    def __init__(self, root: Path | None = None) -> None:
        self.root: Path = (root or _default_cache_root()).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _bucket_dir(self, rrd_path: Path) -> Path:
        resolved: Path = rrd_path.resolve()
        sha1: str = hashlib.sha1(str(resolved).encode(), usedforsecurity=False).hexdigest()
        bucket: Path = self.root / sha1
        bucket.mkdir(parents=True, exist_ok=True)
        return bucket

    def _fingerprint(self, rrd_path: Path) -> tuple[int, int]:
        stat_result: os.stat_result = rrd_path.stat()
        return stat_result.st_mtime_ns, stat_result.st_size

    def _metadata_path(self, mp4_path: Path) -> Path:
        return mp4_path.with_suffix(mp4_path.suffix + ".json")

    def _load_metadata(self, metadata_path: Path) -> _VideoCacheMetadata | None:
        try:
            payload: Any = json.loads(metadata_path.read_text())
            return _VideoCacheMetadata(
                rrd_mtime_ns=int(payload["rrd_mtime_ns"]),
                rrd_size=int(payload["rrd_size"]),
            )
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def get(self, *, rrd_path: Path, camera_name: str) -> Path | None:
        bucket: Path = self._bucket_dir(rrd_path)
        cached_mp4: Path = bucket / f"{camera_name}.mp4"
        metadata_path: Path = self._metadata_path(cached_mp4)
        metadata: _VideoCacheMetadata | None = self._load_metadata(metadata_path)
        if metadata is None or not cached_mp4.exists():
            return None
        current_mtime, current_size = self._fingerprint(rrd_path)
        if metadata.rrd_mtime_ns != current_mtime or metadata.rrd_size != current_size:
            try:
                cached_mp4.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return cached_mp4

    def store(self, *, rrd_path: Path, camera_name: str, source_path: Path) -> None:
        bucket: Path = self._bucket_dir(rrd_path)
        dest: Path = bucket / f"{camera_name}.mp4"
        metadata_path: Path = self._metadata_path(dest)
        tmp_dest: Path = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(source_path, tmp_dest)
        os.replace(tmp_dest, dest)
        mtime, size = self._fingerprint(rrd_path)
        metadata_payload: dict[str, int] = {"rrd_mtime_ns": mtime, "rrd_size": size}
        metadata_path.write_text(json.dumps(metadata_payload))


_CACHE_DISABLED: bool = os.environ.get("SIMPLECV_VIDEO_CACHE_DISABLE", "0") in {"1", "true", "True"}
_VIDEO_CACHE: VideoCache | None = None


def get_video_cache() -> VideoCache | None:
    """Return process-wide video cache unless disabled via env."""

    global _VIDEO_CACHE
    if _CACHE_DISABLED:
        return None
    if _VIDEO_CACHE is None:
        _VIDEO_CACHE = VideoCache()
    return _VIDEO_CACHE


def get_safe_application_id() -> str:
    """Get application ID safely, with fallback if __main__.__file__ doesn't exist"""
    try:
        main = sys.modules.get("__main__")
        if main:
            file_attr = getattr(main, "__file__", None)
            if isinstance(file_attr, str):
                return Path(file_attr).stem
    except Exception:
        pass
    return "rerun-application"  # Default fallback


@dataclass
class RerunTyroConfig:
    application_id: str = field(default_factory=get_safe_application_id)
    """Name of the application"""
    recording_id: str | None = None
    """Recording ID; pin one (any stable string) so separate processes/restarts
    land in the same recording instead of forking new ones (tyro cannot parse a
    UUID union member, which made this field non-configurable from the CLI)."""
    connect: bool = False
    """Whether to connect to an existing rerun instance or not"""
    save: Path | None = None
    """Path to save the rerun data, this will make it so no data is visualized but saved"""
    serve: bool = False
    """Serve the rerun data"""
    headless: bool = False
    """Run rerun in headless mode"""
    live: bool = False
    """When combined with ``save``, stream to a spawned viewer AND write the .rrd
    file simultaneously (via ``set_sinks``), instead of ``save`` being file-only.
    Ignored when ``headless`` (no viewer), or when ``serve``/``connect`` is set."""
    port: int = 9876
    """Port of the viewer's gRPC proxy (used by ``spawn``, ``live`` + ``save``, and ``connect``)."""
    server_memory_limit: str = "4GB"
    """Memory budget for the spawned viewer's gRPC proxy buffer. The SDK default
    (~1 GiB) silently drops the oldest messages on long multi-camera recordings
    (e.g. a 46-min EPFL session), so the viewer shows truncated history. Accepts a
    size (``"4GB"``) or a percentage of RAM (``"25%"``)."""
    executable_name: str = "rerun"
    """Executable name passed to ``rerun.spawn`` when launching the viewer."""
    executable_path: str | None = None
    """Optional absolute or relative path to the Rerun executable."""

    def __post_init__(self):
        rr.init(
            application_id=self.application_id,
            recording_id=self.recording_id,
            default_enabled=True,
            strict=True,
        )
        self.rec_stream: rr.RecordingStream = rr.get_global_data_recording()  # type: ignore[assignment]

        if self.serve:
            rr.serve_grpc(server_memory_limit=self.server_memory_limit)
            rr.serve_web_viewer(open_browser=not self.headless)
        elif self.connect:
            # Send logging data to a separate, already-running `rerun` process
            # (honors ``port`` so multiple viewers can coexist on one machine).
            rr.connect_grpc(f"rerun+http://127.0.0.1:{self.port}/proxy")
        elif self.save is not None and self.live and not self.headless:
            # Stream to a spawned viewer AND save to a .rrd at the same time by
            # fanning out through explicit sinks. ``spawn``/``save`` each install a
            # single sink that would replace the other, so we spawn the viewer
            # process without auto-connecting and wire both sinks ourselves.
            rr.spawn(
                port=self.port,
                connect=False,
                server_memory_limit=self.server_memory_limit,
                executable_name=self.executable_name,
                executable_path=self.executable_path,
            )
            rr.set_sinks(
                rr.GrpcSink(f"rerun+http://127.0.0.1:{self.port}/proxy"),
                rr.FileSink(str(self.save)),
            )
        elif self.save is not None:
            rr.save(self.save)
        elif not self.headless:
            rr.spawn(
                port=self.port,
                server_memory_limit=self.server_memory_limit,
                executable_name=self.executable_name,
                executable_path=self.executable_path,
            )


def log_pinhole(
    camera: PinholeParameters | Fisheye62Parameters,
    cam_log_path: Path,
    image_plane_distance: int | float = 0.5,
    static: bool = False,
    *,
    recording: rr.RecordingStream | None = None,
    include_distortion: bool = True,
) -> None:
    """
    Logs the pinhole camera parameters and transformation data.

    Parameters:
    camera (PinholeParameters): The pinhole camera parameters including intrinsics and extrinsics.
    cam_log_path (Path): The path where the camera log will be saved.
    image_plane_distance (float, optional): The distance of the image plane from the camera. Defaults to 0.5.
    static (bool, optional): If True, the log data will be marked as static. Defaults to False.

    Returns:
    None
    """
    # camera intrinsics
    rr.log(
        f"{cam_log_path}/pinhole",
        PinholeWithDistortion.from_camera(
            camera,
            image_plane_distance=image_plane_distance,
            include_distortion=include_distortion,
        ),
        static=static,
        recording=recording,
    )
    # camera extrinsics
    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=camera.extrinsics.cam_t_world,
            mat3x3=camera.extrinsics.cam_R_world,
            from_parent=True,
        ),
        static=static,
        recording=recording,
    )


VideoLogMethod = Literal["video_stream", "asset_video"]


# Keyed on libavcodec AVCodecID (not name) so decoder aliases like
# ``libdav1d`` / ``libaom-av1`` resolve transparently — ``codec.id`` is stable.
_CODEC_ID_MAP: dict[int, rr.VideoCodec] = {
    27: rr.VideoCodec.H264,  # AV_CODEC_ID_H264
    173: rr.VideoCodec.H265,  # AV_CODEC_ID_HEVC
    225: rr.VideoCodec.AV1,  # AV_CODEC_ID_AV1
    167: rr.VideoCodec.VP9,  # AV_CODEC_ID_VP9
    139: rr.VideoCodec.VP8,  # AV_CODEC_ID_VP8
}

# H.264/H.265 packets from MP4 (avcC/hvcC) need Annex B framing for ``rr.VideoStream``.
_BSF_FOR_CODEC_ID: dict[int, str] = {
    27: "h264_mp4toannexb",
    173: "hevc_mp4toannexb",
}

# Aria Gen2 RGB's AV1 SPS ends after 11 semantic bytes + a trailing-bits byte.
# Rerun 0.32's GOP detector (scuffle-av1) parses the SPS without seeking to
# the declared OBU size, so it reads that trailing byte as the next OBU
# header and never finds the keyframe — keyframes get classified as non-sync
# and the viewer renders black. Replace only that exact 12-byte SPS with a
# decoder-equivalent that flips initial_display_delay_present_flag and
# clears initial_display_delay_present_for_this_op; the extra syntax bit
# consumes the trailing-bits byte without changing what dav1d decodes.
_ARIA_GEN2_RGB_AV1_SEQUENCE_HEADER_OBU: bytes = bytes.fromhex("0a0c00000062ea7ffbf804330080")
_ARIA_GEN2_RGB_AV1_RERUN_SEQUENCE_HEADER_OBU: bytes = bytes.fromhex("0a0c02000061753ffdfc02198040")


def _normalize_av1_sample_for_rerun(sample: bytes) -> bytes:
    """Rewrite the aria-gen2 RGB SPS to one rerun's GOP detector can parse."""
    if sample.startswith(_ARIA_GEN2_RGB_AV1_SEQUENCE_HEADER_OBU):
        return _ARIA_GEN2_RGB_AV1_RERUN_SEQUENCE_HEADER_OBU + sample[len(_ARIA_GEN2_RGB_AV1_SEQUENCE_HEADER_OBU) :]
    return sample


def log_video(
    video_source: Path | bytes,
    video_log_path: Path,
    timeline: str = "video_time",
    *,
    method: VideoLogMethod = "video_stream",
    recording: rr.RecordingStream | None = None,
) -> Int[ndarray, "num_frames"]:
    """
    Logs a video and its frame timestamps.

    Args:
        video_source: Path to video file or raw video bytes.
        video_log_path: The entity path where the video log will be saved.
        timeline: Timeline name for frame timestamps.
        method: ``"video_stream"`` (default) demuxes encoded codec samples into
            ``rr.VideoStream``. ``"asset_video"`` embeds the whole MP4 blob via
            ``rr.AssetVideo`` + ``rr.VideoFrameReference``; callers must request
            it explicitly.
        recording: Optional specific recording stream to log to.

    Returns:
        Frame timestamps in nanoseconds, sorted ascending.

    Raises:
        ValueError: When ``method`` is invalid, or when ``method="video_stream"``
            and the source codec is not supported by ``rr.VideoStream``.
    """
    if method == "asset_video":
        return _log_asset_video(video_source, video_log_path, timeline, recording=recording)
    if method != "video_stream":
        raise ValueError(f"Unsupported video logging method: {method!r}.")
    return _log_video_stream(video_source, video_log_path, timeline, recording=recording)


def _log_asset_video(
    video_source: Path | bytes,
    video_log_path: Path,
    timeline: str,
    *,
    recording: rr.RecordingStream | None,
) -> Int[ndarray, "num_frames"]:
    """Embed the MP4 as an ``rr.AssetVideo`` and emit ``VideoFrameReference`` rows."""
    video_asset = (
        rr.AssetVideo(contents=video_source) if isinstance(video_source, bytes) else rr.AssetVideo(path=video_source)
    )

    rr.log(str(video_log_path), video_asset, static=True, recording=recording)

    try:
        frame_timestamps_ns: Int[ndarray, "num_frames"] = video_asset.read_frame_timestamps_nanos()
    except RuntimeError as exc:
        warnings.warn(
            f"Rerun could not read AssetVideo frame timestamps ({exc}); falling back to PyAV demux timestamps.",
            stacklevel=2,
        )
        frame_timestamps_ns = _read_frame_timestamps_nanos_pyav(video_source)

    rr.send_columns(
        f"{video_log_path}",
        indexes=[rr.TimeColumn(timeline, duration=1e-9 * frame_timestamps_ns)],
        columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
        recording=recording,
    )
    return frame_timestamps_ns


def _read_frame_timestamps_nanos_pyav(video_source: Path | bytes) -> Int[ndarray, "num_frames"]:
    """Read display timestamps by demuxing with PyAV when Rerun cannot inspect a codec."""
    source_handle: io.BytesIO | str = io.BytesIO(video_source) if isinstance(video_source, bytes) else str(video_source)
    container: av.container.InputContainer = av.open(source_handle, mode="r")
    timestamps_ns: list[int] = []
    try:
        in_stream: av.video.stream.VideoStream = container.streams.video[0]
        time_base: Fraction | None = in_stream.time_base
        if time_base is None:
            raise ValueError("Input video stream has no time_base; cannot derive frame timestamps.")
        ns_scale: Fraction = Fraction(1_000_000_000, 1)
        for packet in container.demux(in_stream):
            if packet.pts is None:
                continue
            timestamps_ns.append(round(packet.pts * time_base * ns_scale))
    finally:
        container.close()
    if not timestamps_ns:
        raise RuntimeError("PyAV could not derive any frame timestamps from the video stream.")
    return np.sort(np.asarray(timestamps_ns, dtype=np.int64))


def _log_video_stream(
    video_source: Path | bytes,
    video_log_path: Path,
    timeline: str,
    *,
    recording: rr.RecordingStream | None,
) -> Int[ndarray, "num_frames"]:
    """Demux MP4 samples into ``rr.VideoStream`` without pixel decode or re-encode.

    H.264/H.265 length-prefix framing is converted to Annex B; AV1 OBUs are
    logged as-is except for the aria-gen2 RGB SPS rewrite in
    :func:`_normalize_av1_sample_for_rerun`. VideoStream currently has no
    separate DTS field, so B-frame streams use DTS as the rerun sample timeline
    to preserve decode order. Returns PTS (display order) for caller alignment.
    """
    source_handle: io.BytesIO | str = io.BytesIO(video_source) if isinstance(video_source, bytes) else str(video_source)
    container: av.container.InputContainer = av.open(source_handle, mode="r")

    pts_ns_list: list[int] = []
    dts_ns_list: list[int] = []
    samples: list[bytes] = []
    is_keyframes: list[bool] = []
    try:
        in_stream: av.video.stream.VideoStream = container.streams.video[0]
        codec_id: int = in_stream.codec_context.codec.id
        codec_name: str = in_stream.codec_context.name
        codec: rr.VideoCodec | None = _CODEC_ID_MAP.get(codec_id)
        if codec is None:
            raise ValueError(
                f"Codec {codec_name!r} (id={codec_id}) is not supported by rr.VideoStream. "
                f"Pass method='asset_video', or transcode the source to H.264/H.265/AV1 "
                f"(see simplecv.video_encoder.VideoEncoder for an NVENC-accelerated path)."
            )

        bsf_name: str | None = _BSF_FOR_CODEC_ID.get(codec_id)
        bsf: av.BitStreamFilterContext | None = (
            av.BitStreamFilterContext(bsf_name, in_stream) if bsf_name is not None else None
        )

        rr.log(
            str(video_log_path),
            rr.VideoStream(codec=codec),
            static=True,
            recording=recording,
        )

        # The bsf preserves both pts and dts but may drop time_base; the
        # input stream's time_base is the source of truth for both.
        in_time_base: Fraction | None = in_stream.time_base
        if in_time_base is None:
            raise ValueError("Input video stream has no time_base; cannot derive sample timestamps.")
        time_base: Fraction = in_time_base
        ns_scale: Fraction = Fraction(1_000_000_000, 1)
        # First DTS may be negative (B-frame leading-frame convention);
        # normalize so the rerun timeline starts at 0.
        first_dts: int | None = None
        for raw_packet in container.demux(in_stream):
            if raw_packet.pts is None or raw_packet.dts is None:
                continue
            filtered_packets: list[av.Packet] = bsf.filter(raw_packet) if bsf is not None else [raw_packet]
            for packet in filtered_packets:
                if packet.pts is None or packet.dts is None:
                    continue
                sample: bytes = bytes(packet)
                if codec_id == 225:  # AV_CODEC_ID_AV1
                    sample = _normalize_av1_sample_for_rerun(sample)
                if first_dts is None:
                    first_dts = packet.dts
                # Round-to-nearest matches rerun's AssetVideo computation;
                # plain int() would truncate and drift by 1 ns per frame.
                pts_ns_list.append(round(packet.pts * time_base * ns_scale))
                dts_ns_list.append(round((packet.dts - first_dts) * time_base * ns_scale))
                samples.append(sample)
                is_keyframes.append(packet.is_keyframe)
    finally:
        container.close()

    # Batched columnar logging — one rerun API call instead of per-packet.
    # Indexing by DTS keeps samples in decode order so the H.264 decoder
    # can reconstruct B/P frames; ``mux_h264_to_mp4`` reads them back in
    # the same order.
    dts_ns_array: ndarray = np.asarray(dts_ns_list, dtype=np.int64)
    rr.send_columns(
        str(video_log_path),
        indexes=[rr.TimeColumn(timeline, duration=1e-9 * dts_ns_array)],
        columns=rr.VideoStream.columns(
            sample=samples,
            is_keyframe=is_keyframes,
        ),
        recording=recording,
    )

    frame_timestamps_ns: Int[ndarray, "num_frames"] = np.sort(np.asarray(pts_ns_list, dtype=np.int64))
    return frame_timestamps_ns


def read_video_stream_from_rrd(
    rrd_path: str, video_entity: str, timeline: str
) -> tuple[rr.VideoCodec, ChunkedArray, ChunkedArray]:
    """Read a ``rr.VideoStream`` entity back from an ``.rrd`` recording.

    Args:
        rrd_path: Path to the recording on disk.
        video_entity: Entity path where ``rr.VideoStream`` was logged.
        timeline: Timeline used as the sample index when the stream was logged.

    Returns:
        ``(codec, times, samples)``. ``codec`` is the static codec component.
        ``times`` is the per-sample timeline column (nanoseconds). ``samples``
        is the per-sample encoded byte column (Annex B for H.264/H.265,
        OBU/IVF-style for AV1/VP9).
    """
    normalized_entity: str = video_entity.lstrip("/")
    query_session = RRDQuerySession(Path(rrd_path))

    codec_table = query_session.read_arrow(
        contents=normalized_entity,
        selectors=[f"{normalized_entity}:VideoStream:codec"],
        index=None,
    )
    if codec_table.num_rows == 0:
        codec_table = query_session.read_arrow(
            contents=normalized_entity,
            selectors=[f"{normalized_entity}:VideoStream:codec"],
            index=timeline,
        )
        codec_column = codec_table.column(1) if codec_table.num_columns > 1 else codec_table.column(0)
    else:
        codec_column = codec_table.column(0)

    if codec_table.num_rows == 0:
        raise ValueError(f"There's no video stream codec specified at {video_entity} for timeline {timeline}.")

    codec_value_raw = first_valid_value(
        codec_column,
        component_name=f"{normalized_entity}:VideoStream:codec",
    )
    codec_value: int = int(np.asarray(codec_value_raw).reshape(-1)[0])
    codec: rr.VideoCodec = rr.VideoCodec(codec_value)

    timestamps_and_samples = query_session.read_arrow(
        contents=normalized_entity,
        selectors=[f"{normalized_entity}:VideoStream:sample"],
        index=timeline,
    )
    if timestamps_and_samples.num_rows == 0:
        raise ValueError(f"No video samples found at {video_entity} for timeline {timeline}.")

    times: ChunkedArray = timestamps_and_samples.column(0)
    samples: ChunkedArray = timestamps_and_samples.column(1)

    return codec, times, samples


def read_h264_samples_from_rrd(rrd_path: str, video_entity: str, timeline: str) -> tuple[ChunkedArray, ChunkedArray]:
    """Read H.264 ``rr.VideoStream`` samples from an ``.rrd`` recording.

    Thin wrapper around :func:`read_video_stream_from_rrd` that enforces
    H.264. Kept for back-compat with callers that assume H.264 (e.g.
    :func:`mux_h264_to_mp4`).
    """
    codec, times, samples = read_video_stream_from_rrd(rrd_path, video_entity, timeline)
    if codec != rr.VideoCodec.H264:
        raise ValueError(
            f"Video stream codec is not H.264 at {video_entity} for timeline {timeline}. "
            f"Got {hex(codec.value)}, but the value for H.264 is {hex(rr.VideoCodec.H264.value)}."
        )
    return times, samples


def extract_asset_video_blob_fast(
    video_entity: str,
    timeline: str = "video_time",
    *,
    query_session: RRDQuerySession | None = None,
    rrd_path: Path | str | None = None,
) -> bytes:
    """Extract AssetVideo blob bytes from a Rerun recording using fast pyarrow buffer access.

    This method is ~680x faster than the slow as_py() approach for large videos.
    It directly accesses the underlying pyarrow buffer without creating
    intermediate Python objects.

    Args:
        video_entity: Entity path (without leading ``/``) containing the AssetVideo component.
        timeline: Timeline used to index the recording view.
        query_session: Optional shared RRD query session for catalog reads.
        rrd_path: Optional RRD path used to create a temporary query session.

    Returns:
        Video bytes suitable for TorchCodec VideoDecoder.

    Raises:
        ValueError: If no AssetVideo blob found.
    """
    import pyarrow as pa

    normalized_entity: str = video_entity.lstrip("/")
    blob_column: str = f"{normalized_entity}:AssetVideo:blob"

    active_session = query_session
    if active_session is None and rrd_path is not None:
        active_session = RRDQuerySession(Path(rrd_path))
    if active_session is None:
        raise ValueError("extract_asset_video_blob_fast requires either query_session or rrd_path.")

    table = active_session.read_arrow(
        contents=normalized_entity,
        selectors=[blob_column],
        index=None,
    )
    blob_column_idx = 0
    if table.num_rows == 0:
        table = active_session.read_arrow(
            contents=normalized_entity,
            selectors=[blob_column],
            index=timeline,
        )
        blob_column_idx = 1
    if table.num_rows == 0:
        raise ValueError(f"No AssetVideo blob found for entity {video_entity}")
    column: pa.Array | pa.ChunkedArray = table.column(blob_column_idx)

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    # FAST PATH: Access pyarrow buffer directly without Python list intermediate
    # Structure: list<list<uint8>> -> values -> list<uint8> -> values -> uint8[]
    try:
        inner_list: pa.ListArray = column.values  # type: ignore[assignment]  # Inner list<uint8>
        uint8_values: pa.UInt8Array = inner_list.values  # type: ignore[assignment]  # The actual uint8 array
        buffers = uint8_values.buffers()
        # Buffer 0 is validity bitmap (null), Buffer 1 is data
        if len(buffers) >= 2 and buffers[1] is not None:
            blob: bytes = buffers[1].to_pybytes()
            return blob
    except Exception:
        pass  # Fall back to slow path

    # SLOW FALLBACK: Use as_py() if buffer access fails
    first_row = column[0].as_py()
    first_row = unwrap_singleton_lists(first_row)
    return bytes(first_row)


def mux_h264_to_mp4(times: ChunkedArray, samples: ChunkedArray, output_path: str) -> None:
    """Mux H.264 Annex B samples to an mp4 file using PyAV."""
    # See https://pyav.basswood-io.com/docs/stable/cookbook/basics.html#remuxing

    # Flatten out sample list into a single byte buffer.
    sample_array = samples.combine_chunks()
    if isinstance(sample_array, ListArray | LargeListArray):
        sample_array = sample_array.flatten(recursive=True)
    buffer = sample_array.buffers()[1]
    if buffer is None:
        raise ValueError("Missing H.264 sample buffer.")
    sample_bytes = io.BytesIO(buffer.to_pybytes())

    # Setup samples as input container.
    input_container = av.open(sample_bytes, mode="r", format="h264")  # Input is AnnexB H.264 stream.
    input_stream = input_container.streams.video[0]

    # Setup output container.
    output_container = av.open(output_path, mode="w")
    output_stream = output_container.add_stream_from_template(input_stream)
    # Preserve nanosecond timeline from the recording to avoid fps skew when
    # remuxing. Without this, ffmpeg/pyav may infer a default time_base that
    # snaps frames to a different cadence than the recorded timestamps.
    output_stream.time_base = Fraction(1, 1_000_000_000)
    if output_stream.codec_context is not None:
        output_stream.codec_context.time_base = output_stream.time_base

    # Timestamps are made relative to the first timestamp.
    start_time = times.chunk(0)[0]
    print(f"Offsetting timestamps with start time: {start_time}")

    # Demux and mux packets.
    ns_time_base = Fraction(1, 1_000_000_000)
    for packet, time in zip(input_container.demux(input_stream), times, strict=False):
        packet.time_base = ns_time_base  # timestamps stored in nanoseconds
        packet.pts = int(time.value - start_time.value)
        packet.dts = packet.pts  # dts == pts since there's no B-frames.
        packet.stream = output_stream
        output_container.mux(packet)

    input_container.close()
    output_container.close()
