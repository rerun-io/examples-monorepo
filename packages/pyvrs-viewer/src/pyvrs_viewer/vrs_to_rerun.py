"""Core VRS-to-Rerun conversion logic.

Opens a VRS file, discovers streams, iterates records in timestamp order,
and dispatches to FramePlayer / IMUPlayer based on stream type.

When encode_video=True, uses a two-phase parallel pipeline:
  Phase 1: Read VRS + parallel JPEG decode to YUV (ThreadPoolExecutor)
  Phase 2: Encode + log per stream (parallel NVENC sessions via threads)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rerun as rr
from jaxtyping import UInt8
from pyvrs import SyncVRSReader
from simplecv.rerun_log_utils import RerunTyroConfig
from tqdm import tqdm
from turbojpeg import TurboJPEG

from pyvrs_viewer.blueprint import create_vrs_blueprint
from pyvrs_viewer.frame_player import FramePlayer
from pyvrs_viewer.imu_player import IMUPlayer, might_contain_imu_data
from pyvrs_viewer.video_encoder import VideoCodecChoice

logger: logging.Logger = logging.getLogger(__name__)

# Thread-local TurboJPEG instances for parallel decode
_tj: TurboJPEG = TurboJPEG()


class _DecodedFrame(NamedTuple):
    """Pre-decoded image frame ready for encoding."""

    stream_id: str
    timestamp_sec: float
    yuv_planes: list[UInt8[np.ndarray, "..."]]
    metadata: dict[str, object]


@dataclass
class VrsToRerunConfig:
    """Convert a VRS file to Rerun .rrd format."""

    vrs_path: Path
    """Path to the input .vrs file."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun save/connect configuration."""
    encode_video: bool = True
    """Re-encode camera streams to video codec for smaller RRD files (default: on)."""
    video_codec: VideoCodecChoice = VideoCodecChoice.AV1
    """Video codec to use when encode_video is on (default: AV1 for best compression)."""
    decode_threads: int = 8
    """Number of threads for parallel JPEG decoding."""


def _parse_recordable_type_id(stream_id: str) -> int:
    """Extract the numeric RecordableTypeId from a stream_id string like '1201-1'."""
    parts: list[str] = stream_id.split("-")
    try:
        return int(parts[0])
    except (ValueError, IndexError):
        return 0


def _build_metadata(record: object) -> dict[str, object]:
    """Extract metadata from all metadata blocks in a record."""
    metadata: dict[str, object] = {}
    n_blocks: int = getattr(record, "n_metadata_blocks", 0)
    if n_blocks > 0:
        blocks: object = record.metadata_blocks  # type: ignore[attr-defined]
        for j in range(n_blocks):
            block: object = blocks[j]
            if isinstance(block, dict):
                metadata.update(block)
    return metadata


def _build_image_spec_dict(spec: object) -> dict[str, object]:
    """Convert an ImageSpec pybind object to a plain dict."""
    result: dict[str, object] = {}
    for attr in ("image_format", "width", "height", "pixel_format", "codec_name", "codec_quality", "stride", "buffer_size"):
        if hasattr(spec, attr):
            result[attr] = getattr(spec, attr)
    return result


def _decode_jpeg_to_yuv(jpeg_bytes: bytes) -> list[UInt8[np.ndarray, "..."]]:
    """Decode JPEG bytes to YUV planes via turbojpeg (releases GIL, thread-safe)."""
    return _tj.decode_to_yuv_planes(jpeg_bytes)


def vrs_to_rerun(config: VrsToRerunConfig) -> None:
    """Read a VRS file and log all supported streams to Rerun.

    When encode_video=True, uses a parallel pipeline:
      Phase 1: Read all VRS records. IMU records are logged immediately.
               Image JPEG bytes are collected for batch parallel decoding.
      Phase 2: Decode all JPEGs to YUV in parallel (ThreadPoolExecutor).
      Phase 3: Encode YUV → video codec and log to Rerun per stream.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    vrs_path: Path = config.vrs_path
    if not vrs_path.exists():
        msg: str = f"VRS file not found: {vrs_path}"
        raise FileNotFoundError(msg)

    logger.info("Opening VRS file: %s", vrs_path)
    logger.info("Video encoding: %s", f"ON ({config.video_codec.value})" if config.encode_video else "OFF (raw images)")
    reader: SyncVRSReader = SyncVRSReader(str(vrs_path))

    # Discover streams and create players
    frame_players: dict[str, FramePlayer] = {}
    imu_players: dict[str, IMUPlayer] = {}

    all_stream_ids: list[str] = sorted(reader.stream_ids)
    flavor_counts: dict[str, int] = {}
    for sid in all_stream_ids:
        flavor_val: str = str(reader.get_stream_info(sid).get("flavor", ""))
        flavor_counts[flavor_val] = flavor_counts.get(flavor_val, 0) + 1

    for stream_id in all_stream_ids:
        info: dict[str, object] = reader.get_stream_info(stream_id)
        flavor: str = str(info.get("flavor", ""))
        entity_name: str = flavor if flavor and flavor_counts.get(flavor, 0) == 1 else stream_id
        recordable_type_id: int = _parse_recordable_type_id(stream_id)

        if reader.might_contain_images(stream_id):
            logger.info("Stream %s (%s): handled by FramePlayer", stream_id, entity_name)
            frame_players[stream_id] = FramePlayer(stream_id, entity_name, encode_video=config.encode_video, video_codec=config.video_codec)
        elif might_contain_imu_data(recordable_type_id):
            logger.info("Stream %s (%s): handled by IMUPlayer", stream_id, entity_name)
            imu_players[stream_id] = IMUPlayer(stream_id, entity_name)
        else:
            logger.info("Stream %s (%s): no player available, skipping", stream_id, entity_name)

    # Send dynamic blueprint
    camera_entities: list[str] = [p.entity_path for p in frame_players.values()]
    imu_entity_paths: list[str] = [p.entity_path for p in imu_players.values()]
    blueprint = create_vrs_blueprint(camera_entities, imu_entity_paths)
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    t_start: float = time.perf_counter()
    total_records: int = reader.n_records

    if config.encode_video:
        _process_with_parallel_encode(reader, frame_players, imu_players, total_records, config.decode_threads)
    else:
        _process_sequential(reader, frame_players, imu_players, total_records)

    # Flush video encoders
    for fp in frame_players.values():
        fp.flush()

    total_sec: float = time.perf_counter() - t_start
    logger.info("Done. Processed %d records in %.1fs (%.0f records/sec).", total_records, total_sec, total_records / total_sec if total_sec > 0 else 0)

    for sid, fp in frame_players.items():
        stats: dict[str, object] | None = fp.encoder_stats
        if stats is not None:
            logger.info(
                "Encoder stats [%s]: %s %s, %d frames, %.1fs encode time, %.1f encode fps, %.1f MB output",
                sid,
                stats["encoder"],
                stats["codec"],
                stats["frames"],  # type: ignore[arg-type]
                stats["total_encode_sec"],  # type: ignore[arg-type]
                stats["fps"],  # type: ignore[arg-type]
                stats["total_bytes"] / 1024 / 1024,  # type: ignore[operator]
            )

    if config.rr_config.save is not None:
        logger.info("RRD saved to: %s", config.rr_config.save)


def _process_with_parallel_encode(
    reader: SyncVRSReader,
    frame_players: dict[str, FramePlayer],
    imu_players: dict[str, IMUPlayer],
    total_records: int,
    decode_threads: int,
) -> None:
    """Two-phase parallel pipeline for video encoding.

    Phase 1: Read VRS records. Log IMU/config inline. Collect JPEG bytes for images.
    Phase 2: Parallel decode all JPEGs to YUV planes.
    Phase 3: Encode + log per-stream frames serially (NVENC handles the GPU work).
    """
    # Phase 1: Collect image frames and handle non-image records inline
    pending_jpeg: list[tuple[str, float, bytes, dict[str, object]]] = []

    for record in tqdm(reader, total=total_records, desc="Reading VRS", unit="rec"):
        stream_id_str: str = str(record.stream_id)
        record_type: str = str(record.record_type)
        timestamp_sec: float = float(record.timestamp)
        metadata: dict[str, object] = _build_metadata(record)

        if stream_id_str in frame_players:
            player: FramePlayer = frame_players[stream_id_str]
            if record_type == "configuration":
                player.on_configuration_record(metadata)
            elif record_type == "data" and record.n_image_blocks > 0:
                image_spec: dict[str, object] = _build_image_spec_dict(record.image_specs[0])
                image_format: str = str(image_spec.get("image_format", "raw"))
                if image_format in ("jpg", "png"):
                    jpeg_bytes: bytes = record.image_blocks[0].tobytes()
                    pending_jpeg.append((stream_id_str, timestamp_sec, jpeg_bytes, metadata))
                elif image_format == "video":
                    player.on_data_record(timestamp_sec, image_spec, record.image_blocks[0], metadata)
                else:
                    player.on_data_record(timestamp_sec, image_spec, record.image_blocks[0], metadata)

        elif stream_id_str in imu_players:
            imu: IMUPlayer = imu_players[stream_id_str]
            if record_type == "configuration":
                imu.on_configuration_record(metadata)
            elif record_type == "data":
                imu.on_data_record(timestamp_sec, metadata)

    if not pending_jpeg:
        return

    # Phase 2: Parallel JPEG decode to YUV planes
    jpeg_bytes_list: list[bytes] = [item[2] for item in pending_jpeg]
    logger.info("Decoding %d JPEG frames with %d threads...", len(jpeg_bytes_list), decode_threads)

    with ThreadPoolExecutor(max_workers=decode_threads) as pool:
        yuv_results: list[list[UInt8[np.ndarray, "..."]]] = list(
            tqdm(pool.map(_decode_jpeg_to_yuv, jpeg_bytes_list), total=len(jpeg_bytes_list), desc="Decoding JPEG", unit="img")
        )

    # Phase 3: Encode + log per frame in VRS order (preserves timestamp ordering)
    for (stream_id_str, timestamp_sec, _jpeg, metadata), yuv_planes in tqdm(
        zip(pending_jpeg, yuv_results, strict=True), total=len(pending_jpeg), desc="Encoding video", unit="frame"
    ):
        frame_players[stream_id_str].encode_and_log_yuv(timestamp_sec, yuv_planes, metadata)


def _process_sequential(
    reader: SyncVRSReader,
    frame_players: dict[str, FramePlayer],
    imu_players: dict[str, IMUPlayer],
    total_records: int,
) -> None:
    """Simple sequential processing (no video encoding or passthrough)."""
    for record in tqdm(reader, total=total_records, desc="Processing VRS", unit="rec"):
        stream_id_str: str = str(record.stream_id)
        record_type: str = str(record.record_type)
        timestamp_sec: float = float(record.timestamp)
        metadata: dict[str, object] = _build_metadata(record)

        if stream_id_str in frame_players:
            player: FramePlayer = frame_players[stream_id_str]
            if record_type == "configuration":
                player.on_configuration_record(metadata)
            elif record_type == "data" and record.n_image_blocks > 0:
                image_spec: dict[str, object] = _build_image_spec_dict(record.image_specs[0])
                image_block = record.image_blocks[0]
                player.on_data_record(timestamp_sec, image_spec, image_block, metadata)

        elif stream_id_str in imu_players:
            imu: IMUPlayer = imu_players[stream_id_str]
            if record_type == "configuration":
                imu.on_configuration_record(metadata)
            elif record_type == "data":
                imu.on_data_record(timestamp_sec, metadata)
