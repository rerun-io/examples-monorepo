"""Core VRS-to-Rerun conversion logic.

Opens a VRS file, discovers streams, iterates records in timestamp order,
and dispatches to FramePlayer / IMUPlayer based on stream type.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
from pyvrs import SyncVRSReader
from simplecv.rerun_log_utils import RerunTyroConfig

from pyvrs_viewer.blueprint import create_vrs_blueprint
from pyvrs_viewer.frame_player import FramePlayer
from pyvrs_viewer.imu_player import IMUPlayer, might_contain_imu_data
from pyvrs_viewer.video_encoder import VideoCodecChoice

logger: logging.Logger = logging.getLogger(__name__)


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


def vrs_to_rerun(config: VrsToRerunConfig) -> None:
    """Read a VRS file and log all supported streams to Rerun.

    Supported stream types:
      - Image streams (cameras): logged via FramePlayer
      - IMU streams (accel/gyro/mag): logged via IMUPlayer
      - Other streams: skipped with a log message

    Args:
        config: Configuration with VRS path and Rerun output settings.
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

    # Count flavor usage to detect non-unique flavors (e.g., Aria "device/ariane" for all)
    all_stream_ids: list[str] = sorted(reader.stream_ids)
    flavor_counts: dict[str, int] = {}
    for sid in all_stream_ids:
        flavor_val: str = str(reader.get_stream_info(sid).get("flavor", ""))
        flavor_counts[flavor_val] = flavor_counts.get(flavor_val, 0) + 1

    for stream_id in all_stream_ids:
        info: dict[str, object] = reader.get_stream_info(stream_id)
        flavor: str = str(info.get("flavor", ""))
        # Use flavor if it uniquely identifies the stream, otherwise fall back to stream_id
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

    # Iterate all records in timestamp order
    t_start: float = time.perf_counter()
    record_count: int = 0
    for record in reader:
        stream_id_str: str = str(record.stream_id)
        record_type: str = str(record.record_type)
        timestamp_sec: float = float(record.timestamp)

        metadata: dict[str, object] = _build_metadata(record)

        # Dispatch to frame player
        if stream_id_str in frame_players:
            player: FramePlayer = frame_players[stream_id_str]
            if record_type == "configuration":
                player.on_configuration_record(metadata)
            elif record_type == "data" and record.n_image_blocks > 0:
                image_spec: dict[str, object] = _build_image_spec_dict(record.image_specs[0])
                image_block = record.image_blocks[0]
                player.on_data_record(timestamp_sec, image_spec, image_block, metadata)

        # Dispatch to IMU player
        elif stream_id_str in imu_players:
            imu: IMUPlayer = imu_players[stream_id_str]
            if record_type == "configuration":
                imu.on_configuration_record(metadata)
            elif record_type == "data":
                imu.on_data_record(timestamp_sec, metadata)

        record_count += 1
        if record_count % 1000 == 0:
            logger.info("Processed %d records...", record_count)

    # Flush video encoders
    for fp in frame_players.values():
        fp.flush()

    total_sec: float = time.perf_counter() - t_start
    logger.info("Done. Processed %d records in %.1fs (%.0f records/sec).", record_count, total_sec, record_count / total_sec if total_sec > 0 else 0)

    # Print per-stream encoding stats
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
