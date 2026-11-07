"""Timestamped depth and confidence assets. Depth PNGs are encoded and decoded by ``depth_png`` (packages/depth-png)."""

from enum import IntEnum
from pathlib import Path

from arkitscenes_download.ingest.clock import timestamp_from_path


class ArkitDepthConfidence(IntEnum):
    """ARKit per-pixel depth confidence, stored as uint8 in the raw confidence PNGs."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2


def sorted_timestamped_paths(directory: Path, suffix: str = ".png") -> list[Path]:
    """List assets in timestamp order."""
    return sorted(directory.glob(f"*{suffix}"), key=timestamp_from_path)
