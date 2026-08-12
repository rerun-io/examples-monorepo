"""Read the laser-GT subset of a CA-1M tar without altering pixels."""

import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from jaxtyping import Float64
from PIL import Image


@dataclass(frozen=True, slots=True)
class Ca1mFrame:
    """One complete CA-1M high-resolution laser-GT frame."""

    timestamp_ns: int
    """ARKit uptime timestamp encoded in the member name, in nanoseconds."""
    faro_from_camera_44: Float64[np.ndarray, "4 4"]
    """FARO-world-from-camera pose from ``RT.json``."""
    depth_png: bytes
    """Original encoded uint16 depth PNG bytes."""
    intrinsics_33: Float64[np.ndarray, "3 3"]
    """OpenCV/RDF image-from-camera matrix."""
    resolution_wh: tuple[int, int]
    """Actual PNG width and height."""


FrameParts: TypeAlias = dict[str, bytes]

_MEMBER_PATTERN: re.Pattern[str] = re.compile(r"^(?P<video_id>[^/]+)/(?P<timestamp_ns>\d+)(?P<suffix>.*)$")
_REQUIRED_SUFFIXES: dict[str, str] = {
    ".gt/RT.json": "rt",
    ".gt/depth.png": "depth",
    ".gt/depth/K.json": "intrinsics",
}


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    """Return one regular tar member's bytes."""
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"could not read CA-1M tar member {member.name}")
    return extracted.read()


def _decode_frame(timestamp_ns: int, parts: FrameParts) -> Ca1mFrame:
    """Validate and decode one complete grouped frame."""
    missing_parts: list[str] = sorted(set(_REQUIRED_SUFFIXES.values()) - parts.keys())
    if missing_parts:
        raise ValueError(f"CA-1M frame {timestamp_ns} is missing {', '.join(missing_parts)}")

    faro_from_camera_44: np.ndarray = np.asarray(json.loads(parts["rt"].decode()), dtype=np.float64)
    intrinsics_33: np.ndarray = np.asarray(json.loads(parts["intrinsics"].decode()), dtype=np.float64)
    if faro_from_camera_44.shape != (4, 4):
        raise ValueError(f"CA-1M frame {timestamp_ns} RT has shape {faro_from_camera_44.shape}, expected (4, 4)")
    if intrinsics_33.shape != (3, 3):
        raise ValueError(f"CA-1M frame {timestamp_ns} K has shape {intrinsics_33.shape}, expected (3, 3)")

    # Header-level validation only: the PNG bytes are forwarded unchanged, so a
    # full pixel decode per frame (millions across the corpus) buys nothing.
    # Mode "I;16" is PIL's 16-bit grayscale; verify() checks stream integrity.
    with Image.open(io.BytesIO(parts["depth"])) as image:
        resolution_wh: tuple[int, int] = image.size
        image_format: str | None = image.format
        image_mode: str = image.mode
        image.verify()
    if image_format != "PNG" or image_mode != "I;16":
        raise ValueError(f"CA-1M frame {timestamp_ns} depth must be a 2D uint16 PNG, got {image_format} {image_mode}")
    return Ca1mFrame(timestamp_ns, faro_from_camera_44, parts["depth"], intrinsics_33, resolution_wh)


def parse_archive(archive_path: Path, expected_video_id: str | None = None) -> list[Ca1mFrame]:
    """Read complete laser-GT frame groups from a plain CA-1M tar.

    Args:
        archive_path: Path to a plain, uncompressed CA-1M tar.
        expected_video_id: Optional capture identifier required in every selected member.

    Returns:
        Complete frames sorted by their nanosecond uptime timestamp.
    """
    grouped_parts: dict[int, FrameParts] = {}
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match: re.Match[str] | None = _MEMBER_PATTERN.fullmatch(member.name)
            if match is None:
                continue
            suffix: str = match.group("suffix")
            part_name: str | None = _REQUIRED_SUFFIXES.get(suffix)
            if part_name is None:
                continue
            video_id: str = match.group("video_id")
            if expected_video_id is not None and video_id != expected_video_id:
                raise ValueError(f"CA-1M tar contains video_id {video_id}, expected {expected_video_id}")
            timestamp_ns: int = int(match.group("timestamp_ns"))
            parts: FrameParts = grouped_parts.setdefault(timestamp_ns, {})
            if part_name in parts:
                raise ValueError(f"duplicate {part_name} member for CA-1M frame {timestamp_ns}")
            parts[part_name] = _read_member(archive, member)
    if not grouped_parts:
        raise ValueError(f"no CA-1M laser-GT frames found in {archive_path}")
    return [_decode_frame(timestamp_ns, grouped_parts[timestamp_ns]) for timestamp_ns in sorted(grouped_parts)]
