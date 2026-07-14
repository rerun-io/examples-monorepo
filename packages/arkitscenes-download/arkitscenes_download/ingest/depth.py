"""Timestamped depth and confidence assets."""

import struct
import zlib
from pathlib import Path

import imagecodecs
import numpy as np

from arkitscenes_download.ingest.clock import timestamp_from_path


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """Build one length-prefixed, checksummed PNG chunk."""
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def encode_depth_png(image_hw: np.ndarray, level: int = 4) -> bytes:
    """Encode a standard 16-bit grayscale PNG with an Up filter and libdeflate.

    Level 4 favors size for ingest's write-once assets; inference-time encoders
    pass ``level=1`` for speed (11.5 ms / 596 KiB vs 22.2 ms / 531 KiB on a
    representative 1920x1440 ARKitScenes frame).
    """
    raw_hw2: np.ndarray = np.ascontiguousarray(image_hw).astype(">u2").view(np.uint8).reshape(image_hw.shape[0], -1)
    up_hw2: np.ndarray = np.empty((raw_hw2.shape[0], raw_hw2.shape[1] + 1), np.uint8)
    up_hw2[:, 0] = 2
    up_hw2[0, 1:] = raw_hw2[0]
    up_hw2[1:, 1:] = raw_hw2[1:] - raw_hw2[:-1]
    idat: bytes = imagecodecs.deflate_encode(up_hw2.tobytes(), level=level, raw=False)  # pyrefly: ignore  # bad-assignment
    ihdr: bytes = struct.pack(">IIBBBBB", image_hw.shape[1], image_hw.shape[0], 16, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def sorted_timestamped_paths(directory: Path, suffix: str = ".png") -> list[Path]:
    """List assets in timestamp order."""
    return sorted(directory.glob(f"*{suffix}"), key=timestamp_from_path)
