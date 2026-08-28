"""Timestamped depth and confidence assets."""

import struct
import zlib
from enum import IntEnum
from pathlib import Path

import imagecodecs
import numpy as np
from jaxtyping import Shaped, UInt8, UInt16
from numpy import ndarray

from arkitscenes_download.ingest.clock import timestamp_from_path

_PNG_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"


class ArkitDepthConfidence(IntEnum):
    """ARKit per-pixel depth confidence, stored as uint8 in the raw confidence PNGs."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """Build one length-prefixed, checksummed PNG chunk."""
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def encode_depth_png(image_hw: UInt16[ndarray, "h w"], level: int = 4) -> bytes:
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


def inflate_depth_png_rows(blob_u8: UInt8[ndarray, "n"]) -> tuple[UInt8[ndarray, "h row_bytes"], tuple[int, int]] | None:
    """Inflate filtered rows from an encoder-contract depth PNG.

    The fast contract is a non-interlaced 16-bit grayscale PNG with the Up
    filter on every row, exactly as :func:`encode_depth_png` writes it. PNG CRCs
    are skipped because catalog ingest already validated stored blobs.

    Args:
        blob_u8: Complete encoded PNG bytes with shape ``(N,)`` and dtype uint8.

    Returns:
        Filtered rows and ``(height, width)``, or None when a valid PNG is
        outside the encoder contract.

    Raises:
        ValueError: If the PNG structure is truncated or malformed.
    """
    if blob_u8.ndim != 1:
        raise ValueError("depth PNG blob must be one-dimensional")
    blob_view: memoryview = memoryview(blob_u8)
    if len(blob_view) < len(_PNG_SIGNATURE) or bytes(blob_view[:8]) != _PNG_SIGNATURE:
        raise ValueError("invalid PNG signature")

    position: int = len(_PNG_SIGNATURE)
    width: int | None = None
    height: int | None = None
    fast_ihdr: bool = False
    idat_payloads: list[memoryview] = []
    while position < len(blob_view):
        if position + 12 > len(blob_view):
            raise ValueError("truncated PNG chunk header")
        length: int = struct.unpack_from(">I", blob_view, position)[0]
        payload_start: int = position + 8
        payload_end: int = payload_start + length
        chunk_end: int = payload_end + 4
        if chunk_end > len(blob_view):
            raise ValueError("truncated PNG chunk payload")
        tag: bytes = bytes(blob_view[position + 4 : position + 8])
        payload: memoryview = blob_view[payload_start:payload_end]
        position = chunk_end
        if tag == b"IHDR":
            if length != 13:
                raise ValueError("PNG IHDR must contain 13 bytes")
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", payload)
            if width <= 0 or height <= 0:
                raise ValueError("PNG dimensions must be positive")
            fast_ihdr = (bit_depth, color_type, compression, filter_method, interlace) == (16, 0, 0, 0, 0)
        elif tag == b"IDAT":
            idat_payloads.append(payload)
        elif tag == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("PNG has no IHDR chunk")
    if not idat_payloads:
        raise ValueError("PNG has no IDAT chunk")
    if not fast_ihdr:
        return None

    row_bytes: int = 1 + 2 * width
    inflated_size: int = height * row_bytes
    inflated_n: UInt8[ndarray, "inflated_size"] = np.empty(inflated_size, dtype=np.uint8)
    compressed: memoryview | bytes = idat_payloads[0] if len(idat_payloads) == 1 else b"".join(idat_payloads)
    inflated_result: bytes | bytearray | memoryview = imagecodecs.deflate_decode(compressed, out=memoryview(inflated_n), raw=False)
    if len(inflated_result) != inflated_size:
        raise ValueError(f"PNG inflated to {len(inflated_result)} bytes, expected {inflated_size}")
    filtered_hwb: UInt8[ndarray, "h row_bytes"] = inflated_n.reshape(height, row_bytes)
    filter_types_h: UInt8[ndarray, "h"] = filtered_hwb[:, 0]
    if not bool(np.all(filter_types_h == 2)):
        return None
    return filtered_hwb, (height, width)


def decode_depth_png_fast(blob_u8: UInt8[ndarray, "n"]) -> UInt16[ndarray, "h w"] | None:
    """Decode an encoder-contract depth PNG without a general PNG codec.

    Args:
        blob_u8: Complete encoded PNG bytes with shape ``(N,)`` and dtype uint8.

    Returns:
        Native-endian uint16 depth with shape ``(H, W)``, or None when the PNG
        is outside the encoder contract.
    """
    inflated: tuple[UInt8[ndarray, "h row_bytes"], tuple[int, int]] | None = inflate_depth_png_rows(blob_u8)
    if inflated is None:
        return None
    filtered_hwb: UInt8[ndarray, "h row_bytes"] = inflated[0]
    height: int = inflated[1][0]
    width: int = inflated[1][1]
    # Undo the Up filter: each byte is the sum of the bytes above it, modulo 256.
    reconstructed_hwb: UInt8[ndarray, "h pixel_bytes"] = np.cumsum(filtered_hwb[:, 1:], axis=0, dtype=np.uint8)
    big_endian_hw: Shaped[ndarray, "h w"] = reconstructed_hwb.view(">u2").reshape(height, width)
    return big_endian_hw.astype(np.uint16, copy=True)


def decode_depth_png(blob_u8: UInt8[ndarray, "n"]) -> UInt16[ndarray, "h w"]:
    """Decode a general PNG and enforce the depth-image contract.

    Args:
        blob_u8: Complete encoded PNG bytes with shape ``(N,)`` and dtype uint8.

    Returns:
        Native-endian uint16 depth with shape ``(H, W)``.

    Raises:
        RuntimeError: If decoding does not produce a two-dimensional uint16 image.
    """
    decoded: ndarray = imagecodecs.png_decode(np.ascontiguousarray(blob_u8))
    if decoded.dtype != np.uint16 or decoded.ndim != 2:
        raise RuntimeError("depth PNG failed to decode as a two-dimensional uint16 image")
    depth_mm_hw: UInt16[ndarray, "h w"] = decoded
    return depth_mm_hw


def sorted_timestamped_paths(directory: Path, suffix: str = ".png") -> list[Path]:
    """List assets in timestamp order."""
    return sorted(directory.glob(f"*{suffix}"), key=timestamp_from_path)
