"""Bit-exact tests for the canonical ARKitScenes depth PNG decoders."""

import struct
import zlib

import cv2
import numpy as np
import pytest
from jaxtyping import UInt8, UInt16
from numpy import ndarray
from numpy.testing import assert_array_equal

from arkitscenes_download.ingest.depth import decode_depth_png, decode_depth_png_fast, encode_depth_png, inflate_depth_png_rows


def _structured_depth() -> UInt16[ndarray, "h w"]:
    """Build deterministic metric depth with both smooth structure and noise."""
    rng: np.random.Generator = np.random.default_rng(7)
    y_h1: UInt16[ndarray, "h 1"] = np.linspace(100, 2800, 96, dtype=np.uint16)[:, None]
    x_1w: UInt16[ndarray, "1 w"] = np.linspace(0, 900, 128, dtype=np.uint16)[None, :]
    noise_hw: UInt16[ndarray, "h w"] = rng.integers(0, 301, (96, 128), dtype=np.uint16)
    return np.clip(y_h1 + x_1w + noise_hw, 100, 4000).astype(np.uint16)


@pytest.mark.parametrize("level", [1, 4])
def test_fast_and_general_decoders_match_ingest_pngs(level: int) -> None:
    """Decode representative Up-filtered ingest PNGs bit-exactly."""
    expected_hw: UInt16[ndarray, "h w"] = _structured_depth()
    blob_u8: UInt8[ndarray, "n"] = np.frombuffer(encode_depth_png(expected_hw, level=level), dtype=np.uint8)

    fast_hw: UInt16[ndarray, "h w"] | None = decode_depth_png_fast(blob_u8)
    general_hw: UInt16[ndarray, "h w"] = decode_depth_png(blob_u8)

    assert fast_hw is not None
    assert fast_hw.dtype == np.uint16
    assert_array_equal(fast_hw, expected_hw)
    assert_array_equal(general_hw, expected_hw)


def test_fast_decoder_declines_non_up_filters_and_general_decoder_handles_them() -> None:
    """Return None outside the fast contract without weakening the general decoder."""
    expected_hw: UInt16[ndarray, "h w"] = _structured_depth()
    encoded: tuple[bool, ndarray] = cv2.imencode(".png", expected_hw)
    assert encoded[0]
    blob_u8: UInt8[ndarray, "n"] = encoded[1]

    assert inflate_depth_png_rows(blob_u8) is None
    assert decode_depth_png_fast(blob_u8) is None
    assert_array_equal(decode_depth_png(blob_u8), expected_hw)


def test_fast_decoder_concatenates_multiple_idat_chunks() -> None:
    """Decode one zlib stream split across two consecutive IDAT chunks."""
    expected_hw: UInt16[ndarray, "h w"] = _structured_depth()
    original: bytes = encode_depth_png(expected_hw, level=1)
    position: int = 8
    chunks: list[bytes] = []
    while position < len(original):
        length: int = struct.unpack_from(">I", original, position)[0]
        tag: bytes = original[position + 4 : position + 8]
        payload: bytes = original[position + 8 : position + 8 + length]
        position += length + 12
        if tag == b"IDAT":
            midpoint: int = len(payload) // 2
            for half in (payload[:midpoint], payload[midpoint:]):
                chunks.append(struct.pack(">I", len(half)) + tag + half + struct.pack(">I", zlib.crc32(tag + half)))
        else:
            chunks.append(struct.pack(">I", length) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload)))
    blob_u8: UInt8[ndarray, "n"] = np.frombuffer(b"\x89PNG\r\n\x1a\n" + b"".join(chunks), dtype=np.uint8)

    actual_hw: UInt16[ndarray, "h w"] | None = decode_depth_png_fast(blob_u8)

    assert actual_hw is not None
    assert_array_equal(actual_hw, expected_hw)


def test_general_decoder_rejects_wrong_bit_depth() -> None:
    """Reject an 8-bit image after the decoder reports a non-depth dtype."""
    image_hw: UInt8[ndarray, "h w"] = np.arange(64, dtype=np.uint8).reshape(8, 8)
    encoded: tuple[bool, ndarray] = cv2.imencode(".png", image_hw)
    assert encoded[0]

    with pytest.raises(RuntimeError, match="uint16"):
        decode_depth_png(encoded[1])
