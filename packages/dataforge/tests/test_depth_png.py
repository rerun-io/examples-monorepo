"""The depth_png module (packages/depth-png) as dataforge calls it: dtypes, layouts and errors at the Python boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import depth_png
import numpy as np
import pytest
from numpy import ndarray


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("shape", [(1, 1), (5, 3), (61, 97)])
def test_round_trip_keeps_dtype_shape_and_values(dtype: type, shape: tuple[int, int]) -> None:
    depth: ndarray = np.random.default_rng(1).integers(0, np.iinfo(dtype).max, size=shape, endpoint=True).astype(dtype)
    for level in (0, 1, 4, 9):
        decoded: ndarray = depth_png.decode(depth_png.encode(depth, level=level))
        assert decoded.dtype == dtype and decoded.shape == shape
        np.testing.assert_array_equal(decoded, depth)


def test_arrays_it_cannot_encode_are_refused() -> None:
    depth: ndarray = np.arange(12, dtype=np.uint16).reshape(3, 4)
    with pytest.raises(TypeError):
        depth_png.encode(depth.astype(np.float32))
    with pytest.raises(TypeError):
        depth_png.encode(depth.reshape(-1))
    with pytest.raises(ValueError, match="C-contiguous"):
        depth_png.encode(np.asfortranarray(depth))
    with pytest.raises(ValueError, match="C-contiguous"):
        depth_png.encode(depth[:, ::2])
    with pytest.raises(ValueError, match="0-9"):
        depth_png.encode(depth, level=10)


def test_threads_encode_and_decode_at_once() -> None:
    """The GIL is released inside the codec; concurrent calls must still return each caller's own image."""
    frames: list[ndarray] = [np.full((416, 512), value, dtype=np.uint16) + np.arange(512, dtype=np.uint16) for value in range(16)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        decoded: list[ndarray] = list(pool.map(lambda frame: depth_png.decode(depth_png.encode(frame)), frames))
    for frame, image in zip(frames, decoded, strict=True):
        np.testing.assert_array_equal(image, frame)


def banded_frame() -> ndarray:
    """A 1920x1440 millimetre frame: large enough for 12 bands of 120 rows."""
    rows, cols = np.mgrid[0:1440, 0:1920]
    return (400 + rows // 3 + cols // 5 + np.random.default_rng(2).integers(0, 4, size=(1440, 1920))).astype(np.uint16)


def test_banded_pngs_read_as_standard_pngs() -> None:
    import io

    import cv2
    from PIL import Image

    depth: ndarray = banded_frame()
    blob: bytes = depth_png.encode(depth, threads=4)
    assert b"dpBN" in blob, "a 1920x1440 frame is encoded in bands"
    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_UNCHANGED), depth)
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(blob))), depth)
    for threads in (1, 2, 8):
        np.testing.assert_array_equal(depth_png.decode(blob, threads=threads), depth)


def test_decode_into_fills_a_caller_buffer_from_bytes_or_an_array() -> None:
    depth: ndarray = banded_frame()
    blob: bytes = depth_png.encode(depth)
    out: ndarray = np.zeros_like(depth)
    assert depth_png.header(blob) == (1440, 1920, 16)
    depth_png.decode_into(np.frombuffer(blob, dtype=np.uint8), out, threads=4)
    np.testing.assert_array_equal(out, depth)
    with pytest.raises(ValueError):
        depth_png.decode_into(blob, np.zeros((1920, 1440), dtype=np.uint16))
    with pytest.raises(ValueError):
        depth_png.decode_into(blob, np.zeros((1440, 1920), dtype=np.uint8))
    with pytest.raises(TypeError):
        depth_png.decode_into(blob, np.zeros((1440, 1920), dtype=np.float32))
