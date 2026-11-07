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
