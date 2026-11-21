"""NVDEC must not hide a source/calibration resolution mismatch by resizing it."""

import numpy as np
import pytest
import torch
from fixture_types import Mp4Factory
from jaxtyping import UInt8
from numpy import ndarray

pytest.importorskip("torchcodec", reason="requires the torchcodec dependency in this environment")

from slam_rs.cuda_decode import decode_gray_cuda

pytestmark = pytest.mark.integration


@pytest.fixture
def video(synthetic_mp4: Mp4Factory) -> bytes:
    if not torch.cuda.is_available():
        pytest.skip("CUDA decoder test requires an NVIDIA GPU")
    return synthetic_mp4(np.full((64, 96, 3), 100, dtype=np.uint8), 1)


def test_cuda_rejects_source_resolution_mismatch(video: bytes) -> None:
    with pytest.raises(ValueError, match="resolution"):
        list(decode_gray_cuda(video, (32, 48)))


@pytest.mark.parametrize(("downscale", "height", "width"), [(1, 64, 96), (3, 21, 32)])
def test_cuda_preserves_calibrated_integer_downscale(video: bytes, downscale: int, height: int, width: int) -> None:
    pairs = list(decode_gray_cuda(video, (height, width), downscale))
    assert len(pairs) == 1 and all(color is None for _, color in pairs)
    frames: list[UInt8[ndarray, "h w"]] = [gray for gray, _ in pairs]
    for frame in frames:
        assert frame.shape == (height, width)
        assert frame.dtype == np.uint8
        assert frame.flags.c_contiguous
        np.testing.assert_allclose(frame, 100, atol=2)


def test_cuda_pairs_each_gray_frame_with_the_cpu_sized_colour_copy(synthetic_mp4: Mp4Factory) -> None:
    """The NVDEC copy has the CPU copy's size (each side floored) and keeps each side's colour."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA decoder test requires an NVIDIA GPU")
    picture: UInt8[ndarray, "h w 3"] = np.zeros((66, 98, 3), dtype=np.uint8)
    picture[:, :48, 0] = 220
    picture[:, 50:, 2] = 220
    pairs = list(decode_gray_cuda(synthetic_mp4(picture, 1), (66, 98), 1, color_downscale=4))
    assert len(pairs) == 1
    gray, copy = pairs[0]
    assert gray.shape == (66, 98) and copy is not None
    rgb: UInt8[ndarray, "ch cw 3"] = copy()
    assert rgb.shape == (16, 24, 3) and rgb.dtype == np.uint8
    assert rgb[:, :10, 0].min() > 150 and rgb[:, 14:, 2].min() > 150
