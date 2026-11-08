"""GPU-side OpenCV resampling matches OpenCV on the host."""

import cv2
import numpy as np
import pytest
import torch
from jaxtyping import UInt8
from numpy import ndarray
from torch import Tensor

from monopriors.gpu_image import area_resize, area_weights


@pytest.mark.parametrize(("source_hw", "target_hw"), [((108, 192), (50, 89)), ((108, 192), (54, 96)), ((63, 112), (63, 112)), ((90, 160), (63, 112))])
def test_area_resize_matches_cv2_inter_area(source_hw: tuple[int, int], target_hw: tuple[int, int]) -> None:
    """Integer, fractional and identity ratios: OpenCV's level, off by one only on rounding ties."""
    rng: np.random.Generator = np.random.default_rng(0)
    frames: UInt8[ndarray, "n h w 3"] = rng.integers(0, 256, size=(3, *source_hw, 3), dtype=np.uint8)
    resized: UInt8[Tensor, "n 3 th tw"] = area_resize(torch.from_numpy(frames).permute(0, 3, 1, 2), *target_hw)
    for frame, ours in zip(frames, resized, strict=True):
        reference: UInt8[ndarray, "th tw 3"] = cv2.resize(frame, target_hw[::-1], interpolation=cv2.INTER_AREA)
        diff: ndarray = np.abs(ours.permute(1, 2, 0).numpy().astype(np.int16) - reference.astype(np.int16))
        assert diff.max() <= 1
        assert (diff > 0).mean() < 0.01


def test_area_weights_rows_sum_to_one_and_refuse_upscale() -> None:
    weights: ndarray = area_weights(1920, 896)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-6)
    assert (weights >= 0.0).all()
    with pytest.raises(ValueError, match="downscale only"):
        area_weights(630, 1120)
