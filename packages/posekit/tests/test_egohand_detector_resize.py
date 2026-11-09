"""The optimized detector must retain PIL's separable uint8 resize semantics."""

import numpy as np
import pytest
import torch
from jaxtyping import Float32, UInt8
from torch import Tensor

pytest.importorskip("egohand_inference", reason="requires posekit-hands detector preprocessing")

from posekit.models.egohand import _resize_with_pil
from posekit.ops.pil_resize import resize_bilinear_uint8


@pytest.mark.parametrize("shape", [(240, 320), (1300, 1600), (1408, 1408)])
def test_detector_resize_matches_pil_pixel_levels(shape: tuple[int, int]) -> None:
    rng: np.random.Generator = np.random.default_rng(123)
    frame: UInt8[Tensor, "1 h w 3"] = torch.from_numpy(rng.integers(0, 256, (*shape, 3), dtype=np.uint8))[None]
    expected: Float32[Tensor, "1 3 640 640"] = _resize_with_pil(frame)
    actual: Float32[Tensor, "1 3 640 640"] = resize_bilinear_uint8(frame, (640, 640))
    mismatched: float = ((actual - expected).abs() > 0.5 / 255.0).float().mean().item()
    assert torch.equal(actual, expected), f"{mismatched:.2%} of detector input channels differ by a pixel level"
