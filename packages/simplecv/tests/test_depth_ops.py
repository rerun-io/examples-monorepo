"""Tests for shared metric-depth conversion helpers."""

import torch
from jaxtyping import Float32, UInt16
from torch import Tensor

from simplecv.ops.depth import quantize_depth_m_to_mm


def test_torch_depth_quantization_rounds_clips_and_zeros_nonfinite() -> None:
    """Torch metre depth matches the numpy quantization policy on-device."""
    depth_m: Float32[Tensor, "6"] = torch.tensor([torch.nan, torch.inf, -1.0, 1.2344, 1.2346, 70.0], dtype=torch.float32)

    depth_mm: UInt16[Tensor, "6"] = quantize_depth_m_to_mm(depth_m)

    assert torch.equal(depth_mm, torch.tensor([0, 0, 0, 1234, 1235, 65535], dtype=torch.uint16))
