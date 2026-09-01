"""Tests for shared metric-depth conversion helpers."""

import numpy as np
import torch
from jaxtyping import Float32, UInt16
from numpy import ndarray
from numpy.testing import assert_array_equal
from torch import Tensor

from simplecv.ops.depth import depth_meters_to_uint16_mm, quantize_depth_m_to_mm


def test_numpy_depth_quantization_rounds_clips_and_zeros_nonfinite() -> None:
    """Numpy metre depth uses round-to-nearest and a zero invalid sentinel."""
    depth_m: Float32[ndarray, "6"] = np.array([np.nan, np.inf, -1.0, 1.2344, 1.2346, 70.0], dtype=np.float32)

    depth_mm: UInt16[ndarray, "6"] = depth_meters_to_uint16_mm(depth_m)

    assert_array_equal(depth_mm, np.array([0, 0, 0, 1234, 1235, 65535], dtype=np.uint16))


def test_torch_depth_quantization_rounds_clips_and_zeros_nonfinite() -> None:
    """Torch metre depth matches the numpy quantization policy on-device."""
    depth_m: Float32[Tensor, "6"] = torch.tensor([torch.nan, torch.inf, -1.0, 1.2344, 1.2346, 70.0], dtype=torch.float32)

    depth_mm: UInt16[Tensor, "6"] = quantize_depth_m_to_mm(depth_m)

    assert torch.equal(depth_mm, torch.tensor([0, 0, 0, 1234, 1235, 65535], dtype=torch.uint16))
