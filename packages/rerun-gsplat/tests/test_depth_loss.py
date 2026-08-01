"""Unit tests for the depth-loss sampling math (CPU, no catalog needed)."""

import torch
from jaxtyping import Float32
from test_segment_views import synthetic_sample
from torch import Tensor

from rerun_gsplat.apis.segment_views import SplatView, view_from_sample
from rerun_gsplat.apis.train import masked_depth_l1, sample_rendered_depth


def test_sample_rendered_depth_affine_map() -> None:
    # synthetic K: fx=100 cx=48; lowres K: fx=25 cx=12 -> u_rgb = 4 * u_lo exactly.
    view: SplatView = view_from_sample(synthetic_sample(), downscale=1)
    ramp_hw: Float32[Tensor, "64 96"] = torch.arange(96, dtype=torch.float32).expand(64, 96).contiguous()
    sampled: Float32[Tensor, "16 24"] = sample_rendered_depth(ramp_hw, view, device="cpu")
    expected: Float32[Tensor, "16 24"] = 4.0 * torch.arange(24, dtype=torch.float32).expand(16, 24)
    assert torch.allclose(sampled, expected, atol=1e-4)


def test_masked_depth_l1_uses_only_valid_pixels() -> None:
    view: SplatView = view_from_sample(synthetic_sample(), downscale=1)
    # Rendered depth = sensor depth + 0.25 m everywhere -> masked L1 is exactly 0.25.
    rendered_hw: Float32[Tensor, "64 96"] = torch.full((64, 96), 1.5 + 0.25)
    error: Float32[Tensor, ""] = masked_depth_l1(rendered_hw, view, device="cpu")
    assert torch.isclose(error, torch.tensor(0.25), atol=1e-5)
