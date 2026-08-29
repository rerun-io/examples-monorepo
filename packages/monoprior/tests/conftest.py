"""Shared test helpers for monoprior."""

import pytest
import torch
from jaxtyping import Float32, UInt8
from torch import Tensor

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
"""Skip without a GPU; use for tests that load a checkpoint or build an engine."""

slow_cuda = pytest.mark.slow
"""Loads a checkpoint or builds a TensorRT engine; excluded from the default ``pytest`` run, select with ``-m slow``."""


def synthetic_rgb_batch(batch_size: int, image_hw: tuple[int, int]) -> UInt8[Tensor, "b h w 3"]:
    """Create deterministic RGB gradients directly on CUDA.

    Args:
        batch_size: Number of frames to create.
        image_hw: Frame height and width.

    Returns:
        uint8 CUDA RGB frames shaped ``b h w 3``.
    """
    height: int = image_hw[0]
    width: int = image_hw[1]
    x_w: Float32[Tensor, "w"] = torch.linspace(0, 255, width, device="cuda", dtype=torch.float32)
    y_h: Float32[Tensor, "h"] = torch.linspace(0, 255, height, device="cuda", dtype=torch.float32)
    red_hw: Float32[Tensor, "h w"] = x_w[None, :].expand(height, width)
    green_hw: Float32[Tensor, "h w"] = y_h[:, None].expand(height, width)
    blue_hw: Float32[Tensor, "h w"] = (red_hw + green_hw) / 2.0
    rgb_hw3: UInt8[Tensor, "h w 3"] = torch.stack((red_hw, green_hw, blue_hw), dim=-1).to(torch.uint8)
    return rgb_hw3[None].expand(batch_size, -1, -1, -1).contiguous()
