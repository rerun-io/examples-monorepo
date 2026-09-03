"""Shared metric-depth conversion helpers."""

import torch
from jaxtyping import Float, UInt16
from torch import Tensor


def quantize_depth_m_to_mm(depth_m: Float[Tensor, "*shape"]) -> UInt16[Tensor, "*shape"]:
    """Quantize torch metre depth to uint16 millimetres on-device.

    Args:
        depth_m: Floating-point metric depth tensor with arbitrary shape.
            Non-finite values become zero.

    Returns:
        Same-shape uint16 millimetres, rounded to nearest and clipped to
        ``[0, 65535]``.
    """
    safe_depth_m: Float[Tensor, "*shape"] = torch.where(torch.isfinite(depth_m), depth_m, torch.zeros_like(depth_m))
    depth_mm: UInt16[Tensor, "*shape"] = torch.round(safe_depth_m * 1000.0).clamp(0.0, 65535.0).to(dtype=torch.uint16)
    return depth_mm


__all__ = ("quantize_depth_m_to_mm",)
