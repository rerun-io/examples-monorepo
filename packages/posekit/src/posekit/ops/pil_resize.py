"""Device bilinear resize with Pillow's separable, fixed-point uint8 rounding.

The coefficient construction follows Pillow's published Resample.c algorithm:
https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
Only the small coefficient tables are prepared on the host; pixels stay on device.
"""

from functools import lru_cache
from math import ceil

import numpy as np
import torch
from jaxtyping import Float32, Float64, Int32, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from posekit.ops.compile import compile_unless_dev


@lru_cache(maxsize=32)
def _coefficients(source: int, target: int, device: torch.device) -> tuple[Int64[Tensor, "o k"], Int32[Tensor, "o k"]]:
    """Prepare normalized bilinear coefficients with Pillow's 22-bit precision."""
    scale: float = source / target
    support: float = max(scale, 1.0)
    taps: int = 2 * ceil(support) + 1
    centers: Float64[ndarray, "o 1"] = (np.arange(target, dtype=np.float64)[:, None] + 0.5) * scale
    start: Int64[ndarray, "o 1"] = np.maximum((centers - support + 0.5).astype(np.int64), 0)
    end: Int64[ndarray, "o 1"] = np.minimum((centers + support + 0.5).astype(np.int64), source)
    indices: Int64[ndarray, "o k"] = start + np.arange(taps, dtype=np.int64)[None]
    weights: Float64[ndarray, "o k"] = np.maximum(1.0 - np.abs((indices - centers + 0.5) / support), 0.0)
    weights[indices >= end] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)
    fixed: Int32[ndarray, "o k"] = np.floor(weights * (1 << 22) + 0.5).astype(np.int32)
    return torch.from_numpy(indices.clip(0, source - 1)).to(device), torch.from_numpy(fixed).to(device)


@compile_unless_dev
def _resize(
    frames: UInt8[Tensor, "b h w 3"],
    x_indices: Int64[Tensor, "ow kx"],
    x_weights: Int32[Tensor, "ow kx"],
    y_indices: Int64[Tensor, "oh ky"],
    y_weights: Int32[Tensor, "oh ky"],
) -> Float32[Tensor, "b 3 oh ow"]:
    """Fuse integer gathers, weighted sums, and the two uint8 rounding steps."""
    horizontal: Int32[Tensor, "b h ow 3"] = (frames[:, :, x_indices, :].to(torch.int32) * x_weights[None, None, :, :, None]).sum(
        dim=3, dtype=torch.int32
    )
    rounded: UInt8[Tensor, "b h ow 3"] = ((horizontal + (1 << 21)) >> 22).clamp(0, 255).to(torch.uint8)
    vertical: Int32[Tensor, "b oh ow 3"] = (rounded[:, y_indices, :, :].to(torch.int32) * y_weights[None, :, :, None, None]).sum(
        dim=2, dtype=torch.int32
    )
    return ((vertical + (1 << 21)) >> 22).clamp(0, 255).permute(0, 3, 1, 2).float() / 255.0


def resize_bilinear_uint8(frames: UInt8[Tensor, "b h w 3"], size: tuple[int, int]) -> Float32[Tensor, "b 3 oh ow"]:
    """Resize NHWC RGB pixels to ``(height, width)`` and normalize to [0, 1]."""
    x_indices, x_weights = _coefficients(int(frames.shape[2]), size[1], frames.device)
    y_indices, y_weights = _coefficients(int(frames.shape[1]), size[0], frames.device)
    return _resize(frames, x_indices, x_weights, y_indices, y_weights)
