"""GPU image operations on uint8 frame batches."""

import torch
from jaxtyping import UInt8


def area_downscale_uint8(frames: UInt8[torch.Tensor, "b 3 h w"], size_hw: tuple[int, int]) -> UInt8[torch.Tensor, "b 3 hh ww"]:
    """Area (box-filter) resize of uint8 frames, as ``interpolate(mode="area")``, rounded back to uint8.

    For an integer factor (3840x2160 -> 1280x720 is exactly 3x) area interpolation is the mean of each
    factor x factor block, i.e. ``avg_pool2d`` with that kernel and stride; torch runs ``mode="area"`` as
    ``adaptive_avg_pool2d``, whose kernel fills a few percent of an RTX 5090 at 4K (1.2 ms for 4 frames).
    Other sizes keep ``interpolate(mode="area")``. (Not cv2's INTER_AREA, which rounds half up on integer
    ratios.)
    """
    h, w = int(frames.shape[2]), int(frames.shape[3])
    factor_h, factor_w = h // size_hw[0], w // size_hw[1]
    if factor_h >= 1 and factor_h == factor_w and h == size_hw[0] * factor_h and w == size_hw[1] * factor_w:
        pooled: torch.Tensor = torch.nn.functional.avg_pool2d(frames.float(), kernel_size=factor_h, stride=factor_h)
    else:
        pooled = torch.nn.functional.interpolate(frames.float(), size=size_hw, mode="area")
    return pooled.round_().clamp_(0, 255).to(torch.uint8)


__all__ = ("area_downscale_uint8",)
