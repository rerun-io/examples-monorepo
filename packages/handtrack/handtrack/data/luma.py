"""Grey images from decoded luma planes, matching what an RGB decode of the same frame gives for grey pixels."""

import torch
from jaxtyping import Bool, UInt8
from torch import Tensor

LIMITED_BLACK: float = 16.0
LIMITED_SPAN: float = 219.0
"""Limited ("TV") range codes luma 16-235; an RGB decode stretches that to 0-255."""


def grey_from_luma(y: UInt8[Tensor, 'n h w'], full_range: Bool[Tensor, 'n']) -> UInt8[Tensor, 'n h w']:
    """Full-range luma is grey as is; limited-range luma is stretched and clipped. Unspecified metadata counts as limited, as FFmpeg does."""
    stretched: UInt8[Tensor, 'n h w'] = ((y.float() - LIMITED_BLACK) * (255.0 / LIMITED_SPAN)).round().clamp(0.0, 255.0).to(torch.uint8)
    return torch.where(full_range.to(y.device)[:, None, None], y, stretched)


BT709: tuple[float, float, float] = (0.2126, 0.7152, 0.0722)
"""Luma weights of R, G, B (BT.709)."""


def grey_from_rgb(rgb: UInt8[Tensor, 'n 3 h w']) -> UInt8[Tensor, 'n h w']:
    """BT.709 luma of decoded RGB frames, on their device; a grey camera's frames (R = G = B) pass through unchanged."""
    weights: Tensor = torch.tensor(BT709, device=rgb.device)
    return torch.einsum('nchw,c->nhw', rgb.float(), weights).round().clamp(0.0, 255.0).to(torch.uint8)
