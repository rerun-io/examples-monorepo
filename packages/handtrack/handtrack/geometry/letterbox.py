"""Pixel-centre maps from supported camera images to the network frame."""

from dataclasses import dataclass
from math import isfinite

import torch
import torch.nn.functional as F
from jaxtyping import Float32, UInt8
from torch import Tensor

NET_WIDTH: int = 640
NET_HEIGHT: int = 480


@dataclass(frozen=True, slots=True)
class Letterbox:
    """Rotate, resize, then pad a camera image horizontally."""

    source_width: int
    """Original image width in pixels."""
    source_height: int
    """Original image height in pixels."""
    quarter_turn_cw: bool
    """Rotate clockwise before resizing."""
    scale: float
    """Isotropic resize factor."""
    pad_x: float
    """Left padding in network pixels."""

    def __post_init__(self) -> None:
        """Require an exact full-height resize and nonnegative whole-pixel padding."""
        width: float = (self.source_height if self.quarter_turn_cw else self.source_width) * self.scale
        height: float = (self.source_width if self.quarter_turn_cw else self.source_height) * self.scale
        if (self.source_width <= 0 or self.source_height <= 0 or not isfinite(width)
                or not isfinite(height) or not isfinite(self.pad_x) or self.scale <= 0
                or height != NET_HEIGHT or not width.is_integer() or self.pad_x != int(self.pad_x)
                or self.pad_x < 0 or width + self.pad_x > NET_WIDTH):
            raise ValueError('Letterbox must map to 640x480 with whole-pixel horizontal padding')

    def to_net(self, uv: Float32[Tensor, '*b 2']) -> Float32[Tensor, '*b 2']:
        """Map source pixel centres into network pixel centres."""
        rotated: Float32[Tensor, '*b 2'] = torch.stack((self.source_height - 1 - uv[..., 1], uv[..., 0]), dim=-1) if self.quarter_turn_cw else uv
        scaled: Float32[Tensor, '*b 2'] = (rotated + 0.5) * self.scale - 0.5
        return torch.stack((scaled[..., 0] + self.pad_x, scaled[..., 1]), dim=-1)

    def from_net(self, uv: Float32[Tensor, '*b 2']) -> Float32[Tensor, '*b 2']:
        """Invert the network pixel-centre map."""
        rotated: Float32[Tensor, '*b 2'] = (torch.stack((uv[..., 0] - self.pad_x, uv[..., 1]), dim=-1) + 0.5) / self.scale - 0.5
        return torch.stack((rotated[..., 1], self.source_height - 1 - rotated[..., 0]), dim=-1) if self.quarter_turn_cw else rotated

    def apply(self, image: UInt8[Tensor, '*b h w']) -> UInt8[Tensor, '*b 480 640']:
        """Apply the image map on the input device, with antialiased bilinear resize and zero padding."""
        if image.shape[-2:] != (self.source_height, self.source_width):
            raise ValueError('Image shape does not match letterbox source dimensions')
        rotated: UInt8[Tensor, '*b h w'] = torch.rot90(image, -1, (-2, -1)) if self.quarter_turn_cw else image
        width: int = round(rotated.shape[-1] * self.scale)
        height: int = round(rotated.shape[-2] * self.scale)
        # Flatten arbitrary leading dimensions for interpolate, then restore them.
        resized: UInt8[Tensor, '*b h w'] = rotated if self.scale == 1.0 else F.interpolate(rotated.reshape(-1, 1, *rotated.shape[-2:]).float(), size=(height, width), mode='bilinear', antialias=True, align_corners=False).round().to(torch.uint8).reshape(*image.shape[:-2], height, width)
        return F.pad(resized, (int(self.pad_x), NET_WIDTH - width - int(self.pad_x)))


def letterbox_for(width: int, height: int) -> Letterbox:
    """Return the fixed map for UmeTrack real/synthetic or SHOW3D."""
    if (width, height) == (636, 480):
        return Letterbox(width, height, False, 1.0, 2.0)
    if (width, height) == (640, 480):
        return Letterbox(width, height, False, 1.0, 0.0)
    if (width, height) == (1024, 1280):
        return Letterbox(width, height, True, 0.46875, 20.0)
    raise ValueError(f'Unsupported camera resolution: {width}x{height}')
