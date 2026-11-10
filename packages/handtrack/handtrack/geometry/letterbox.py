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
    """Rotate, resize, then pad a camera image with black bars."""

    source_width: int
    """Original image width in pixels."""
    source_height: int
    """Original image height in pixels."""
    quarter_turns: int
    """Clockwise quarter turns (0-3) before resizing."""
    scale: float
    """Isotropic resize factor."""
    pad_x: float
    """Left padding in network pixels."""
    pad_y: float = 0.0
    """Top padding in network pixels."""

    def __post_init__(self) -> None:
        """Require whole-pixel resized extents and nonnegative whole-pixel padding inside 640x480."""
        if self.quarter_turns not in (0, 1, 2, 3):
            raise ValueError('quarter_turns must be 0, 1, 2 or 3')
        odd: bool = self.quarter_turns % 2 == 1
        width: float = (self.source_height if odd else self.source_width) * self.scale
        height: float = (self.source_width if odd else self.source_height) * self.scale
        pads: tuple[float, float] = (self.pad_x, self.pad_y)
        if (self.source_width <= 0 or self.source_height <= 0 or self.scale <= 0 or not all(isfinite(v) for v in (width, height, *pads))
                or not width.is_integer() or not height.is_integer() or any(p != int(p) or p < 0 for p in pads)
                or width + self.pad_x > NET_WIDTH or height + self.pad_y > NET_HEIGHT):
            raise ValueError('Letterbox must map into 640x480 with whole-pixel extents and padding')

    def to_net(self, uv: Float32[Tensor, '*b 2']) -> Float32[Tensor, '*b 2']:
        """Map source pixel centres into network pixel centres."""
        rotated: Float32[Tensor, '*b 2'] = uv
        width, height = self.source_width, self.source_height
        for _ in range(self.quarter_turns):  # one clockwise turn: (u, v) of a W x H image -> (H - 1 - v, u) of the H x W image
            rotated = torch.stack((height - 1 - rotated[..., 1], rotated[..., 0]), dim=-1)
            width, height = height, width
        scaled: Float32[Tensor, '*b 2'] = (rotated + 0.5) * self.scale - 0.5
        return torch.stack((scaled[..., 0] + self.pad_x, scaled[..., 1] + self.pad_y), dim=-1)

    def from_net(self, uv: Float32[Tensor, '*b 2']) -> Float32[Tensor, '*b 2']:
        """Invert the network pixel-centre map."""
        rotated: Float32[Tensor, '*b 2'] = (torch.stack((uv[..., 0] - self.pad_x, uv[..., 1] - self.pad_y), dim=-1) + 0.5) / self.scale - 0.5
        odd: bool = self.quarter_turns % 2 == 1
        width, height = (self.source_height, self.source_width) if odd else (self.source_width, self.source_height)
        for _ in range(self.quarter_turns):  # one counter-clockwise turn: (u, v) of a W x H image -> (v, W - 1 - u) of the H x W image
            rotated = torch.stack((rotated[..., 1], width - 1 - rotated[..., 0]), dim=-1)
            width, height = height, width
        return rotated

    def apply(self, image: UInt8[Tensor, '*b h w']) -> UInt8[Tensor, '*b 480 640']:
        """Apply the image map on the input device, with antialiased bilinear resize and zero padding."""
        if image.shape[-2:] != (self.source_height, self.source_width):
            raise ValueError('Image shape does not match letterbox source dimensions')
        rotated: UInt8[Tensor, '*b h w'] = torch.rot90(image, -self.quarter_turns, (-2, -1)) if self.quarter_turns else image
        width: int = round(rotated.shape[-1] * self.scale)
        height: int = round(rotated.shape[-2] * self.scale)
        # Flatten arbitrary leading dimensions for interpolate, then restore them.
        resized: UInt8[Tensor, '*b h w'] = rotated if self.scale == 1.0 else F.interpolate(rotated.reshape(-1, 1, *rotated.shape[-2:]).float(), size=(height, width), mode='bilinear', antialias=True, align_corners=False).round().to(torch.uint8).reshape(*image.shape[:-2], height, width)
        return F.pad(resized, (int(self.pad_x), NET_WIDTH - width - int(self.pad_x), int(self.pad_y), NET_HEIGHT - height - int(self.pad_y)))


FITS: dict[tuple[int, int, bool], tuple[float, float, float]] = {
    (636, 480, False): (1.0, 2.0, 0.0), (636, 480, True): (0.75, 140.0, 1.0),
    (640, 480, False): (1.0, 0.0, 0.0), (640, 480, True): (0.75, 140.0, 0.0),
    (1024, 1280, False): (0.375, 128.0, 0.0), (1024, 1280, True): (0.46875, 20.0, 0.0),
    (512, 640, False): (0.75, 128.0, 0.0), (512, 640, True): (0.9375, 20.0, 0.0),
    (1920, 1080, False): (1.0 / 3.0, 0.0, 60.0), (1920, 1080, True): (0.25, 185.0, 0.0),
    (1600, 1300, False): (0.36, 32.0, 6.0), (1600, 1300, True): (0.3, 125.0, 0.0),
}
"""(width, height, turned a quarter) -> (scale, pad_x, pad_y): the largest whole-pixel fit into 640x480, centred."""


def letterbox_for(width: int, height: int, quarter_turns: int | None = None) -> Letterbox:
    """The fixed map for UmeTrack real/synthetic, SHOW3D/HOT3D Quest 3, GSI six-view or Gen-HumanEgo, after ``quarter_turns``
    clockwise turns. None keeps the legacy map: Quest 3 portrait frames (SHOW3D 1024x1280, HOT3D 512x640) turned once, every other camera as stored (the input
    repro-1 and the deployed DetNet-F were trained on; KeyNet and the tracker still use it)."""
    turns: int = (1 if (width, height) in ((1024, 1280), (512, 640)) else 0) if quarter_turns is None else quarter_turns
    fit: tuple[float, float, float] | None = FITS.get((width, height, turns % 2 == 1))
    if fit is None:
        raise ValueError(f'Unsupported camera resolution: {width}x{height}')
    return Letterbox(width, height, turns, *fit)
