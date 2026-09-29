"""Device-local crop geometry and augmentation for KeyNet."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float32, Int64, UInt8
from torch import Tensor

from handtrack.geometry.letterbox import NET_HEIGHT, NET_WIDTH
from handtrack.labels.circles import square_boxes

CROP_SIZE: int = 96
BOX_ENLARGE: float = 1.2


def crop_boxes(circles: Float32[Tensor, '*b 3']) -> Float32[Tensor, '*b 4']:
    """Enlarge the enclosing-circle square by 20 percent."""
    return square_boxes(circles, BOX_ENLARGE)


@dataclass(frozen=True, slots=True)
class CropJitter:
    """Perturb the sampling box in the network frame before mirroring."""

    rotation: Float32[Tensor, 'b']
    """Clockwise angle of the sampling box in radians (image y points down)."""
    scale: Float32[Tensor, 'b']
    """Box side multiplier; values above one zoom out."""
    shift: Float32[Tensor, 'b 2']
    """Centre offset along network x/y, as fractions of the original side."""


def crop_from_net(boxes: Float32[Tensor, 'b 4'], mirror: Bool[Tensor, 'b'], jitter: CropJitter | None = None) -> Float32[Tensor, 'b 3 3']:
    """Map box edges [x0,x1) onto crop edges [-0.5,95.5), then mirror.

    Jitter shifts the centre in network axes, enlarges the side, and rotates
    the sampling box clockwise about that shifted centre. Its inverse acts
    on points, so the same affine labels the sampled image.
    """
    side: Float32[Tensor, 'b 2'] = boxes[:, 2:] - boxes[:, :2]
    centre: Float32[Tensor, 'b 2'] = (boxes[:, 2:] + boxes[:, :2]) * 0.5
    angle: Float32[Tensor, 'b'] = boxes.new_zeros(boxes.shape[0]) if jitter is None else jitter.rotation
    valid: Bool[Tensor, ''] = torch.isfinite(boxes).all() & (side > 0).all()
    if jitter is not None:
        valid = valid & torch.isfinite(jitter.rotation).all() & torch.isfinite(jitter.scale).all() & torch.isfinite(jitter.shift).all()
        centre = centre + jitter.shift * side
        side = side * jitter.scale[:, None]
    if not bool(valid & torch.isfinite(side).all() & (side > 0).all() & torch.isfinite(centre).all()):
        raise ValueError('Crop boxes must have positive finite sides and jitter scale')
    affine: Float32[Tensor, 'b 3 3'] = boxes.new_zeros((boxes.shape[0], 3, 3))
    cosine: Float32[Tensor, 'b'] = angle.cos()
    sine: Float32[Tensor, 'b'] = angle.sin()
    affine[:, :2, :2] = torch.stack((cosine, sine, -sine, cosine), dim=-1).reshape(-1, 2, 2) * (CROP_SIZE / side)[:, :, None]
    affine[:, :2, 2] = (CROP_SIZE - 1) * 0.5 - torch.einsum('bij,bj->bi', affine[:, :2, :2], centre)
    affine[:, 2, 2] = 1.0
    affine[:, 0, :] = torch.where(mirror[:, None], -affine[:, 0, :], affine[:, 0, :])
    affine[:, 0, 2] += mirror * (CROP_SIZE - 1)
    return affine


def apply_affine(affine: Float32[Tensor, 'b 3 3'], points: Float32[Tensor, 'b n 2']) -> Float32[Tensor, 'b n 2']:
    """Apply a batch of homogeneous affine transforms to point sets."""
    return torch.einsum('bij,bnj->bni', affine[:, :2, :2], points) + affine[:, None, :2, 2]


def cut_crops(frames: UInt8[Tensor, 'f 480 640'], frame_index: Int64[Tensor, 'b'], crop_from_net: Float32[Tensor, 'b 3 3']) -> Float32[Tensor, 'b 1 96 96']:
    """Sample inverse-mapped crop centres with bilinear interpolation and zero padding."""
    axis: Float32[Tensor, '96'] = torch.arange(CROP_SIZE, device=frames.device, dtype=torch.float32)
    grid: Float32[Tensor, '96 96 2'] = torch.stack(torch.meshgrid(axis, axis, indexing='xy'), dim=-1)
    pixels: Float32[Tensor, 'b n 2'] = apply_affine(torch.linalg.inv_ex(crop_from_net).inverse, rearrange(grid, 'h w xy -> (h w) xy')[None].expand(frame_index.numel(), -1, -1))
    normalized: Float32[Tensor, 'b n 2'] = torch.stack(((pixels[..., 0] + 0.5) * (2.0 / NET_WIDTH) - 1.0, (pixels[..., 1] + 0.5) * (2.0 / NET_HEIGHT) - 1.0), dim=-1)
    return F.grid_sample(frames[frame_index, None].float(), rearrange(normalized, 'b (h w) xy -> b h w xy', h=CROP_SIZE, w=CROP_SIZE), mode='bilinear', padding_mode='zeros', align_corners=False) / 255.0


def sample_jitter(n: int, generator: torch.Generator, device: torch.device | str, max_rotation: float, scale_range: tuple[float, float], max_shift: float) -> CropJitter:
    """Draw independent uniform rotation, scale and network-axis centre shifts.

    The generator must belong to the requested device.
    """
    if max_rotation < 0 or max_shift < 0 or not 0 < scale_range[0] <= scale_range[1]:
        raise ValueError('Jitter requires nonnegative limits and an ordered positive scale range')
    draws: Float32[Tensor, 'b 4'] = torch.rand((n, 4), generator=generator, device=device)
    return CropJitter((draws[:, 0] * 2 - 1) * max_rotation, scale_range[0] + draws[:, 1] * (scale_range[1] - scale_range[0]), (draws[:, 2:] * 2 - 1) * max_shift)


def count_inside_crop(points_crop: Float32[Tensor, 'b n 2'], in_front: Bool[Tensor, 'b n'], margin_px: float = 0.0) -> Int64[Tensor, 'b']:
    """Count front-facing points inside the pixel-centre extent [-0.5,95.5) on both axes, grown by ``margin_px`` on every side."""
    low: float = -0.5 - margin_px
    high: float = CROP_SIZE - 0.5 + margin_px
    return (in_front & (points_crop >= low).all(dim=-1) & (points_crop < high).all(dim=-1)).sum(dim=-1)


def boundary_occlusion(n: int, generator: torch.Generator, device: torch.device | str, probability: float, max_fraction: float) -> Bool[Tensor, 'b 96 96']:
    """Draw one border-attached rectangle per selected crop.

    Choose the border uniformly, depth uniformly from 1..floor(96*fraction),
    and extent from two uniform border endpoints. A zero depth limit is empty.
    The generator must belong to the requested device.
    """
    if not 0.0 <= probability <= 1.0 or not 0.0 <= max_fraction <= 1.0:
        raise ValueError('Probability and maximum fraction must be in [0,1]')
    depth_limit: int = int(CROP_SIZE * max_fraction)
    if depth_limit == 0:
        return torch.zeros((n, CROP_SIZE, CROP_SIZE), dtype=torch.bool, device=device)
    draws: Float32[Tensor, 'b 5'] = torch.rand((n, 5), generator=generator, device=device)
    border: Int64[Tensor, 'b 1 1'] = (draws[:, 1] * 4).long()[:, None, None]
    depth: Int64[Tensor, 'b 1 1'] = (1 + (draws[:, 2] * depth_limit).long())[:, None, None]
    endpoints: Int64[Tensor, 'b 2'] = (draws[:, 3:] * CROP_SIZE).long().sort(dim=-1).values
    low: Int64[Tensor, 'b 1 1'] = endpoints[:, 0, None, None]
    high: Int64[Tensor, 'b 1 1'] = endpoints[:, 1, None, None] + 1
    axis: Int64[Tensor, '96'] = torch.arange(CROP_SIZE, device=device)
    x: Int64[Tensor, '1 1 96'] = axis[None, None, :]
    y: Int64[Tensor, '1 96 1'] = axis[None, :, None]
    along: Bool[Tensor, 'b 96 96'] = torch.where(border < 2, (y >= low) & (y < high), (x >= low) & (x < high))
    inward: Bool[Tensor, 'b 96 96'] = ((border == 0) & (x < depth)) | ((border == 1) & (x >= CROP_SIZE - depth)) | ((border == 2) & (y < depth)) | ((border == 3) & (y >= CROP_SIZE - depth))
    return (draws[:, 0, None, None] < probability) & along & inward


def scale_intensity(images: Float32[Tensor, 'b 1 h w'], generator: torch.Generator, low: float, high: float) -> Float32[Tensor, 'b 1 h w']:
    """Multiply each image by one uniform factor and clamp to [0,1]."""
    factors: Float32[Tensor, 'b 1 1 1'] = low + (high - low) * torch.rand((images.shape[0], 1, 1, 1), generator=generator, device=images.device)
    return (images * factors).clamp(0.0, 1.0)
