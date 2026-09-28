"""Gaussian labels and local log-quadratic decoding for KeyNet outputs."""

import torch
from einops import rearrange
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from handtrack.labels.crops import CROP_SIZE

HEATMAP_SIZE: int = 18
DISTANCE_BINS: int = 18
HEATMAP_SIGMA: float = 1.0
"""Our choice: Gaussian width of one heatmap pixel."""
DISTANCE_RANGE_MM: float = 130.0
"""Our choice: ±130 mm, 10.9% above 117.191 mm from 4,096 seeded generic-hand poses."""
DISTANCE_SIGMA: float = 1.0
"""Our choice: Gaussian width of one distance bin."""


def crop_to_heatmap(points_crop: Float32[Tensor, '*b 2']) -> Float32[Tensor, '*b 2']:
    """Map 96-pixel crop centres into 18-pixel heatmap centres."""
    return (points_crop + 0.5) * (HEATMAP_SIZE / CROP_SIZE) - 0.5


def heatmap_to_crop(points_heatmap: Float32[Tensor, '*b 2']) -> Float32[Tensor, '*b 2']:
    """Invert the crop-to-heatmap pixel-centre map."""
    return (points_heatmap + 0.5) * (CROP_SIZE / HEATMAP_SIZE) - 0.5


def render_heatmaps(points_crop: Float32[Tensor, 'b k 2'], sigma: float = HEATMAP_SIGMA) -> Float32[Tensor, 'b k 18 18']:
    """Sample unit-amplitude Gaussians; off-grid peaks are not renormalised."""
    if sigma <= 0:
        raise ValueError('Heatmap sigma must be positive')
    axis: Float32[Tensor, '18'] = torch.arange(HEATMAP_SIZE, dtype=torch.float32, device=points_crop.device)
    grid: Float32[Tensor, '18 18 2'] = torch.stack(torch.meshgrid(axis, axis, indexing='xy'), dim=-1)
    delta: Float32[Tensor, 'b k 18 18 2'] = grid - crop_to_heatmap(points_crop)[:, :, None, None, :]
    return torch.exp(-delta.square().sum(dim=-1) / (2 * sigma * sigma))


def _refine_peak(profiles: Float32[Tensor, '*b n'], peak: Int64[Tensor, '*b']) -> Float32[Tensor, '*b']:
    """Fit three log samples; shift the stencil inward at a boundary.

    Unlike a local soft-argmax, this exactly recovers a sampled Gaussian's
    vertex. Flat, nonpositive or nonconcave samples retain the integer peak.
    Limit extrapolation to two bins from the stencil centre for noisy outputs.
    """
    centre: Int64[Tensor, '*b'] = peak.clamp(1, profiles.shape[-1] - 2)
    indices: Int64[Tensor, '*b 3'] = centre[..., None] + torch.arange(-1, 2, device=profiles.device)
    samples: Float32[Tensor, '*b 3'] = profiles.gather(-1, indices)
    logs: Float32[Tensor, '*b 3'] = samples.clamp_min(1e-30).log()
    curvature: Float32[Tensor, '*b'] = logs[..., 0] - 2 * logs[..., 1] + logs[..., 2]
    usable: Bool[Tensor, '*b'] = (curvature < -1e-6) & (samples > 0).all(dim=-1)
    offset: Float32[Tensor, '*b'] = 0.5 * (logs[..., 0] - logs[..., 2]) / torch.where(usable, curvature, torch.ones_like(curvature))
    return torch.where(usable, centre.float() + offset.clamp(-2.0, 2.0), peak.float())


def decode_heatmaps(heatmaps: Float32[Tensor, 'b k 18 18']) -> tuple[Float32[Tensor, 'b k 2'], Float32[Tensor, 'b k']]:
    """Return crop pixels and sampled peak values using separable log-quadratic fits."""
    flat: Float32[Tensor, 'b k n'] = rearrange(heatmaps, 'b k h w -> b k (h w)')
    index: Int64[Tensor, 'b k'] = flat.argmax(dim=-1)
    x: Int64[Tensor, 'b k'] = index % HEATMAP_SIZE
    y: Int64[Tensor, 'b k'] = index // HEATMAP_SIZE
    rows: Float32[Tensor, 'b k 18'] = heatmaps.gather(-2, y[..., None, None].expand(-1, -1, 1, HEATMAP_SIZE)).squeeze(-2)
    columns: Float32[Tensor, 'b k 18'] = heatmaps.gather(-1, x[..., None, None].expand(-1, -1, HEATMAP_SIZE, 1)).squeeze(-1)
    points: Float32[Tensor, 'b k 2'] = torch.stack((_refine_peak(rows, x), _refine_peak(columns, y)), dim=-1)
    return heatmap_to_crop(points), flat.gather(-1, index[..., None]).squeeze(-1)


def render_distance(d_rel_mm: Float32[Tensor, 'b k'], sigma: float = DISTANCE_SIGMA) -> Float32[Tensor, 'b k 18']:
    """Sample Gaussians at 18 centres spanning [-R,R], clamping distances first."""
    if sigma <= 0:
        raise ValueError('Distance sigma must be positive')
    centre: Float32[Tensor, 'b k'] = (d_rel_mm.clamp(-DISTANCE_RANGE_MM, DISTANCE_RANGE_MM) + DISTANCE_RANGE_MM) * ((DISTANCE_BINS - 1) / (2 * DISTANCE_RANGE_MM))
    bins: Float32[Tensor, '18'] = torch.arange(DISTANCE_BINS, dtype=torch.float32, device=d_rel_mm.device)
    return torch.exp(-((bins - centre[..., None]) / sigma).square() * 0.5)


def decode_distance(heatmaps: Float32[Tensor, 'b k 18']) -> Float32[Tensor, 'b k']:
    """Decode a local log-quadratic peak to millimetres within [-R,R]."""
    index: Float32[Tensor, 'b k'] = _refine_peak(heatmaps, heatmaps.argmax(dim=-1)).clamp(0, DISTANCE_BINS - 1)
    return index * (2 * DISTANCE_RANGE_MM / (DISTANCE_BINS - 1)) - DISTANCE_RANGE_MM
