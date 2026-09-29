"""GPU augmentation of DetNet training batches, for cameras unlike the training rigs.

The target is RoboCap: 640x360 grey (16:9) images letterboxed into the 640x480 net frame with 60 px bars at the top and
bottom, other mounting angles and other optics. Geometry is a random similarity (scale, rotation incl. quarter turns,
shift, horizontal mirror) of the whole frame. Labels stay exact: a smallest enclosing circle maps to the smallest
enclosing circle of the mapped points under a similarity, and the keypoints are recounted to re-decide present /
partial / absent with the stream's rule (``MIN_VISIBLE_KEYPOINTS``). A mirror swaps the left and right slots.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32
from serde import serde
from torch import Tensor

from handtrack.data.batches import DetNetBatch
from handtrack.labels.validity import MIN_VISIBLE_KEYPOINTS

NET_WIDTH: float = 640.0
NET_HEIGHT: float = 480.0
WIDE_HEIGHT: float = 360.0
"""Content height of a 16:9 (640x360) image in the net frame; the rest becomes black bars."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DetNetAugment:
    """Augmentation of cached DetNet batches; off by default (the paper applies intensity scaling only)."""

    enabled: bool = False
    geometric_probability: float = 0.7
    """Share of images that get a random scale, rotation and shift; the rest keep their geometry."""
    scale_range: tuple[float, float] = (0.75, 1.3)
    """Zoom factor, drawn log-uniformly."""
    max_rotation_deg: float = 15.0
    quarter_turn_probability: float = 0.15
    """Among geometric images: add +-90 degrees (other camera mounts)."""
    max_shift: float = 0.15
    """Shift as a fraction of the frame width and height."""
    flip_probability: float = 0.5
    """Horizontal mirror; swaps the left and right hand slots."""
    wide_probability: float = 0.3
    """Black out all but a 360-row band (a 640x360 camera letterboxed into 640x480), at a random height."""
    gamma_range: tuple[float, float] = (0.7, 1.4)
    contrast_range: tuple[float, float] = (0.7, 1.3)
    noise_probability: float = 0.3
    max_noise_std: float = 0.03
    blur_probability: float = 0.2

    def __post_init__(self) -> None:
        probabilities: tuple[float, ...] = (self.geometric_probability, self.quarter_turn_probability, self.flip_probability,
                                            self.wide_probability, self.noise_probability, self.blur_probability)
        if not all(0.0 <= p <= 1.0 for p in probabilities):
            raise ValueError("augmentation probabilities must lie in [0, 1]")
        if not 0 < self.scale_range[0] <= self.scale_range[1] or not 0 < self.gamma_range[0] <= self.gamma_range[1]:
            raise ValueError("scale_range and gamma_range must be positive and ordered")
        if not 0 <= self.contrast_range[0] <= self.contrast_range[1] or self.max_shift < 0 or self.max_noise_std < 0:
            raise ValueError("contrast_range, max_shift and max_noise_std must be nonnegative")


def _uniform(generator: torch.Generator, shape: tuple[int, ...], low: float, high: float, device: torch.device) -> Float32[Tensor, "..."]:
    return low + (high - low) * torch.rand(shape, generator=generator, device=device)


def augment_detnet(batch: DetNetBatch, points: Float32[Tensor, "b 2 21 2"], in_front: Bool[Tensor, "b 2 21"],
                   settings: DetNetAugment, generator: torch.Generator) -> DetNetBatch:
    """Apply ``settings`` to one batch; ``points``/``in_front`` are the batch's net-frame keypoints (the cache stores them)."""
    device: torch.device = batch.pooled.device
    b: int = batch.pooled.shape[0]
    centre: Float32[Tensor, "2"] = torch.tensor([NET_WIDTH / 2, NET_HEIGHT / 2], device=device)

    # One similarity per image: x_out = A (x_in - centre) + centre + shift, A = s R(theta) M (M = mirror).
    geometric: Bool[Tensor, "b"] = torch.rand(b, generator=generator, device=device) < settings.geometric_probability
    log_low, log_high = math.log(settings.scale_range[0]), math.log(settings.scale_range[1])
    scale: Float32[Tensor, "b"] = torch.where(geometric, _uniform(generator, (b,), log_low, log_high, device).exp(), torch.ones(b, device=device))
    theta: Float32[Tensor, "b"] = _uniform(generator, (b,), -math.radians(settings.max_rotation_deg), math.radians(settings.max_rotation_deg), device)
    turn: Bool[Tensor, "b"] = torch.rand(b, generator=generator, device=device) < settings.quarter_turn_probability
    sign: Float32[Tensor, "b"] = torch.where(torch.rand(b, generator=generator, device=device) < 0.5, -1.0, 1.0)
    theta = torch.where(geometric, theta + turn * sign * (math.pi / 2), torch.zeros_like(theta))
    shift: Float32[Tensor, "b 2"] = _uniform(generator, (b, 2), -settings.max_shift, settings.max_shift, device) * torch.tensor([NET_WIDTH, NET_HEIGHT], device=device)
    shift = torch.where(geometric[:, None], shift, torch.zeros_like(shift))
    mirror: Bool[Tensor, "b"] = torch.rand(b, generator=generator, device=device) < settings.flip_probability
    cos, sin = theta.cos(), theta.sin()
    rotation: Float32[Tensor, "b 2 2"] = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    mirror_matrix: Float32[Tensor, "b 2 2"] = torch.diag_embed(torch.stack([torch.where(mirror, -1.0, 1.0), torch.ones(b, device=device)], -1))
    forward: Float32[Tensor, "b 2 2"] = scale[:, None, None] * rotation @ mirror_matrix

    # grid_sample maps output normalised coordinates u to input ones: u_in = D^-1 A^-1 (D u_out - shift), D = frame half-size.
    half: Float32[Tensor, "2 2"] = torch.diag(centre)
    inverse: Float32[Tensor, "b 2 2"] = torch.linalg.inv(forward)
    linear: Float32[Tensor, "b 2 2"] = torch.linalg.inv(half) @ inverse @ half
    offset: Float32[Tensor, "b 2"] = -(torch.linalg.inv(half) @ inverse @ shift[..., None])[..., 0]
    grid: Float32[Tensor, "b h w 2"] = F.affine_grid(torch.cat([linear, offset[..., None]], -1), list(batch.pooled.shape), align_corners=False)
    warped: Float32[Tensor, "b 1 h w"] = F.grid_sample(batch.pooled, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # 16:9: keep a 360-row band at a random height, black above and below.
    wide: Bool[Tensor, "b"] = torch.rand(b, generator=generator, device=device) < settings.wide_probability
    top: Float32[Tensor, "b"] = torch.where(wide, _uniform(generator, (b,), 0.0, NET_HEIGHT - WIDE_HEIGHT, device), torch.zeros(b, device=device))
    bottom: Float32[Tensor, "b"] = torch.where(wide, top + WIDE_HEIGHT, torch.full((b,), NET_HEIGHT, device=device))
    rows: Float32[Tensor, "h"] = (torch.arange(warped.shape[-2], device=device, dtype=torch.float32) + 0.5) * (NET_HEIGHT / warped.shape[-2])
    band: Bool[Tensor, "b 1 h 1"] = ((rows[None] >= top[:, None]) & (rows[None] < bottom[:, None]))[:, None, :, None]
    image: Float32[Tensor, "b 1 h w"] = warped * band

    # Photometric: gamma, contrast about the image mean, Gaussian noise, a light 3x3 blur.
    gamma: Float32[Tensor, "b 1 1 1"] = _uniform(generator, (b, 1, 1, 1), math.log(settings.gamma_range[0]), math.log(settings.gamma_range[1]), device).exp()
    image = image.clamp(0.0, 1.0).pow(gamma)
    contrast: Float32[Tensor, "b 1 1 1"] = _uniform(generator, (b, 1, 1, 1), *settings.contrast_range, device)
    mean: Float32[Tensor, "b 1 1 1"] = image.mean(dim=(-2, -1), keepdim=True)
    image = (image - mean) * contrast + mean
    noisy: Float32[Tensor, "b 1 1 1"] = (torch.rand((b, 1, 1, 1), generator=generator, device=device) < settings.noise_probability).float()
    sigma: Float32[Tensor, "b 1 1 1"] = noisy * _uniform(generator, (b, 1, 1, 1), 0.0, settings.max_noise_std, device)
    image = image + sigma * torch.randn(image.shape, generator=generator, device=device)
    kernel: Float32[Tensor, "1 1 3 3"] = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=device).div(16.0)[None, None]
    blurred: Float32[Tensor, "b 1 h w"] = F.conv2d(F.pad(image, (1, 1, 1, 1), mode="replicate"), kernel)
    blur: Bool[Tensor, "b 1 1 1"] = (torch.rand((b, 1, 1, 1), generator=generator, device=device) < settings.blur_probability)
    image = torch.where(blur, blurred, image).clamp(0.0, 1.0) * band  # the bars stay exactly black

    # Labels: move circles exactly, recount visible keypoints, then swap slots for mirrored images.
    size: Float32[Tensor, "3"] = torch.tensor([NET_WIDTH, NET_HEIGHT, NET_WIDTH], device=device)
    circle_px: Float32[Tensor, "b 2 3"] = batch.circle * size
    moved_centre: Float32[Tensor, "b 2 2"] = ((circle_px[..., :2] - centre) @ forward.transpose(-1, -2)) + centre + shift[:, None]
    moved_circle: Float32[Tensor, "b 2 3"] = torch.cat([moved_centre, circle_px[..., 2:] * scale[:, None, None]], -1) / size
    inside_before: Bool[Tensor, "b 2 21"] = (points[..., 0] >= 0) & (points[..., 0] < NET_WIDTH) & (points[..., 1] >= 0) & (points[..., 1] < NET_HEIGHT)
    moved_points: Float32[Tensor, "b 2 21 2"] = ((points - centre) @ forward[:, None].transpose(-1, -2)) + centre + shift[:, None, None]
    inside_after: Bool[Tensor, "b 2 21"] = ((moved_points[..., 0] >= 0) & (moved_points[..., 0] < NET_WIDTH)
                                            & (moved_points[..., 1] >= top[:, None, None]) & (moved_points[..., 1] < bottom[:, None, None]))
    visible: Tensor = (in_front & inside_before & inside_after).sum(-1)
    was_present: Bool[Tensor, "b 2"] = batch.circle_mask
    still_present: Bool[Tensor, "b 2"] = was_present & (visible >= MIN_VISIBLE_KEYPOINTS)
    now_partial: Bool[Tensor, "b 2"] = was_present & (visible > 0) & ~still_present
    circle: Float32[Tensor, "b 2 3"] = torch.where(still_present[..., None], moved_circle, torch.zeros_like(moved_circle))
    presence: Float32[Tensor, "b 2"] = torch.where(was_present, still_present.float(), batch.presence)
    presence_mask: Bool[Tensor, "b 2"] = batch.presence_mask & ~now_partial
    swap: Bool[Tensor, "b 1"] = mirror[:, None]
    return DetNetBatch(
        pooled=image,
        circle=torch.where(swap[..., None], circle.flip(1), circle),
        presence=torch.where(swap, presence.flip(1), presence),
        circle_mask=torch.where(swap, still_present.flip(1), still_present),
        presence_mask=torch.where(swap, presence_mask.flip(1), presence_mask),
        dataset=batch.dataset,
    )
