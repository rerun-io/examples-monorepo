"""MEgATrack KeyNet-F, supplement Table 5, with an added presence head.

The first image block's expansion entry 1 means no expansion (hidden width
32). All other expansion entries are absolute widths. Heatmap padding 2 is
literal: its convolutions grow 6 to 8 and 16 to 18 pixels.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from einops import rearrange
from jaxtyping import Bool, Float32
from torch import Tensor, nn
from torch.nn import functional as F

from handtrack.labels.heatmaps import DISTANCE_RANGE_MM, heatmap_to_crop
from handtrack.models.blocks import inverted_residual_stack

HeatmapReduction: TypeAlias = Literal["mean", "pixel_sum"]
"""The paper writes squared L2 norms (sums): ``pixel_sum`` sums each keypoint's 18x18 pixels (and 18 distance bins),
then averages over keypoints and positive crops. ``mean`` averages every heatmap value; using it during warm-up
is our optimisation policy."""


@dataclass(frozen=True, slots=True)
class KeyNetOutput:
    """Heatmaps and presence for a crop in left-hand orientation."""

    heatmaps: Float32[Tensor, "b 21 18 18"]
    """Nonnegative 2D keypoint heatmaps in landmark order."""
    distance: Float32[Tensor, "b 21 18"]
    """Nonnegative relative-distance heatmaps in landmark order."""
    presence_logit: Float32[Tensor, "b"]
    """Unbounded presence logit for each crop."""
    visibility_logit: Float32[Tensor, "b 21"] | None = None
    """Per keypoint: the logit that it is visible (inside the crop, not behind a hand surface); None without the visibility head."""


@dataclass(frozen=True, slots=True)
class KeyNetLoss:
    """Training objective and detached, unweighted logging terms."""

    total: Float32[Tensor, ""]
    """Heatmap MSE + 0.05 distance MSE + presence_weight BCE, with gradients."""
    heatmap: Float32[Tensor, ""]
    """Detached 2D heatmap MSE over positive crops, in the loss's ``HeatmapReduction``."""
    distance: Float32[Tensor, ""]
    """Detached 1D heatmap MSE over positive crops, in the loss's ``HeatmapReduction``."""
    presence: Float32[Tensor, ""]
    """Detached presence BCE over crops with valid presence labels."""
    visibility: Float32[Tensor, ""] | None = None
    """Detached per-keypoint visibility BCE over positive crops with visibility labels; None without the head."""
    pinch: Float32[Tensor, ""] | None = None
    """Detached pinch-relation loss (``pinch_loss``); None when not computed."""


DEFAULT_BN_EPS: float = 1e-5


class KeyNetF(nn.Module):
    """Table 5 image/keypoint fusion network plus a 161-parameter presence head, and optionally a 3,381-parameter per-keypoint
    visibility head (``visibility_head``) on the same pooled features. ``bn_eps`` sets every BatchNorm's epsilon: a larger one bounds
    how much a near-dead channel (running variance ~0) is amplified in eval mode."""

    def __init__(self, visibility_head: bool = False, bn_eps: float = DEFAULT_BN_EPS) -> None:
        super().__init__()
        self.image: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            inverted_residual_stack(
                32,
                [
                    (32, 32, 1, 1),
                    (96, 32, 2, 1),
                    (192, 32, 1, 1),
                    (192, 64, 2, 1),
                    (384, 64, 1, 2),
                ],
            ),
        )
        self.keypoints: nn.Sequential = nn.Sequential(nn.Linear(63, 4608), nn.ReLU())
        self.fused: nn.Sequential = inverted_residual_stack(
            96,
            [
                (384, 64, 1, 2),
                (384, 64, 1, 3),
                (384, 96, 1, 1),
                (480, 96, 1, 2),
                (576, 128, 2, 1),
                (768, 128, 1, 2),
                (768, 160, 1, 1),
            ],
        )
        self.heatmap_head: nn.Sequential = nn.Sequential(
            nn.Conv2d(160, 63, 3, padding=2, bias=False),
            nn.BatchNorm2d(63),
            nn.ReLU(),
            nn.ConvTranspose2d(63, 42, 2, 2, bias=True),
            nn.Conv2d(42, 21, 3, padding=2, bias=False),
            nn.BatchNorm2d(21),
            nn.ReLU(),
        )
        self.distance_head: nn.Sequential = nn.Sequential(nn.AvgPool2d(6, 6), nn.Conv2d(160, 378, 1), nn.ReLU())
        self.presence_head: nn.Linear = nn.Linear(160, 1)
        self.visibility_head: nn.Linear | None = nn.Linear(160, 21) if visibility_head else None
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = bn_eps
        if bn_eps != DEFAULT_BN_EPS:  # BatchNorm's eps is not part of a state_dict: keep a non-default one with the weights
            self.register_buffer("bn_eps", torch.tensor(bn_eps, dtype=torch.float64))

    def forward(self, crop: Float32[Tensor, "b 1 96 96"], keypoints: Float32[Tensor, "b 63"]) -> KeyNetOutput:
        """Predict heatmaps and presence from a crop and prior keypoints.

        Args:
            crop: Float32[Tensor, 'b 1 96 96'], mono crop valued in [0, 1].
            keypoints: Float32[Tensor, 'b 63'], 21 (u, v, relative-distance)
                triples, or zeros when untracked. Right hands must be mirrored
                into left-hand orientation by the caller.

        Returns:
            Table 5 heatmaps and an unbounded crop-presence logit.
        """
        image_features: Float32[Tensor, "b 64 12 12"] = self.image(crop)
        keypoint_features: Float32[Tensor, "b 32 12 12"] = rearrange(self.keypoints(keypoints), "b (c h w) -> b c h w", c=32, h=12, w=12)
        fused: Float32[Tensor, "b 160 6 6"] = self.fused(torch.cat((image_features, keypoint_features), dim=1))
        pooled: Float32[Tensor, "b 160"] = fused.mean(dim=(2, 3))
        return KeyNetOutput(
            heatmaps=self.heatmap_head(fused),
            distance=rearrange(self.distance_head(fused), "b (joint bin) 1 1 -> b joint bin", joint=21, bin=18),
            presence_logit=self.presence_head(pooled).squeeze(-1),
            visibility_logit=None if self.visibility_head is None else self.visibility_head(pooled),
        )


def keynet_loss(
    output: KeyNetOutput,
    heatmaps: Float32[Tensor, "b 21 18 18"],
    distance: Float32[Tensor, "b 21 18"],
    presence_target: Float32[Tensor, "b"],
    positive: Bool[Tensor, "b"],
    presence_mask: Bool[Tensor, "b"],
    presence_weight: float,
    heatmap_reduction: HeatmapReduction = "mean",
    heatmap_scale: float = 1.0,
    visible: Bool[Tensor, "b 21"] | None = None,
    visibility_mask: Bool[Tensor, "b"] | None = None,
    visibility_weight: float = 0.0,
    points_crop: Float32[Tensor, "b 21 2"] | None = None,
    d_rel_mm: Float32[Tensor, "b 21"] | None = None,
    pinch_weight: float = 0.0,
) -> KeyNetLoss:
    """Average each term over its valid samples, with zero for empty selections.

    With ``mean`` each heatmap MSE averages all landmarks and bins per positive
    crop, then averages the positive crops; ``pixel_sum`` sums the pixels (bins)
    of each landmark instead, so it is 324 (18) times ``mean``. BCE averages
    only presence_mask crops.
    Masked targets are removed before arithmetic, so NaN placeholders are safe.
    Empty selections contribute differentiable zero; logging terms are unweighted.

    Args:
        output: Predicted heatmaps and presence logits.
        heatmaps: Float32[Tensor, 'b 21 18 18'], target 2D heatmaps.
        distance: Float32[Tensor, 'b 21 18'], target relative-distance heatmaps.
        presence_target: Float32[Tensor, 'b'], binary presence labels.
        positive: Bool[Tensor, 'b'], crops with valid heatmap targets.
        presence_mask: Bool[Tensor, 'b'], crops with valid presence labels.
        presence_weight: Multiplier for presence BCE.
        heatmap_reduction: Reduction of both heatmap MSEs over pixels.
        heatmap_scale: Multiplier of both heatmap MSEs in the total (a warm-up ramp); the logged terms stay unscaled.

    Returns:
        Total MSE(2D) + 0.05 MSE(1D) + presence_weight BCE, and detached terms.
    """
    heatmap_errors: Float32[Tensor, "valid 21 18 18"] = output.heatmaps[positive] - heatmaps[positive]
    distance_errors: Float32[Tensor, "valid 21 18"] = output.distance[positive] - distance[positive]
    logits: Float32[Tensor, "valid"] = output.presence_logit[presence_mask]
    keypoints: int = heatmap_errors.shape[0] * heatmap_errors.shape[1]
    pixel_sum: bool = heatmap_reduction == "pixel_sum"
    heatmap_loss: Float32[Tensor, ""] = heatmap_errors.square().sum() / max(keypoints if pixel_sum else heatmap_errors.numel(), 1)
    distance_loss: Float32[Tensor, ""] = distance_errors.square().sum() / max(keypoints if pixel_sum else distance_errors.numel(), 1)
    presence_loss: Float32[Tensor, ""] = F.binary_cross_entropy_with_logits(logits, presence_target[presence_mask], reduction="sum") / max(
        logits.numel(), 1
    )
    total: Float32[Tensor, ""] = heatmap_scale * (heatmap_loss + 0.05 * distance_loss) + presence_weight * presence_loss
    visibility_loss: Float32[Tensor, ""] | None = None
    if output.visibility_logit is not None and visible is not None and visibility_mask is not None:
        chosen: Float32[Tensor, "valid 21"] = output.visibility_logit[visibility_mask]
        visibility_loss = F.binary_cross_entropy_with_logits(chosen, visible[visibility_mask].float(), reduction="sum") / max(chosen.numel(), 1)
        total = total + visibility_weight * visibility_loss
    pinch: Float32[Tensor, ""] | None = None
    if pinch_weight > 0 and points_crop is not None and d_rel_mm is not None:
        pinch = pinch_loss(output, points_crop, d_rel_mm, positive)
        total = total + heatmap_scale * pinch_weight * pinch
    return KeyNetLoss(total, heatmap_loss.detach(), distance_loss.detach(), presence_loss.detach(),
                      None if visibility_loss is None else visibility_loss.detach(), None if pinch is None else pinch.detach())


PINCH_TIPS: tuple[int, int] = (0, 1)
"""Thumb tip and index fingertip in LANDMARK order."""


def soft_points(heatmaps: Float32[Tensor, "b k 18 18"]) -> Float32[Tensor, "b k 2"]:
    """Differentiable keypoints in crop pixels: the mean of each heatmap's squared (sharpened) positive part."""
    weights: Float32[Tensor, "b k 18 18"] = heatmaps.clamp_min(0.0).square()
    weights = weights / (weights.sum(dim=(-1, -2), keepdim=True) + 1e-6)
    axis: Float32[Tensor, "18"] = torch.arange(heatmaps.shape[-1], dtype=heatmaps.dtype, device=heatmaps.device)
    points: Float32[Tensor, "b k 2"] = torch.stack(((weights.sum(-2) * axis).sum(-1), (weights.sum(-1) * axis).sum(-1)), dim=-1)
    return heatmap_to_crop(points)


def soft_distance(distance: Float32[Tensor, "b k 18"]) -> Float32[Tensor, "b k"]:
    """Differentiable relative distance in mm: the mean bin of each distance heatmap's squared positive part."""
    weights: Float32[Tensor, "b k 18"] = distance.clamp_min(0.0).square()
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
    bins: Float32[Tensor, "18"] = torch.arange(distance.shape[-1], dtype=distance.dtype, device=distance.device)
    return (weights * bins).sum(-1) * (2 * DISTANCE_RANGE_MM / (distance.shape[-1] - 1)) - DISTANCE_RANGE_MM


def pinch_loss(output: KeyNetOutput, points_crop: Float32[Tensor, "b 21 2"], d_rel_mm: Float32[Tensor, "b 21"], positive: Bool[Tensor, "b"]) -> Float32[Tensor, ""]:
    """Our pinch term, the crop-level analogue of UmeTrack's pinch loss (which needs 3D poses): the L1 error of the thumb-tip to index-tip
    vector in crop pixels plus 0.1 x the error of their relative-distance difference in mm, on positives with both tips inside the crop,
    weighted 1 + 4 exp(-|true tip vector| / 8 px) so that near-pinch configurations count up to five times more."""
    thumb, index = PINCH_TIPS
    inside: Bool[Tensor, "b"] = ((points_crop[:, [thumb, index]] >= -0.5) & (points_crop[:, [thumb, index]] < 95.5)).all(dim=(-1, -2))
    chosen: Bool[Tensor, "b"] = positive & inside
    if not bool(chosen.any()):
        return output.heatmaps.sum() * 0.0
    predicted: Float32[Tensor, "n 21 2"] = soft_points(output.heatmaps[chosen])
    depth: Float32[Tensor, "n 21"] = soft_distance(output.distance[chosen])
    truth: Float32[Tensor, "n 21 2"] = points_crop[chosen]
    vector_true: Float32[Tensor, "n 2"] = truth[:, thumb] - truth[:, index]
    vector_error: Float32[Tensor, "n"] = ((predicted[:, thumb] - predicted[:, index]) - vector_true).abs().sum(-1)
    depth_true: Float32[Tensor, "n"] = d_rel_mm[chosen][:, thumb] - d_rel_mm[chosen][:, index]
    depth_error: Float32[Tensor, "n"] = ((depth[:, thumb] - depth[:, index]) - depth_true).abs()
    weight: Float32[Tensor, "n"] = 1.0 + 4.0 * torch.exp(-vector_true.norm(dim=-1) / 8.0)
    return (weight * (vector_error + 0.1 * depth_error)).sum() / weight.sum()


def keynet_for_state(state: dict[str, Tensor]) -> KeyNetF:
    """An untrained KeyNetF shaped like ``state``: the visibility head if the weights have one, their BatchNorm eps if stored."""
    eps: float = float(state["bn_eps"]) if "bn_eps" in state else DEFAULT_BN_EPS
    return KeyNetF(visibility_head="visibility_head.weight" in state, bn_eps=eps)
