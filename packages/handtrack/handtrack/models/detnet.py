"""MEgATrack DetNet-F, supplement Table 4, with logits for stable training.

The table lists C x W x H; torch uses B x C x H x W. Expansion sizes are
absolute widths. Heads retain the biased convolution and batch normalization.
"""

from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Bool, Float32
from torch import Tensor, nn
from torch.nn import functional as F

from handtrack.models.blocks import inverted_residual_stack


@dataclass(frozen=True, slots=True)
class DetNetOutput:
    """Normalized circle predictions; hand slots are left, then right."""

    center: Float32[Tensor, "b 2 2"]
    """(cx / 640, cy / 480) in the net frame, regressed directly."""
    radius: Float32[Tensor, "b 2"]
    """Circle radius / 640, regressed directly."""
    presence_logit: Float32[Tensor, "b 2"]
    """Unbounded presence logits; sigmoid gives probabilities."""


@dataclass(frozen=True, slots=True)
class DetNetLoss:
    """Training objective and detached, unweighted terms for logging."""

    total: Float32[Tensor, ""]
    """circle_weight times the circle loss plus presence_weight times the presence loss, with gradients."""
    circle: Float32[Tensor, ""]
    """Detached sum of per-hand mean circle errors."""
    presence: Float32[Tensor, ""]
    """Detached sum of per-hand mean binary cross-entropies."""


@dataclass(frozen=True, slots=True)
class Detections:
    """Decoded detections in the 640 x 480 net frame, left hand first."""

    circle: Float32[Tensor, "b 2 3"]
    """(cx, cy, radius) in pixels."""
    probability: Float32[Tensor, "b 2"]
    """Presence probabilities."""
    present: Bool[Tensor, "b 2"]
    """True only when probability strictly exceeds the threshold."""
    box: Float32[Tensor, "b 2 4"]
    """Unclipped square (x0, y0, x1, y1) enclosing each circle."""


class DetNetF(nn.Module):
    """Table 4 detector, including the initial 4 x 4 average pool."""

    def __init__(self) -> None:
        super().__init__()
        self.pool: nn.AvgPool2d = nn.AvgPool2d(4, 4)
        self.backbone: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            inverted_residual_stack(
                32,
                [
                    (96, 32, 2, 1),
                    (96, 32, 1, 1),
                    (192, 64, 2, 1),
                    (384, 64, 1, 2),
                    (384, 64, 2, 1),
                    (384, 64, 1, 3),
                    (384, 96, 1, 1),
                    (576, 96, 1, 2),
                    (576, 128, 2, 1),
                    (768, 128, 1, 2),
                    (768, 160, 1, 1),
                ],
            ),
        )
        self.center_head: nn.Sequential = nn.Sequential(nn.Conv2d(160, 4, 1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1))
        self.radius_head: nn.Sequential = nn.Sequential(nn.Conv2d(160, 2, 1), nn.BatchNorm2d(2), nn.AdaptiveAvgPool2d(1))
        self.presence_head: nn.Sequential = nn.Sequential(nn.Conv2d(160, 2, 1), nn.BatchNorm2d(2), nn.AdaptiveAvgPool2d(1))

    def forward(self, frame: Float32[Tensor, "b 1 480 640"]) -> DetNetOutput:
        """Detect hands in Float32[Tensor, 'b 1 480 640'] frames valued in [0, 1]."""
        return self.forward_pooled(self.pool(frame))

    def forward_pooled(self, pooled: Float32[Tensor, "b 1 120 160"]) -> DetNetOutput:
        """Detect hands in Float32[Tensor, 'b 1 120 160'] frames already pooled 4 x 4.

        Coordinates remain normalized to the original 640 x 480 net frame.
        This runs the same layers and weights as forward, skipping only its pool.
        """
        features: Float32[Tensor, "b 160 4 5"] = self.backbone(pooled)
        return DetNetOutput(
            center=rearrange(self.center_head(features), "b (hand xy) 1 1 -> b hand xy", hand=2, xy=2),
            radius=rearrange(self.radius_head(features), "b hand 1 1 -> b hand"),
            presence_logit=rearrange(self.presence_head(features), "b hand 1 1 -> b hand"),
        )


def detnet_loss(
    output: DetNetOutput,
    target_circle: Float32[Tensor, "b 2 3"],
    presence_target: Float32[Tensor, "b 2"],
    presence_mask: Bool[Tensor, "b 2"],
    circle_mask: Bool[Tensor, "b 2"],
    *,
    circle_weight: float,
    presence_weight: float,
) -> DetNetLoss:
    """Sum the two hands' mean circle errors and mean BCEs, weighted.

    For each hand separately, circle MSE averages its three coordinates and
    the batch samples selected by circle_mask. Presence BCE averages that
    hand's samples selected by presence_mask. The two hand means are summed,
    not averaged. Each empty selection contributes a differentiable zero.
    Masked targets are excluded before arithmetic, including NaN placeholders.

    Args:
        output: Normalized predictions and unbounded presence logits.
        target_circle: Float32[Tensor, 'b 2 3'], (cx/640, cy/480, r/640).
        presence_target: Float32[Tensor, 'b 2'], binary presence labels.
        presence_mask: Bool[Tensor, 'b 2'], excludes partly visible hands.
        circle_mask: Bool[Tensor, 'b 2'], excludes absent or partly visible hands.
        circle_weight: Multiplier for the summed circle MSE.
        presence_weight: Multiplier for the summed presence BCE.

    Returns:
        Differentiable total with detached, unweighted logging terms.
    """
    predicted_circle: Float32[Tensor, "b 2 3"] = torch.cat((output.center, output.radius.unsqueeze(-1)), dim=-1)
    circle_terms: list[Float32[Tensor, ""]] = []
    presence_terms: list[Float32[Tensor, ""]] = []
    for hand in range(2):
        errors: Float32[Tensor, "valid 3"] = predicted_circle[:, hand][circle_mask[:, hand]] - target_circle[:, hand][circle_mask[:, hand]]
        circle_terms.append(errors.square().sum() / max(errors.numel(), 1))
        logits: Float32[Tensor, "valid"] = output.presence_logit[:, hand][presence_mask[:, hand]]
        targets: Float32[Tensor, "valid"] = presence_target[:, hand][presence_mask[:, hand]]
        presence_terms.append(F.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / max(logits.numel(), 1))
    circle: Float32[Tensor, ""] = torch.stack(circle_terms).sum()
    presence: Float32[Tensor, ""] = torch.stack(presence_terms).sum()
    return DetNetLoss(circle_weight * circle + presence_weight * presence, circle.detach(), presence.detach())


def decode_detections(output: DetNetOutput, threshold: float = 0.5) -> Detections:
    """Scale normalized circles to net pixels and enclose them in square boxes.

    Args:
        output: Normalized circles and presence logits, left slot then right.
        threshold: Strict lower bound on presence probability.

    Returns:
        Pixel circles, probabilities, presence flags, and unclipped square boxes.
        Radius uses frame width (640), including for vertical box extents.
    """
    center: Float32[Tensor, "b 2 2"] = output.center * output.center.new_tensor([640.0, 480.0])
    radius: Float32[Tensor, "b 2 1"] = output.radius.unsqueeze(-1) * 640.0
    probability: Float32[Tensor, "b 2"] = output.presence_logit.sigmoid()
    return Detections(
        circle=torch.cat((center, radius), dim=-1),
        probability=probability,
        present=probability > threshold,
        box=torch.cat((center - radius, center + radius), dim=-1),
    )
