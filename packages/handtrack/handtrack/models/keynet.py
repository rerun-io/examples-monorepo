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


class KeyNetF(nn.Module):
    """Table 5 image/keypoint fusion network plus a 161-parameter presence head."""

    def __init__(self) -> None:
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
        return KeyNetOutput(
            heatmaps=self.heatmap_head(fused),
            distance=rearrange(self.distance_head(fused), "b (joint bin) 1 1 -> b joint bin", joint=21, bin=18),
            presence_logit=self.presence_head(fused.mean(dim=(2, 3))).squeeze(-1),
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
    return KeyNetLoss(total, heatmap_loss.detach(), distance_loss.detach(), presence_loss.detach())
