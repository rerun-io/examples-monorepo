"""Owned LiteAnyStereo V2 feed-forward inference models."""

from typing import Literal, TypeAlias, TypedDict, cast

import torch
import torch.nn.functional as F
from jaxtyping import Float32
from torch import Tensor, nn

from monopriors.third_party.liteanystereo.aggregation_fasternet import Aggregation
from monopriors.third_party.liteanystereo.fnet import FASTERNET_T0_MODEL, FeatureNetFasterNet, FeaturePyramid
from monopriors.third_party.liteanystereo.submodule import (
    BasicConv2d,
    BasicDeconv2d,
    FPNLayer,
    build_correlation_volume,
    context_upsample,
    disparity_regression,
)

LAS2FeedForwardModelSize: TypeAlias = Literal["s", "m", "l"]
LAS2ModelSize: TypeAlias = Literal["s", "m", "l", "h"]


class LAS2SizeConfig(TypedDict):
    """Aggregation settings for one feed-forward model size."""

    blocks: list[int]
    expanse_ratio: int


class LAS2AggregationConfig(TypedDict):
    """Complete constructor settings for feed-forward cost aggregation."""

    in_channels: int
    left_att: bool
    blocks: list[int]
    expanse_ratio: int


LAS2_MODEL_SIZE_CONFIGS: dict[LAS2FeedForwardModelSize, LAS2SizeConfig] = {
    "s": {"blocks": [1, 2, 4], "expanse_ratio": 4},
    "m": {"blocks": [4, 8, 16], "expanse_ratio": 4},
    "l": {"blocks": [8, 16, 32], "expanse_ratio": 8},
}
LAS2_FEED_FORWARD_MODEL_SIZES: tuple[LAS2FeedForwardModelSize, ...] = tuple(LAS2_MODEL_SIZE_CONFIGS)
LAS2_MODEL_SIZES: tuple[LAS2ModelSize, ...] = (*LAS2_FEED_FORWARD_MODEL_SIZES, "h")
DEFAULT_LAS2_MODEL_SIZE: LAS2ModelSize = "m"
LAS2_AGGREGATION_CONFIG: dict[str, int | bool] = {"in_channels": 48, "left_att": True}


def normalize_las2_model_size(model_size: object | None = None) -> LAS2ModelSize:
    """Normalize and validate a LiteAnyStereo V2 model-size selector.

    Args:
        model_size: Case-insensitive size selector, or None for the default.

    Returns:
        A normalized ``"s"``, ``"m"``, ``"l"``, or ``"h"`` selector.

    Raises:
        ValueError: If the selector does not name a release variant.
    """
    if model_size is None:
        return DEFAULT_LAS2_MODEL_SIZE

    normalized_size: str = str(model_size).lower()
    if normalized_size not in LAS2_MODEL_SIZES:
        choices: str = ", ".join(LAS2_MODEL_SIZES)
        raise ValueError(f"Unknown LAS2 model size '{normalized_size}'. Available options: {choices}")
    return cast(LAS2ModelSize, normalized_size)


def _aggregation_config(model_size: LAS2FeedForwardModelSize) -> LAS2AggregationConfig:
    """Build aggregation settings for a feed-forward model size.

    Args:
        model_size: Feed-forward LAS2 size selector.

    Returns:
        Complete aggregation constructor settings.
    """
    size_config: LAS2SizeConfig = LAS2_MODEL_SIZE_CONFIGS[model_size]
    config: LAS2AggregationConfig = {
        "in_channels": int(LAS2_AGGREGATION_CONFIG["in_channels"]),
        "left_att": bool(LAS2_AGGREGATION_CONFIG["left_att"]),
        "blocks": list(size_config["blocks"]),
        "expanse_ratio": size_config["expanse_ratio"],
    }
    return config


def build_liteanystereo(
    model_size: object | None = DEFAULT_LAS2_MODEL_SIZE,
    fnet_pretrained: bool = False,
    max_disp: int = 192,
) -> nn.Module:
    """Build one released LiteAnyStereo V2 inference architecture.

    Args:
        model_size: Case-insensitive ``"s"``, ``"m"``, ``"l"``, or ``"h"`` selector.
        fnet_pretrained: Whether timm should load pretrained FasterNet weights.
        max_disp: Full-resolution disparity range used to construct LAS2-H.

    Returns:
        The selected LiteAnyStereo V2 model.
    """
    normalized_size: LAS2ModelSize = normalize_las2_model_size(model_size)
    if normalized_size == "h":
        from monopriors.third_party.liteanystereo.liteanystereov2_H import LiteAnyStereoH

        return LiteAnyStereoH(fnet_pretrained=fnet_pretrained, max_disp=max_disp)
    return LiteAnyStereoV2(model_size=normalized_size, fnet_pretrained=fnet_pretrained)


class LiteAnyStereoV2(nn.Module):
    """LAS2 feed-forward inference model used by the S, M, and L checkpoints."""

    def __init__(self, model_size: object | None = DEFAULT_LAS2_MODEL_SIZE, fnet_pretrained: bool = False) -> None:
        """Initialize a feed-forward release model.

        Args:
            model_size: Case-insensitive ``"s"``, ``"m"``, or ``"l"`` selector.
            fnet_pretrained: Whether timm should load pretrained FasterNet weights.

        Raises:
            ValueError: If ``model_size`` selects the separately implemented H variant.
        """
        super().__init__()
        normalized_size: LAS2ModelSize = normalize_las2_model_size(model_size)
        if normalized_size == "h":
            raise ValueError("Use build_liteanystereo(model_size='h') or LiteAnyStereoH for the H release model.")
        self.model_size: LAS2FeedForwardModelSize = normalized_size

        self.fnet_name: str = FASTERNET_T0_MODEL
        self.fnet: FeatureNetFasterNet = FeatureNetFasterNet(pretrained=fnet_pretrained)
        self.fnet_channels: list[int] = self.fnet.feature_channels

        image_mean_1311: Float32[Tensor, "1 3 1 1"] = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std_1311: Float32[Tensor, "1 3 1 1"] = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.image_mean: Float32[Tensor, "1 3 1 1"]
        self.image_std: Float32[Tensor, "1 3 1 1"]
        self.register_buffer("image_mean", image_mean_1311, persistent=False)
        self.register_buffer("image_std", image_std_1311, persistent=False)

        self.cost_agg: Aggregation = Aggregation(backbone_channels=self.fnet_channels, **_aggregation_config(self.model_size))
        self.aggregation_name: str = "fasternet"

        self.refine_1: nn.Sequential = nn.Sequential(
            BasicConv2d(
                self.fnet_channels[0],
                self.fnet_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.InstanceNorm2d,
                act_layer=nn.LeakyReLU,
            ),
            BasicConv2d(
                self.fnet_channels[0],
                self.fnet_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.InstanceNorm2d,
                act_layer=nn.ReLU,
            ),
        )
        self.stem_2: nn.Sequential = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU),
        )
        self.refine_2: FPNLayer = FPNLayer(self.fnet_channels[0], 16)
        self.refine_3: BasicDeconv2d = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def normalize_image(self, image_b3hw: Float32[Tensor, "b 3 h w"]) -> Float32[Tensor, "b 3 h w"]:
        """Normalize float32 RGB images in the 0-255 range with ImageNet statistics.

        Args:
            image_b3hw: Float32 RGB tensor with shape ``(batch, 3, height, width)``.

        Returns:
            Float32 normalized RGB tensor with shape ``(batch, 3, height, width)``.
        """
        normalized_b3hw: Float32[Tensor, "b 3 h w"] = ((image_b3hw / 255.0 - self.image_mean) / self.image_std).contiguous()
        return normalized_b3hw

    def forward(
        self,
        left_b3hw: Float32[Tensor, "b 3 h w"],
        right_b3hw: Float32[Tensor, "b 3 h w"],
        max_disp: int = 192,
        test_mode: bool = False,
    ) -> Float32[Tensor, "b 1 h w"]:
        """Predict left-view disparity for a float32 stereo pair.

        Args:
            left_b3hw: Float32 left RGB tensor with shape ``(batch, 3, height, width)`` and values in ``[0, 255]``.
            right_b3hw: Float32 right RGB tensor with shape ``(batch, 3, height, width)`` and values in ``[0, 255]``.
            max_disp: Full-resolution disparity range; release S/M/L checkpoints use 192.
            test_mode: Must be true because this owned fork retains inference outputs only.

        Returns:
            Float32 disparity tensor with shape ``(batch, 1, height, width)``.

        Raises:
            ValueError: If the removed training-output mode is requested.
        """
        if not test_mode:
            raise ValueError("The owned LiteAnyStereo V2 fork supports inference only; pass test_mode=True.")

        normalized_left_b3hw: Float32[Tensor, "b 3 h w"] = self.normalize_image(left_b3hw)
        normalized_right_b3hw: Float32[Tensor, "b 3 h w"] = self.normalize_image(right_b3hw)

        features_left: FeaturePyramid = self.fnet(normalized_left_b3hw)
        features_right: FeaturePyramid = self.fnet(normalized_right_b3hw)
        cost_volume_bdhw: Float32[Tensor, "b disparities h4 w4"] = build_correlation_volume(
            features_left[0], features_right[0], max_disp // 4
        )
        aggregated_cost_bdhw: Float32[Tensor, "b disparities h4 w4"] = self.cost_agg(cost_volume_bdhw, features_left)

        probability_bdhw: Float32[Tensor, "b disparities h4 w4"] = F.softmax(aggregated_cost_bdhw, dim=1)
        disparity_b1hw: Float32[Tensor, "b 1 h4 w4"] = disparity_regression(probability_bdhw, max_disp // 4)

        refined_4_bchw: Float32[Tensor, "b 40 h4 w4"] = self.refine_1(features_left[0])
        stem_2_b16hw: Float32[Tensor, "b 16 h2 w2"] = self.stem_2(normalized_left_b3hw)
        refined_2_b16hw: Float32[Tensor, "b 16 h2 w2"] = self.refine_2(refined_4_bchw, stem_2_b16hw)
        upsample_logits_b9hw: Float32[Tensor, "b 9 h w"] = self.refine_3(refined_2_b16hw)
        upsample_weights_b9hw: Float32[Tensor, "b 9 h w"] = F.softmax(upsample_logits_b9hw, 1)
        disparity_up_b1hw: Float32[Tensor, "b 1 h w"] = context_upsample(disparity_b1hw * 4.0, upsample_weights_b9hw.float())
        return disparity_up_b1hw
