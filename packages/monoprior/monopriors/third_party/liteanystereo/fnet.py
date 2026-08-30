"""FasterNet feature-pyramid extractor for LiteAnyStereo V2."""

from typing import TypeAlias, cast

import timm
from jaxtyping import Float32
from timm.models._features import FeatureListNet
from torch import Tensor, nn

from monopriors.third_party.liteanystereo.submodule import BasicConv2d, FPNLayer

FeaturePyramid: TypeAlias = list[Float32[Tensor, "b _channels _height _width"]]

FASTERNET_T0_MODEL: str = "fasternet_t0"
LAS2_FEATURE_CHANNELS: list[int] = [40, 80, 160, 320]


class FeatureNetFasterNet(nn.Module):
    """Extract the FasterNet feature pyramid used by all LAS2 release models."""

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize the feature extractor.

        Args:
            pretrained: Whether timm should load pretrained FasterNet weights.
        """
        super().__init__()
        self.backbone: FeatureListNet = cast(
            FeatureListNet,
            timm.create_model(
                FASTERNET_T0_MODEL,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            ),
        )

        self.feature_channels: list[int] = list(self.backbone.feature_info.channels())
        if self.feature_channels != LAS2_FEATURE_CHANNELS:
            raise ValueError(f"Expected {FASTERNET_T0_MODEL} channels {LAS2_FEATURE_CHANNELS}, got {self.feature_channels}")

        channels: list[int] = self.feature_channels
        self.fpn_layer4: FPNLayer = FPNLayer(channels[3], channels[2])
        self.fpn_layer3: FPNLayer = FPNLayer(channels[2], channels[1])
        self.fpn_layer2: FPNLayer = FPNLayer(channels[1], channels[0])
        self.out_conv: BasicConv2d = BasicConv2d(
            channels[0],
            channels[0],
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            norm_layer=nn.InstanceNorm2d,
        )

    def forward(self, images_b3hw: Float32[Tensor, "b 3 h w"]) -> FeaturePyramid:
        """Extract a four-level float32 image-feature pyramid.

        Args:
            images_b3hw: Float32 normalized RGB tensor with shape ``(batch, 3, height, width)``.

        Returns:
            Four float32 tensors, each with shape ``(batch, channels, level_height, level_width)``.
        """
        backbone_features: FeaturePyramid = self.backbone(images_b3hw)
        c2_bchw: Float32[Tensor, "b 40 h4 w4"] = backbone_features[0]
        c3_bchw: Float32[Tensor, "b 80 h8 w8"] = backbone_features[1]
        c4_bchw: Float32[Tensor, "b 160 h16 w16"] = backbone_features[2]
        c5_bchw: Float32[Tensor, "b 320 h32 w32"] = backbone_features[3]

        p4_bchw: Float32[Tensor, "b 160 h16 w16"] = self.fpn_layer4(c5_bchw, c4_bchw)
        p3_bchw: Float32[Tensor, "b 80 h8 w8"] = self.fpn_layer3(p4_bchw, c3_bchw)
        p2_bchw: Float32[Tensor, "b 40 h4 w4"] = self.fpn_layer2(p3_bchw, c2_bchw)
        output_p2_bchw: Float32[Tensor, "b 40 h4 w4"] = self.out_conv(p2_bchw)
        return [output_p2_bchw, p3_bchw, p4_bchw, c5_bchw]
