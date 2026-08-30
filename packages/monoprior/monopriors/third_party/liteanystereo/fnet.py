import torch.nn as nn
import timm
from .submodule import BasicConv2d, FPNLayer


FASTERNET_T0_MODEL = "fasternet_t0"
LAS2_FEATURE_CHANNELS = [40, 80, 160, 320]


class FeatureNetFasterNet(nn.Module):
    """FasterNet feature pyramid used by the simplified LAS2 release model."""

    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            FASTERNET_T0_MODEL,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        self.feature_channels = list(self.backbone.feature_info.channels())
        if self.feature_channels != LAS2_FEATURE_CHANNELS:
            raise ValueError(
                f"Expected {FASTERNET_T0_MODEL} channels {LAS2_FEATURE_CHANNELS}, "
                f"got {self.feature_channels}"
            )

        channels = self.feature_channels
        self.fpn_layer4 = FPNLayer(channels[3], channels[2])
        self.fpn_layer3 = FPNLayer(channels[2], channels[1])
        self.fpn_layer2 = FPNLayer(channels[1], channels[0])

        self.out_conv = BasicConv2d(
            channels[0],
            channels[0],
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            norm_layer=nn.InstanceNorm2d,
        )

    def forward(self, images):
        c2, c3, c4, c5 = self.backbone(images)

        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.fpn_layer2(p3, c2)
        p2 = self.out_conv(p2)

        return [p2, p3, p4, c5]
