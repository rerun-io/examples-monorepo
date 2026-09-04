import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation_fasternet import Aggregation
from .fnet import FASTERNET_T0_MODEL, FeatureNetFasterNet
from .submodule import (
    BasicConv2d,
    BasicDeconv2d,
    FPNLayer,
    build_correlation_volume,
    context_upsample,
    disparity_regression,
)


LAS2_MODEL_SIZE_CONFIGS = {
    "s": {
        "blocks": [1, 2, 4],
        "expanse_ratio": 4,
    },
    "m": {
        "blocks": [4, 8, 16],
        "expanse_ratio": 4,
    },
    "l": {
        "blocks": [8, 16, 32],
        "expanse_ratio": 8,
    },
}
LAS2_FEED_FORWARD_MODEL_SIZES = tuple(LAS2_MODEL_SIZE_CONFIGS)
LAS2_MODEL_SIZES = (*LAS2_FEED_FORWARD_MODEL_SIZES, "h")
DEFAULT_LAS2_MODEL_SIZE = "m"


LAS2_AGGREGATION_CONFIG = {
    "in_channels": 48,
    "left_att": True,
}


def normalize_las2_model_size(model_size=None):
    if model_size is None:
        return DEFAULT_LAS2_MODEL_SIZE

    model_size = str(model_size).lower()
    if model_size not in LAS2_MODEL_SIZES:
        choices = ", ".join(LAS2_MODEL_SIZES)
        raise ValueError(f"Unknown LAS2 model size '{model_size}'. Available options: {choices}")
    return model_size


def _aggregation_config(model_size):
    size_config = LAS2_MODEL_SIZE_CONFIGS[model_size]
    config = LAS2_AGGREGATION_CONFIG.copy()
    config["blocks"] = list(size_config["blocks"])
    config["expanse_ratio"] = size_config["expanse_ratio"]
    return config


def build_liteanystereo(model_size=DEFAULT_LAS2_MODEL_SIZE, fnet_pretrained=False, max_disp=192):
    model_size = normalize_las2_model_size(model_size)
    if model_size == "h":
        from .liteanystereov2_H import LiteAnyStereoH

        return LiteAnyStereoH(fnet_pretrained=fnet_pretrained, max_disp=max_disp)
    return LiteAnyStereoV2(model_size=model_size, fnet_pretrained=fnet_pretrained)


class LiteAnyStereoV2(nn.Module):
    """LAS2 feed-forward model used by the S, M, and L release checkpoints."""

    def __init__(self, model_size=DEFAULT_LAS2_MODEL_SIZE, fnet_pretrained=False):
        super(LiteAnyStereoV2, self).__init__()
        self.model_size = normalize_las2_model_size(model_size)
        if self.model_size == "h":
            raise ValueError("Use build_liteanystereo(model_size='h') or LiteAnyStereoH for the H release model.")

        self.fnet_name = FASTERNET_T0_MODEL
        self.fnet = FeatureNetFasterNet(pretrained=fnet_pretrained)
        self.fnet_channels = self.fnet.feature_channels

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.cost_agg = Aggregation(
            backbone_channels=self.fnet_channels,
            **_aggregation_config(self.model_size),
        )
        self.aggregation_name = "fasternet"

        self.refine_1 = nn.Sequential(
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

        self.stem_2 = nn.Sequential(
            BasicConv2d(
                3,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.LeakyReLU,
            ),
            BasicConv2d(
                16,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.ReLU,
            ),
        )
        self.refine_2 = FPNLayer(self.fnet_channels[0], 16)
        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def normalize_image(self, img):
        """Normalize RGB images in 0-255 range using ImageNet statistics."""
        return ((img / 255.0 - self.image_mean) / self.image_std).contiguous()

    def forward(self, left, right, max_disp=192, test_mode=False, kd_mode=False, jetson_mode=False):
        del jetson_mode
        left = self.normalize_image(left)
        right = self.normalize_image(right)

        features_left = self.fnet(left)
        features_right = self.fnet(right)
        cost_volume = build_correlation_volume(features_left[0], features_right[0], max_disp // 4)
        cv = self.cost_agg(cost_volume, features_left)

        prob = F.softmax(cv, dim=1)
        disp = disparity_regression(prob, max_disp // 4)

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(left))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)
        disp_up = context_upsample(disp * 4.0, spx_pred.float())

        if test_mode:
            return disp_up

        disp_linear = F.interpolate(disp, left.shape[2:], mode="bilinear", align_corners=False)
        if kd_mode:
            return [disp_up, disp_linear * 4.0], features_left, features_right
        return [disp_up, disp_linear * 4.0]


class LiteAnyStereoS(LiteAnyStereoV2):
    def __init__(self, fnet_pretrained=False):
        super().__init__(model_size="s", fnet_pretrained=fnet_pretrained)


class LiteAnyStereoM(LiteAnyStereoV2):
    def __init__(self, fnet_pretrained=False):
        super().__init__(model_size="m", fnet_pretrained=fnet_pretrained)


class LiteAnyStereoL(LiteAnyStereoV2):
    def __init__(self, fnet_pretrained=False):
        super().__init__(model_size="l", fnet_pretrained=fnet_pretrained)


LiteAnyStereo = LiteAnyStereoV2
