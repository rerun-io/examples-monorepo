import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation_fasternet import Aggregation as FasterNetAggregation, FasterNetBlock
from .fnet import FASTERNET_T0_MODEL, FeatureNetFasterNet
from .submodule import (
    BasicConv2d,
    BasicDeconv2d,
    build_gwc_volume_fast,
    context_upsample,
    disparity_regression,
)


H_BASE_MODEL_CONFIG = {
    "blocks": [4, 8, 16],
    "expanse_ratio": 4,
}

H_TAIL_CONFIG = {
    "hidden_dim": 64,
    "motion_corr_dim": 128,
    "motion_disp_dim": 32,
    "mask_inter_dim": 32,
    "mask_dim": 16,
    "stem_dim": 16,
    "spx_dim": 16,
    "spx_out_dim": 32,
}


def _sample_1d(img, x):
    """Sample a [N,C,1,W] tensor with pixel x coordinates shaped [N,1,K,1]."""
    _, _, height, width = img.shape
    assert height == 1
    xgrid = 2 * x / max(width - 1, 1) - 1
    ygrid = torch.zeros_like(xgrid)
    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", align_corners=True)


class CombinedGeoEncodingVolume:
    def __init__(self, fmap1, fmap2, geo_volume, num_levels=2):
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        init_corr = self.corr(fmap1, fmap2)
        batch, height, width, _, width2 = init_corr.shape
        batch, channels, disp, height, width = geo_volume.shape

        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(batch * height * width, channels, 1, disp)
        init_corr = init_corr.view(batch * height * width, 1, 1, width2)

        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for _ in range(self.num_levels - 1):
            geo_volume = F.avg_pool2d(geo_volume, [1, 2], stride=[1, 2])
            init_corr = F.avg_pool2d(init_corr, [1, 2], stride=[1, 2])
            self.geo_volume_pyramid.append(geo_volume)
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords, dx):
        batch, _, height, width = disp.shape
        outputs = []
        for level in range(self.num_levels):
            scale = 2 ** level
            disp_flat = disp.view(batch * height * width, 1, 1, 1) / scale
            dx_level = dx.to(device=disp.device, dtype=disp.dtype)

            geo_x = dx_level + disp_flat
            geo = _sample_1d(self.geo_volume_pyramid[level], geo_x)
            geo = geo.view(batch, height, width, -1)

            corr_x = coords.view(batch * height * width, 1, 1, 1) / scale - disp_flat + dx_level
            corr = _sample_1d(self.init_corr_pyramid[level], corr_x)
            corr = corr.view(batch, height, width, -1)

            outputs.append(geo)
            outputs.append(corr)

        return torch.cat(outputs, dim=-1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, channels, height, width1 = fmap1.shape
        _, _, _, width2 = fmap2.shape
        corr = torch.einsum("bchw,bchv->bhwv", fmap1, fmap2) / channels
        return corr.view(batch, height, width1, 1, width2).to(fmap1.dtype)


class ContextNetSharedBackbone(nn.Module):
    def __init__(self, c04, hidden_dim):
        super().__init__()
        self.hidden_conv = nn.Conv2d(c04, hidden_dim, kernel_size=3, padding=1)
        self.input_conv = nn.Conv2d(c04, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x4, x8=None, x16=None):
        return ([self.hidden_conv(x4), self.input_conv(x4)],)


class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class FasterNetConvEncoder(nn.Module):
    def __init__(self, dim, mlp_ratio=4, n_div=4):
        super().__init__()
        self.block = FasterNetBlock(dim, mlp_ratio=mlp_ratio, n_div=n_div, act_layer=nn.GELU)

    def forward(self, x):
        return self.block(x)


class DispHead(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            FasterNetConvEncoder(input_dim, mlp_ratio=4),
            FasterNetConvEncoder(input_dim, mlp_ratio=4),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class Conv2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        concat=True,
        bn=True,
        relu=True,
        rem_channels=None,
        fused_channels=None,
    ):
        super().__init__()
        norm_layer = nn.BatchNorm2d if bn else None
        act_layer = nn.LeakyReLU if relu else None
        if deconv:
            self.conv1 = BasicDeconv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.conv1 = BasicConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )

        self.concat = concat
        rem_channels = out_channels if rem_channels is None else rem_channels
        conv2_in_channels = out_channels + rem_channels if concat else out_channels
        conv2_out_channels = fused_channels if fused_channels is not None else (out_channels * 2 if concat else out_channels)
        self.conv2 = BasicConv2d(
            conv2_in_channels,
            conv2_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape[-2:] != rem.shape[-2:]:
            x = F.interpolate(x, size=rem.shape[-2:], mode="bilinear", align_corners=False)
        if self.concat:
            x = torch.cat((x, rem), dim=1)
        else:
            x = x + rem
        return self.conv2(x)


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius, volume_dim, hidden_dim, motion_corr_dim, motion_disp_dim):
        super().__init__()
        corr_planes = corr_levels * (2 * corr_radius + 1) * (volume_dim + 1)
        self.convc1 = nn.Conv2d(corr_planes, motion_corr_dim, kernel_size=1)
        self.convc2 = nn.Conv2d(motion_corr_dim, motion_corr_dim, kernel_size=3, padding=1)
        self.convd1 = nn.Conv2d(1, motion_disp_dim, kernel_size=7, padding=3)
        self.convd2 = nn.Conv2d(motion_disp_dim, motion_disp_dim, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(motion_disp_dim + motion_corr_dim, hidden_dim - 1, kernel_size=1)

    def forward(self, disp, corr):
        corr = F.relu(self.convc1(corr), inplace=True)
        corr = F.relu(self.convc2(corr), inplace=True)
        disp_feat = F.relu(self.convd1(disp), inplace=True)
        disp_feat = F.relu(self.convd2(disp_feat), inplace=True)
        out = F.relu(self.conv(torch.cat([corr, disp_feat], dim=1)), inplace=True)
        return torch.cat([out, disp], dim=1)


class RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, h, x, hx):
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, small_kernel_size=1, large_kernel_size=3):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, input_dim + hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.small_gru = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, x):
        x = self.conv0(x)
        hx = self.conv1(torch.cat([x, h], dim=1))
        return self.small_gru(h, x, hx) * att + self.large_gru(h, x, hx) * (1 - att)


class BasicSelectiveUpdateBlock(nn.Module):
    def __init__(
        self,
        corr_levels,
        corr_radius,
        volume_dim,
        hidden_dim,
        motion_corr_dim,
        motion_disp_dim,
        mask_inter_dim,
        mask_dim,
    ):
        super().__init__()
        self.encoder = BasicMotionEncoder(
            corr_levels, corr_radius, volume_dim, hidden_dim, motion_corr_dim, motion_disp_dim
        )
        self.gru04 = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        self.disp_head = DispHead(hidden_dim)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, mask_inter_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_inter_dim, mask_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, net, inp, corr, disp, att):
        motion_features = self.encoder(disp, corr)
        motion_features = torch.cat([inp[0], motion_features], dim=1)
        net[0] = self.gru04(att[0], net[0], motion_features)
        delta_disp = self.disp_head(net[0])
        mask_feat_4 = 0.25 * self.mask(net[0])
        return net, mask_feat_4, delta_disp


class LiteAnyStereoH(nn.Module):
    """LAS2-H release model with fixed public architecture settings."""

    def __init__(
        self,
        fnet_pretrained=False,
        valid_iters=4,
        corr_levels=2,
        corr_radius=4,
        cv_group=8,
        max_disp=192,
    ):
        super().__init__()
        tail_config = H_TAIL_CONFIG.copy()

        self.model_size = "h"
        self.base_model_size = "m"
        self.h_tail = "lite"
        self.tail_config = tail_config
        self.valid_iters = valid_iters
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.hidden_dim = tail_config["hidden_dim"]
        self.cv_group = cv_group
        self.max_disp = max_disp
        self.mask_dim = tail_config["mask_dim"]

        self.fnet_name = FASTERNET_T0_MODEL
        self.fnet = FeatureNetFasterNet(pretrained=fnet_pretrained)
        self.fnet_channels = self.fnet.feature_channels
        if self.fnet_channels[0] % self.cv_group != 0:
            raise ValueError(
                f"First feature level has {self.fnet_channels[0]} channels, "
                f"which is not divisible by cv_group={self.cv_group}"
            )

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
        dx = torch.arange(-self.corr_radius, self.corr_radius + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        self.register_buffer("dx", dx, persistent=False)

        self.cost_agg = FasterNetAggregation(
            in_channels=self.max_disp // 4,
            left_att=True,
            blocks=list(H_BASE_MODEL_CONFIG["blocks"]),
            expanse_ratio=H_BASE_MODEL_CONFIG["expanse_ratio"],
            backbone_channels=self.fnet_channels,
        )
        self.aggregation_name = "fasternet"

        stem_dim = tail_config["stem_dim"]
        spx_dim = tail_config["spx_dim"]
        spx_out_dim = tail_config["spx_out_dim"]
        self.stem_2 = nn.Sequential(
            BasicConv2d(
                3,
                stem_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_layer=nn.InstanceNorm2d,
                act_layer=nn.LeakyReLU,
            ),
            BasicConv2d(
                stem_dim,
                stem_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.InstanceNorm2d,
                act_layer=nn.ReLU,
            ),
        )
        self.spx_2_gru = Conv2x(
            self.mask_dim,
            spx_dim,
            deconv=True,
            concat=True,
            bn=False,
            rem_channels=stem_dim,
            fused_channels=spx_out_dim,
        )
        self.spx_gru = nn.ConvTranspose2d(spx_out_dim, 9, kernel_size=4, stride=2, padding=1)

        self.cnet = ContextNetSharedBackbone(self.fnet_channels[0], self.hidden_dim)
        self.cam = ChannelAttentionEnhancement(self.hidden_dim)
        self.sam = SpatialAttentionExtractor()
        self.update_block = BasicSelectiveUpdateBlock(
            corr_levels=self.corr_levels,
            corr_radius=self.corr_radius,
            volume_dim=self.cv_group,
            hidden_dim=self.hidden_dim,
            motion_corr_dim=tail_config["motion_corr_dim"],
            motion_disp_dim=tail_config["motion_disp_dim"],
            mask_inter_dim=tail_config["mask_inter_dim"],
            mask_dim=self.mask_dim,
        )

    def normalize_image(self, img):
        return ((img / 255.0 - self.image_mean) / self.image_std).contiguous()

    def build_upsample_mask(self, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        xspx = self.spx_gru(xspx)
        return F.softmax(xspx, 1)

    def upsample_disp_with_mask(self, disp, spx_pred):
        return context_upsample(disp * 4.0, spx_pred.float())

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        return self.upsample_disp_with_mask(disp, self.build_upsample_mask(mask_feat_4, stem_2x))

    def _prepare_context(self, features_left):
        cnet_list = list(self.cnet(features_left[0], features_left[1], features_left[2]))
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [self.cam(x) * x for x in inp_list]
        att = [self.sam(x) for x in inp_list]
        return net_list, inp_list, att

    def forward(self, left, right, max_disp=None, iters=None, test_mode=False, kd_mode=False, jetson_mode=False):
        del jetson_mode
        max_disp = self.max_disp if max_disp is None else max_disp
        iters = self.valid_iters if iters is None else iters
        if max_disp // 4 != self.max_disp // 4:
            raise ValueError(
                f"LiteAnyStereoH was built for max_disp={self.max_disp}; got max_disp={max_disp}. "
                "The Fasternet cost aggregation channel count is fixed at construction time."
            )

        left = self.normalize_image(left)
        right = self.normalize_image(right)

        stem_2x = self.stem_2(left)
        features_left = self.fnet(left)
        features_right = self.fnet(right)

        gwc_volume = build_gwc_volume_fast(features_left[0], features_right[0], max_disp // 4, self.cv_group)
        cost_volume = gwc_volume.mean(dim=1)
        cv = self.cost_agg(cost_volume, features_left)

        prob = F.softmax(cv, dim=1)
        init_disp = disparity_regression(prob, max_disp // 4)
        disp = init_disp

        net_list, inp_list, att = self._prepare_context(features_left)
        geo_fn = CombinedGeoEncodingVolume(features_left[0], features_right[0], gwc_volume, num_levels=self.corr_levels)

        batch, _, height, width = features_left[0].shape
        coords = torch.arange(width, dtype=disp.dtype, device=disp.device).reshape(1, 1, width, 1)
        coords = coords.repeat(batch, height, 1, 1)

        disp_predictions = []
        disp_up = None

        for step in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, self.dx)
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
            disp = disp + delta_disp
            if not test_mode or step == iters - 1:
                disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
                if not test_mode:
                    disp_predictions.append(disp_up)

        if test_mode:
            return disp_up

        init_disp_up = F.interpolate(init_disp, left.shape[2:], mode="bilinear", align_corners=False) * 4.0
        if not disp_predictions:
            zero_mask_feat = disp.new_zeros(batch, self.mask_dim, height, width)
            disp_predictions.append(self.upsample_disp(disp, zero_mask_feat, stem_2x))

        disp_preds = list(reversed(disp_predictions))
        disp_preds.append(init_disp_up)
        if kd_mode:
            return disp_preds, features_left, features_right
        return disp_preds
