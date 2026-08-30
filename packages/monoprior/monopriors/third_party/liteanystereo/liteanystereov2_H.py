"""Owned LiteAnyStereo V2 H-variant inference model."""

from typing import TypeAlias, TypedDict

import torch
import torch.nn.functional as F
from jaxtyping import Float32
from torch import Tensor, nn

from monopriors.third_party.liteanystereo.aggregation_fasternet import Aggregation as FasterNetAggregation
from monopriors.third_party.liteanystereo.aggregation_fasternet import FasterNetBlock
from monopriors.third_party.liteanystereo.fnet import FASTERNET_T0_MODEL, FeatureNetFasterNet, FeaturePyramid
from monopriors.third_party.liteanystereo.submodule import (
    BasicConv2d,
    BasicDeconv2d,
    build_gwc_volume_fast,
    context_upsample,
    disparity_regression,
)

TensorPyramid: TypeAlias = list[Float32[Tensor, "b _channels h w"]]
ContextNetOutput: TypeAlias = tuple[list[Float32[Tensor, "b hidden h w"]]]
PreparedContext: TypeAlias = tuple[TensorPyramid, TensorPyramid, TensorPyramid]
UpdateBlockOutput: TypeAlias = tuple[
    TensorPyramid,
    Float32[Tensor, "b mask_channels h w"],
    Float32[Tensor, "b 1 h w"],
]


class HBaseModelConfig(TypedDict):
    """FasterNet aggregation settings for LAS2-H."""

    blocks: list[int]
    expanse_ratio: int


class HTailConfig(TypedDict):
    """Iterative update-tail settings for LAS2-H."""

    hidden_dim: int
    motion_corr_dim: int
    motion_disp_dim: int
    mask_inter_dim: int
    mask_dim: int
    stem_dim: int
    spx_dim: int
    spx_out_dim: int


H_BASE_MODEL_CONFIG: HBaseModelConfig = {"blocks": [4, 8, 16], "expanse_ratio": 4}
H_TAIL_CONFIG: HTailConfig = {
    "hidden_dim": 64,
    "motion_corr_dim": 128,
    "motion_disp_dim": 32,
    "mask_inter_dim": 32,
    "mask_dim": 16,
    "stem_dim": 16,
    "spx_dim": 16,
    "spx_out_dim": 32,
}


def _sample_1d(
    image_nc1w: Float32[Tensor, "samples channels 1 width"],
    x_coordinates_n1k1: Float32[Tensor, "samples 1 points 1"],
) -> Float32[Tensor, "samples channels 1 points"]:
    """Sample a float32 1D feature row at pixel-space x coordinates.

    Args:
        image_nc1w: Float32 features with shape ``(samples, channels, 1, width)``.
        x_coordinates_n1k1: Float32 x coordinates with shape ``(samples, 1, points, 1)``.

    Returns:
        Float32 samples with shape ``(samples, channels, 1, points)``.
    """
    height: int = image_nc1w.shape[2]
    width: int = image_nc1w.shape[3]
    assert height == 1
    x_grid_n1k1: Float32[Tensor, "samples 1 points 1"] = 2 * x_coordinates_n1k1 / max(width - 1, 1) - 1
    y_grid_n1k1: Float32[Tensor, "samples 1 points 1"] = torch.zeros_like(x_grid_n1k1)
    grid_n1k2: Float32[Tensor, "samples 1 points 2"] = torch.cat([x_grid_n1k1, y_grid_n1k1], dim=-1)
    sampled_nc1k: Float32[Tensor, "samples channels 1 points"] = F.grid_sample(
        image_nc1w,
        grid_n1k2,
        mode="bilinear",
        align_corners=True,
    )
    return sampled_nc1k


class CombinedGeoEncodingVolume:
    """Build and sample geometric and all-pairs correlation pyramids."""

    def __init__(
        self,
        fmap1_bchw: Float32[Tensor, "b channels h w"],
        fmap2_bchw: Float32[Tensor, "b channels h w2"],
        geo_volume_bgdhw: Float32[Tensor, "b groups disparities h w"],
        num_levels: int = 2,
    ) -> None:
        """Initialize the correlation pyramids.

        Args:
            fmap1_bchw: Float32 left features with shape ``(batch, channels, height, width)``.
            fmap2_bchw: Float32 right features with shape ``(batch, channels, height, right_width)``.
            geo_volume_bgdhw: Float32 grouped correlation with shape ``(batch, groups, disparities, height, width)``.
            num_levels: Number of horizontal average-pooling levels.
        """
        self.num_levels: int = num_levels
        self.geo_volume_pyramid: list[Float32[Tensor, "samples groups 1 _disparities"]] = []
        self.init_corr_pyramid: list[Float32[Tensor, "samples 1 1 _right_width"]] = []

        init_corr_bhw1v: Float32[Tensor, "b h w 1 w2"] = self.corr(fmap1_bchw, fmap2_bchw)
        corr_shape: torch.Size = init_corr_bhw1v.shape
        batch_size: int = corr_shape[0]
        height: int = corr_shape[1]
        width: int = corr_shape[2]
        right_width: int = corr_shape[4]

        geo_shape: torch.Size = geo_volume_bgdhw.shape
        channels: int = geo_shape[1]
        disparities: int = geo_shape[2]
        geo_volume_ng1d: Float32[Tensor, "samples groups 1 disparities"] = geo_volume_bgdhw.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, channels, 1, disparities
        )
        init_corr_n11v: Float32[Tensor, "samples 1 1 right_width"] = init_corr_bhw1v.view(batch_size * height * width, 1, 1, right_width)

        self.geo_volume_pyramid.append(geo_volume_ng1d)
        self.init_corr_pyramid.append(init_corr_n11v)
        for _ in range(self.num_levels - 1):
            geo_volume_ng1d = F.avg_pool2d(geo_volume_ng1d, [1, 2], stride=[1, 2])
            init_corr_n11v = F.avg_pool2d(init_corr_n11v, [1, 2], stride=[1, 2])
            self.geo_volume_pyramid.append(geo_volume_ng1d)
            self.init_corr_pyramid.append(init_corr_n11v)

    def __call__(
        self,
        disparity_b1hw: Float32[Tensor, "b 1 h w"],
        coordinates_bhw1: Float32[Tensor, "b h w 1"],
        dx_11r1: Float32[Tensor, "1 1 radius_samples 1"],
    ) -> Float32[Tensor, "b correlation_features h w"]:
        """Sample both pyramids around the current disparity estimate.

        Args:
            disparity_b1hw: Float32 disparity with shape ``(batch, 1, height, width)``.
            coordinates_bhw1: Float32 x coordinates with shape ``(batch, height, width, 1)``.
            dx_11r1: Float32 local offsets with shape ``(1, 1, radius_samples, 1)``.

        Returns:
            Float32 encoded correlation with shape ``(batch, correlation_features, height, width)``.
        """
        shape_b1hw: torch.Size = disparity_b1hw.shape
        batch_size: int = shape_b1hw[0]
        height: int = shape_b1hw[2]
        width: int = shape_b1hw[3]
        outputs_bhwf: list[Float32[Tensor, "b h w _features"]] = []
        for level in range(self.num_levels):
            scale: int = 2**level
            disparity_flat_n111: Float32[Tensor, "samples 1 1 1"] = disparity_b1hw.view(batch_size * height * width, 1, 1, 1) / scale
            dx_level_11r1: Float32[Tensor, "1 1 radius_samples 1"] = dx_11r1.to(device=disparity_b1hw.device, dtype=disparity_b1hw.dtype)

            geo_x_n1r1: Float32[Tensor, "samples 1 radius_samples 1"] = dx_level_11r1 + disparity_flat_n111
            geo_sample_ng1r: Float32[Tensor, "samples groups 1 radius_samples"] = _sample_1d(self.geo_volume_pyramid[level], geo_x_n1r1)
            geo_bhwf: Float32[Tensor, "b h w geo_features"] = geo_sample_ng1r.view(batch_size, height, width, -1)

            corr_x_n1r1: Float32[Tensor, "samples 1 radius_samples 1"] = (
                coordinates_bhw1.view(batch_size * height * width, 1, 1, 1) / scale - disparity_flat_n111 + dx_level_11r1
            )
            corr_sample_n11r: Float32[Tensor, "samples 1 1 radius_samples"] = _sample_1d(self.init_corr_pyramid[level], corr_x_n1r1)
            corr_bhwf: Float32[Tensor, "b h w corr_features"] = corr_sample_n11r.view(batch_size, height, width, -1)

            outputs_bhwf.append(geo_bhwf)
            outputs_bhwf.append(corr_bhwf)

        combined_bhwf: Float32[Tensor, "b h w correlation_features"] = torch.cat(outputs_bhwf, dim=-1)
        combined_bfhw: Float32[Tensor, "b correlation_features h w"] = combined_bhwf.permute(0, 3, 1, 2).contiguous()
        return combined_bfhw

    @staticmethod
    def corr(
        fmap1_bchw: Float32[Tensor, "b channels h w"],
        fmap2_bchw: Float32[Tensor, "b channels h w2"],
    ) -> Float32[Tensor, "b h w 1 w2"]:
        """Compute float32 all-pairs horizontal correlation.

        Args:
            fmap1_bchw: Float32 left features with shape ``(batch, channels, height, width)``.
            fmap2_bchw: Float32 right features with shape ``(batch, channels, height, right_width)``.

        Returns:
            Float32 correlation with shape ``(batch, height, width, 1, right_width)``.
        """
        shape_bchw: torch.Size = fmap1_bchw.shape
        batch_size: int = shape_bchw[0]
        channels: int = shape_bchw[1]
        height: int = shape_bchw[2]
        width: int = shape_bchw[3]
        right_width: int = fmap2_bchw.shape[3]
        correlation_bhwv: Float32[Tensor, "b h w w2"] = torch.einsum("bchw,bchv->bhwv", fmap1_bchw, fmap2_bchw) / channels
        correlation_bhw1v: Float32[Tensor, "b h w 1 w2"] = correlation_bhwv.view(batch_size, height, width, 1, right_width).to(fmap1_bchw.dtype)
        return correlation_bhw1v


class ContextNetSharedBackbone(nn.Module):
    """Create the hidden and input states for the LAS2-H update block."""

    def __init__(self, c04: int, hidden_dim: int) -> None:
        """Initialize the context projections.

        Args:
            c04: Input feature channel count at quarter resolution.
            hidden_dim: Hidden-state channel count.
        """
        super().__init__()
        self.hidden_conv: nn.Conv2d = nn.Conv2d(c04, hidden_dim, kernel_size=3, padding=1)
        self.input_conv: nn.Conv2d = nn.Conv2d(c04, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x4_bchw: Float32[Tensor, "b channels h w"]) -> ContextNetOutput:
        """Project quarter-resolution float32 image features.

        Args:
            x4_bchw: Float32 image features with shape ``(batch, channels, height, width)``.

        Returns:
            One hidden/input pair, each float32 with shape ``(batch, hidden, height, width)``.
        """
        hidden_bchw: Float32[Tensor, "b hidden h w"] = self.hidden_conv(x4_bchw)
        input_bchw: Float32[Tensor, "b hidden h w"] = self.input_conv(x4_bchw)
        return ([hidden_bchw, input_bchw],)


class ChannelAttentionEnhancement(nn.Module):
    """Predict per-channel update-selection weights."""

    def __init__(self, channels: int, ratio: int = 16) -> None:
        """Initialize channel attention.

        Args:
            channels: Input and output channel count.
            ratio: Hidden-channel reduction ratio.
        """
        super().__init__()
        hidden: int = max(channels // ratio, 1)
        self.avg_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.max_pool: nn.AdaptiveMaxPool2d = nn.AdaptiveMaxPool2d(1)
        self.fc: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x_bchw: Float32[Tensor, "b channels h w"]) -> Float32[Tensor, "b channels 1 1"]:
        """Predict float32 channel weights.

        Args:
            x_bchw: Float32 tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float32 weights with shape ``(batch, channels, 1, 1)``.
        """
        average_bchw: Float32[Tensor, "b channels 1 1"] = self.fc(self.avg_pool(x_bchw))
        maximum_bchw: Float32[Tensor, "b channels 1 1"] = self.fc(self.max_pool(x_bchw))
        attention_bchw: Float32[Tensor, "b channels 1 1"] = self.sigmoid(average_bchw + maximum_bchw)
        return attention_bchw


class SpatialAttentionExtractor(nn.Module):
    """Predict per-pixel update-selection weights."""

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize spatial attention.

        Args:
            kernel_size: Spatial convolution kernel size.
        """
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x_bchw: Float32[Tensor, "b channels h w"]) -> Float32[Tensor, "b 1 h w"]:
        """Predict float32 spatial weights.

        Args:
            x_bchw: Float32 tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float32 weights with shape ``(batch, 1, height, width)``.
        """
        average_b1hw: Float32[Tensor, "b 1 h w"] = torch.mean(x_bchw, dim=1, keepdim=True)
        maximum_b1hw: Float32[Tensor, "b 1 h w"] = torch.max(x_bchw, dim=1, keepdim=True)[0]
        pooled_b2hw: Float32[Tensor, "b 2 h w"] = torch.cat([average_b1hw, maximum_b1hw], dim=1)
        attention_b1hw: Float32[Tensor, "b 1 h w"] = self.sigmoid(self.conv(pooled_b2hw))
        return attention_b1hw


class FasterNetConvEncoder(nn.Module):
    """Apply one FasterNet block inside the disparity head."""

    def __init__(self, dim: int, mlp_ratio: int = 4, n_div: int = 4) -> None:
        """Initialize the encoder.

        Args:
            dim: Input and output channel count.
            mlp_ratio: MLP expansion ratio.
            n_div: Partial-convolution channel divisor.
        """
        super().__init__()
        self.block: FasterNetBlock = FasterNetBlock(dim, mlp_ratio=mlp_ratio, n_div=n_div, act_layer=nn.GELU)

    def forward(self, x_bchw: Float32[Tensor, "b channels h w"]) -> Float32[Tensor, "b channels h w"]:
        """Transform a float32 feature map.

        Args:
            x_bchw: Float32 tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float32 tensor with shape ``(batch, channels, height, width)``.
        """
        output_bchw: Float32[Tensor, "b channels h w"] = self.block(x_bchw)
        return output_bchw


class DispHead(nn.Module):
    """Predict a disparity update from the recurrent hidden state."""

    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        """Initialize the disparity head.

        Args:
            input_dim: Hidden-state channel count.
            output_dim: Output disparity channel count.
        """
        super().__init__()
        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            FasterNetConvEncoder(input_dim, mlp_ratio=4),
            FasterNetConvEncoder(input_dim, mlp_ratio=4),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x_bchw: Float32[Tensor, "b channels h w"]) -> Float32[Tensor, "b output_channels h w"]:
        """Predict a float32 disparity update.

        Args:
            x_bchw: Float32 hidden state with shape ``(batch, channels, height, width)``.

        Returns:
            Float32 update with shape ``(batch, output_channels, height, width)``.
        """
        output_bchw: Float32[Tensor, "b output_channels h w"] = self.conv(x_bchw)
        return output_bchw


class Conv2x(nn.Module):
    """Resize a feature map by two, combine a skip, and convolve the result."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        concat: bool = True,
        bn: bool = True,
        relu: bool = True,
        rem_channels: int | None = None,
        fused_channels: int | None = None,
    ) -> None:
        """Initialize the resize-and-fuse block.

        Args:
            in_channels: Main input channel count.
            out_channels: Resized feature channel count.
            deconv: Whether to upsample instead of downsample.
            concat: Whether to concatenate rather than add the skip.
            bn: Whether to apply batch normalization.
            relu: Whether to apply LeakyReLU.
            rem_channels: Skip channel count; defaults to ``out_channels``.
            fused_channels: Final channel count; derived from ``concat`` when None.
        """
        super().__init__()
        norm_layer: type[nn.BatchNorm2d] | None = nn.BatchNorm2d if bn else None
        act_layer: type[nn.LeakyReLU] | None = nn.LeakyReLU if relu else None
        if deconv:
            self.conv1: nn.Module = BasicDeconv2d(
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

        self.concat: bool = concat
        resolved_rem_channels: int = out_channels if rem_channels is None else rem_channels
        conv2_in_channels: int = out_channels + resolved_rem_channels if concat else out_channels
        conv2_out_channels: int = fused_channels if fused_channels is not None else (out_channels * 2 if concat else out_channels)
        self.conv2: BasicConv2d = BasicConv2d(
            conv2_in_channels,
            conv2_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(
        self,
        x_bchw: Float32[Tensor, "b channels h w"],
        rem_bchw: Float32[Tensor, "b rem_channels rem_h rem_w"],
    ) -> Float32[Tensor, "b output_channels rem_h rem_w"]:
        """Resize and fuse two float32 feature maps.

        Args:
            x_bchw: Float32 main tensor with shape ``(batch, channels, height, width)``.
            rem_bchw: Float32 skip tensor with shape ``(batch, rem_channels, rem_height, rem_width)``.

        Returns:
            Float32 fused tensor with shape ``(batch, output_channels, rem_height, rem_width)``.
        """
        resized_bchw: Float32[Tensor, "b resized_channels resized_h resized_w"] = self.conv1(x_bchw)
        if resized_bchw.shape[-2:] != rem_bchw.shape[-2:]:
            resized_bchw = F.interpolate(resized_bchw, size=rem_bchw.shape[-2:], mode="bilinear", align_corners=False)
        if self.concat:
            combined_bchw: Float32[Tensor, "b combined_channels rem_h rem_w"] = torch.cat((resized_bchw, rem_bchw), dim=1)
        else:
            combined_bchw = resized_bchw + rem_bchw
        output_bchw: Float32[Tensor, "b output_channels rem_h rem_w"] = self.conv2(combined_bchw)
        return output_bchw


class BasicMotionEncoder(nn.Module):
    """Encode correlation and current disparity for the recurrent update."""

    def __init__(
        self,
        corr_levels: int,
        corr_radius: int,
        volume_dim: int,
        hidden_dim: int,
        motion_corr_dim: int,
        motion_disp_dim: int,
    ) -> None:
        """Initialize the motion encoder.

        Args:
            corr_levels: Number of correlation-pyramid levels.
            corr_radius: Radius sampled at each level.
            volume_dim: Groupwise-correlation channel count.
            hidden_dim: Output motion-feature channel count.
            motion_corr_dim: Correlation-branch channel count.
            motion_disp_dim: Disparity-branch channel count.
        """
        super().__init__()
        corr_planes: int = corr_levels * (2 * corr_radius + 1) * (volume_dim + 1)
        self.convc1: nn.Conv2d = nn.Conv2d(corr_planes, motion_corr_dim, kernel_size=1)
        self.convc2: nn.Conv2d = nn.Conv2d(motion_corr_dim, motion_corr_dim, kernel_size=3, padding=1)
        self.convd1: nn.Conv2d = nn.Conv2d(1, motion_disp_dim, kernel_size=7, padding=3)
        self.convd2: nn.Conv2d = nn.Conv2d(motion_disp_dim, motion_disp_dim, kernel_size=3, padding=1)
        self.conv: nn.Conv2d = nn.Conv2d(motion_disp_dim + motion_corr_dim, hidden_dim - 1, kernel_size=1)

    def forward(
        self,
        disparity_b1hw: Float32[Tensor, "b 1 h w"],
        correlation_bchw: Float32[Tensor, "b correlation_channels h w"],
    ) -> Float32[Tensor, "b hidden h w"]:
        """Encode float32 disparity and correlation features.

        Args:
            disparity_b1hw: Float32 disparity with shape ``(batch, 1, height, width)``.
            correlation_bchw: Float32 correlation with shape ``(batch, correlation_channels, height, width)``.

        Returns:
            Float32 motion features with shape ``(batch, hidden, height, width)``.
        """
        correlation1_bchw: Float32[Tensor, "b motion_corr_channels h w"] = F.relu(self.convc1(correlation_bchw), inplace=True)
        correlation2_bchw: Float32[Tensor, "b motion_corr_channels h w"] = F.relu(self.convc2(correlation1_bchw), inplace=True)
        disparity1_bchw: Float32[Tensor, "b motion_disp_channels h w"] = F.relu(self.convd1(disparity_b1hw), inplace=True)
        disparity2_bchw: Float32[Tensor, "b motion_disp_channels h w"] = F.relu(self.convd2(disparity1_bchw), inplace=True)
        combined_bchw: Float32[Tensor, "b combined_channels h w"] = torch.cat([correlation2_bchw, disparity2_bchw], dim=1)
        encoded_bchw: Float32[Tensor, "b hidden_minus_one h w"] = F.relu(self.conv(combined_bchw), inplace=True)
        motion_bchw: Float32[Tensor, "b hidden h w"] = torch.cat([encoded_bchw, disparity_b1hw], dim=1)
        return motion_bchw


class RaftConvGRU(nn.Module):
    """RAFT-style convolutional gated recurrent unit."""

    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int) -> None:
        """Initialize the GRU.

        Args:
            hidden_dim: Hidden-state channel count.
            input_dim: Input feature channel count.
            kernel_size: Gate convolution kernel size.
        """
        super().__init__()
        padding: int = kernel_size // 2
        self.convz: nn.Conv2d = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convr: nn.Conv2d = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convq: nn.Conv2d = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)

    def forward(
        self,
        hidden_bchw: Float32[Tensor, "b hidden h w"],
        input_bchw: Float32[Tensor, "b input_channels h w"],
        hidden_input_bchw: Float32[Tensor, "b combined_channels h w"],
    ) -> Float32[Tensor, "b hidden h w"]:
        """Update a float32 recurrent hidden state.

        Args:
            hidden_bchw: Float32 hidden state with shape ``(batch, hidden, height, width)``.
            input_bchw: Float32 input with shape ``(batch, input_channels, height, width)``.
            hidden_input_bchw: Float32 preprocessed concatenation with shape ``(batch, combined_channels, height, width)``.

        Returns:
            Float32 updated hidden state with shape ``(batch, hidden, height, width)``.
        """
        update_gate_bchw: Float32[Tensor, "b hidden h w"] = torch.sigmoid(self.convz(hidden_input_bchw))
        reset_gate_bchw: Float32[Tensor, "b hidden h w"] = torch.sigmoid(self.convr(hidden_input_bchw))
        candidate_input_bchw: Float32[Tensor, "b combined_channels h w"] = torch.cat([reset_gate_bchw * hidden_bchw, input_bchw], dim=1)
        candidate_bchw: Float32[Tensor, "b hidden h w"] = torch.tanh(self.convq(candidate_input_bchw))
        output_bchw: Float32[Tensor, "b hidden h w"] = (1 - update_gate_bchw) * hidden_bchw + update_gate_bchw * candidate_bchw
        return output_bchw


class SelectiveConvGRU(nn.Module):
    """Blend small- and large-kernel GRU updates with learned attention."""

    def __init__(self, hidden_dim: int, input_dim: int, small_kernel_size: int = 1, large_kernel_size: int = 3) -> None:
        """Initialize the selective GRU.

        Args:
            hidden_dim: Hidden-state channel count.
            input_dim: Input feature channel count.
            small_kernel_size: Small GRU kernel size.
            large_kernel_size: Large GRU kernel size.
        """
        super().__init__()
        self.conv0: nn.Sequential = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv1: nn.Sequential = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, input_dim + hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.small_gru: RaftConvGRU = RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru: RaftConvGRU = RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(
        self,
        attention_b1hw: Float32[Tensor, "b 1 h w"],
        hidden_bchw: Float32[Tensor, "b hidden h w"],
        input_bchw: Float32[Tensor, "b input_channels h w"],
    ) -> Float32[Tensor, "b hidden h w"]:
        """Apply a selectively blended recurrent update.

        Args:
            attention_b1hw: Float32 spatial weights with shape ``(batch, 1, height, width)``.
            hidden_bchw: Float32 hidden state with shape ``(batch, hidden, height, width)``.
            input_bchw: Float32 input with shape ``(batch, input_channels, height, width)``.

        Returns:
            Float32 updated hidden state with shape ``(batch, hidden, height, width)``.
        """
        encoded_input_bchw: Float32[Tensor, "b input_channels h w"] = self.conv0(input_bchw)
        hidden_input_bchw: Float32[Tensor, "b combined_channels h w"] = self.conv1(torch.cat([encoded_input_bchw, hidden_bchw], dim=1))
        small_update_bchw: Float32[Tensor, "b hidden h w"] = self.small_gru(hidden_bchw, encoded_input_bchw, hidden_input_bchw)
        large_update_bchw: Float32[Tensor, "b hidden h w"] = self.large_gru(hidden_bchw, encoded_input_bchw, hidden_input_bchw)
        output_bchw: Float32[Tensor, "b hidden h w"] = small_update_bchw * attention_b1hw + large_update_bchw * (1 - attention_b1hw)
        return output_bchw


class BasicSelectiveUpdateBlock(nn.Module):
    """Run one recurrent disparity and upsampling-mask update."""

    def __init__(
        self,
        corr_levels: int,
        corr_radius: int,
        volume_dim: int,
        hidden_dim: int,
        motion_corr_dim: int,
        motion_disp_dim: int,
        mask_inter_dim: int,
        mask_dim: int,
    ) -> None:
        """Initialize the update block.

        Args:
            corr_levels: Number of correlation-pyramid levels.
            corr_radius: Radius sampled at each level.
            volume_dim: Groupwise-correlation channel count.
            hidden_dim: Recurrent hidden-state channel count.
            motion_corr_dim: Correlation-branch channel count.
            motion_disp_dim: Disparity-branch channel count.
            mask_inter_dim: Intermediate mask-feature channel count.
            mask_dim: Output mask-feature channel count.
        """
        super().__init__()
        self.encoder: BasicMotionEncoder = BasicMotionEncoder(
            corr_levels,
            corr_radius,
            volume_dim,
            hidden_dim,
            motion_corr_dim,
            motion_disp_dim,
        )
        self.gru04: SelectiveConvGRU = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        self.disp_head: DispHead = DispHead(hidden_dim)
        self.mask: nn.Sequential = nn.Sequential(
            nn.Conv2d(hidden_dim, mask_inter_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_inter_dim, mask_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        net: TensorPyramid,
        inp: TensorPyramid,
        correlation_bchw: Float32[Tensor, "b correlation_channels h w"],
        disparity_b1hw: Float32[Tensor, "b 1 h w"],
        attention: TensorPyramid,
    ) -> UpdateBlockOutput:
        """Update the float32 hidden state and disparity.

        Args:
            net: Float32 recurrent states, each with shape ``(batch, channels, height, width)``.
            inp: Float32 context inputs, each with shape ``(batch, channels, height, width)``.
            correlation_bchw: Float32 correlation with shape ``(batch, correlation_channels, height, width)``.
            disparity_b1hw: Float32 disparity with shape ``(batch, 1, height, width)``.
            attention: Float32 spatial weights, each with shape ``(batch, channels, height, width)``.

        Returns:
            Updated states, float32 mask features, and float32 disparity update.
        """
        motion_features_bchw: Float32[Tensor, "b hidden h w"] = self.encoder(disparity_b1hw, correlation_bchw)
        recurrent_input_bchw: Float32[Tensor, "b double_hidden h w"] = torch.cat([inp[0], motion_features_bchw], dim=1)
        net[0] = self.gru04(attention[0], net[0], recurrent_input_bchw)
        delta_disparity_b1hw: Float32[Tensor, "b 1 h w"] = self.disp_head(net[0])
        mask_features_bchw: Float32[Tensor, "b mask_channels h w"] = 0.25 * self.mask(net[0])
        return net, mask_features_bchw, delta_disparity_b1hw


class LiteAnyStereoH(nn.Module):
    """LAS2-H inference model with the released iterative ConvGRU architecture."""

    def __init__(
        self,
        fnet_pretrained: bool = False,
        valid_iters: int = 4,
        corr_levels: int = 2,
        corr_radius: int = 4,
        cv_group: int = 8,
        max_disp: int = 192,
    ) -> None:
        """Initialize LAS2-H.

        Args:
            fnet_pretrained: Whether timm should load pretrained FasterNet weights.
            valid_iters: Default number of recurrent inference updates.
            corr_levels: Number of correlation-pyramid levels.
            corr_radius: Radius sampled at each correlation level.
            cv_group: Number of groupwise-correlation channel groups.
            max_disp: Full-resolution disparity range fixed at construction.
        """
        super().__init__()
        tail_config: HTailConfig = H_TAIL_CONFIG.copy()

        self.model_size: str = "h"
        self.base_model_size: str = "m"
        self.h_tail: str = "lite"
        self.tail_config: HTailConfig = tail_config
        self.valid_iters: int = valid_iters
        self.corr_levels: int = corr_levels
        self.corr_radius: int = corr_radius
        self.hidden_dim: int = tail_config["hidden_dim"]
        self.cv_group: int = cv_group
        self.max_disp: int = max_disp
        self.mask_dim: int = tail_config["mask_dim"]

        self.fnet_name: str = FASTERNET_T0_MODEL
        self.fnet: FeatureNetFasterNet = FeatureNetFasterNet(pretrained=fnet_pretrained)
        self.fnet_channels: list[int] = self.fnet.feature_channels
        if self.fnet_channels[0] % self.cv_group != 0:
            raise ValueError(f"First feature level has {self.fnet_channels[0]} channels, which is not divisible by cv_group={self.cv_group}")

        image_mean_1311: Float32[Tensor, "1 3 1 1"] = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std_1311: Float32[Tensor, "1 3 1 1"] = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        dx_11r1: Float32[Tensor, "1 1 radius_samples 1"] = torch.arange(
            -self.corr_radius,
            self.corr_radius + 1,
            dtype=torch.float32,
        ).reshape(1, 1, -1, 1)
        self.image_mean: Float32[Tensor, "1 3 1 1"]
        self.image_std: Float32[Tensor, "1 3 1 1"]
        self.dx: Float32[Tensor, "1 1 radius_samples 1"]
        self.register_buffer("image_mean", image_mean_1311, persistent=False)
        self.register_buffer("image_std", image_std_1311, persistent=False)
        self.register_buffer("dx", dx_11r1, persistent=False)

        self.cost_agg: FasterNetAggregation = FasterNetAggregation(
            in_channels=self.max_disp // 4,
            left_att=True,
            blocks=list(H_BASE_MODEL_CONFIG["blocks"]),
            expanse_ratio=H_BASE_MODEL_CONFIG["expanse_ratio"],
            backbone_channels=self.fnet_channels,
        )
        self.aggregation_name: str = "fasternet"

        stem_dim: int = tail_config["stem_dim"]
        spx_dim: int = tail_config["spx_dim"]
        spx_out_dim: int = tail_config["spx_out_dim"]
        self.stem_2: nn.Sequential = nn.Sequential(
            BasicConv2d(3, stem_dim, kernel_size=3, stride=2, padding=1, norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(stem_dim, stem_dim, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU),
        )
        self.spx_2_gru: Conv2x = Conv2x(
            self.mask_dim,
            spx_dim,
            deconv=True,
            concat=True,
            bn=False,
            rem_channels=stem_dim,
            fused_channels=spx_out_dim,
        )
        self.spx_gru: nn.ConvTranspose2d = nn.ConvTranspose2d(spx_out_dim, 9, kernel_size=4, stride=2, padding=1)

        self.cnet: ContextNetSharedBackbone = ContextNetSharedBackbone(self.fnet_channels[0], self.hidden_dim)
        self.cam: ChannelAttentionEnhancement = ChannelAttentionEnhancement(self.hidden_dim)
        self.sam: SpatialAttentionExtractor = SpatialAttentionExtractor()
        self.update_block: BasicSelectiveUpdateBlock = BasicSelectiveUpdateBlock(
            corr_levels=self.corr_levels,
            corr_radius=self.corr_radius,
            volume_dim=self.cv_group,
            hidden_dim=self.hidden_dim,
            motion_corr_dim=tail_config["motion_corr_dim"],
            motion_disp_dim=tail_config["motion_disp_dim"],
            mask_inter_dim=tail_config["mask_inter_dim"],
            mask_dim=self.mask_dim,
        )

    def normalize_image(self, image_b3hw: Float32[Tensor, "b 3 h w"]) -> Float32[Tensor, "b 3 h w"]:
        """Normalize float32 RGB images in the 0-255 range with ImageNet statistics.

        Args:
            image_b3hw: Float32 RGB tensor with shape ``(batch, 3, height, width)``.

        Returns:
            Float32 normalized RGB tensor with shape ``(batch, 3, height, width)``.
        """
        normalized_b3hw: Float32[Tensor, "b 3 h w"] = ((image_b3hw / 255.0 - self.image_mean) / self.image_std).contiguous()
        return normalized_b3hw

    def build_upsample_mask(
        self,
        mask_features_bchw: Float32[Tensor, "b mask_channels h4 w4"],
        stem_2_bchw: Float32[Tensor, "b stem_channels h2 w2"],
    ) -> Float32[Tensor, "b 9 h w"]:
        """Build float32 convex-upsample weights.

        Args:
            mask_features_bchw: Float32 mask features with shape ``(batch, mask_channels, height / 4, width / 4)``.
            stem_2_bchw: Float32 stem features with shape ``(batch, stem_channels, height / 2, width / 2)``.

        Returns:
            Float32 weights with shape ``(batch, 9, height, width)``.
        """
        xspx_2_bchw: Float32[Tensor, "b spx_channels h2 w2"] = self.spx_2_gru(mask_features_bchw, stem_2_bchw)
        xspx_b9hw: Float32[Tensor, "b 9 h w"] = self.spx_gru(xspx_2_bchw)
        weights_b9hw: Float32[Tensor, "b 9 h w"] = F.softmax(xspx_b9hw, 1)
        return weights_b9hw

    def upsample_disp_with_mask(
        self,
        disparity_b1hw: Float32[Tensor, "b 1 h4 w4"],
        upsample_weights_b9hw: Float32[Tensor, "b 9 h w"],
    ) -> Float32[Tensor, "b 1 h w"]:
        """Upsample float32 disparity with precomputed weights.

        Args:
            disparity_b1hw: Float32 disparity with shape ``(batch, 1, height / 4, width / 4)``.
            upsample_weights_b9hw: Float32 weights with shape ``(batch, 9, height, width)``.

        Returns:
            Float32 disparity with shape ``(batch, 1, height, width)``.
        """
        disparity_up_b1hw: Float32[Tensor, "b 1 h w"] = context_upsample(disparity_b1hw * 4.0, upsample_weights_b9hw.float())
        return disparity_up_b1hw

    def upsample_disp(
        self,
        disparity_b1hw: Float32[Tensor, "b 1 h4 w4"],
        mask_features_bchw: Float32[Tensor, "b mask_channels h4 w4"],
        stem_2_bchw: Float32[Tensor, "b stem_channels h2 w2"],
    ) -> Float32[Tensor, "b 1 h w"]:
        """Build weights and upsample float32 disparity.

        Args:
            disparity_b1hw: Float32 disparity with shape ``(batch, 1, height / 4, width / 4)``.
            mask_features_bchw: Float32 mask features with shape ``(batch, mask_channels, height / 4, width / 4)``.
            stem_2_bchw: Float32 stem features with shape ``(batch, stem_channels, height / 2, width / 2)``.

        Returns:
            Float32 disparity with shape ``(batch, 1, height, width)``.
        """
        upsample_weights_b9hw: Float32[Tensor, "b 9 h w"] = self.build_upsample_mask(mask_features_bchw, stem_2_bchw)
        disparity_up_b1hw: Float32[Tensor, "b 1 h w"] = self.upsample_disp_with_mask(disparity_b1hw, upsample_weights_b9hw)
        return disparity_up_b1hw

    def _prepare_context(self, features_left: FeaturePyramid) -> PreparedContext:
        """Build recurrent states, inputs, and spatial attention.

        Args:
            features_left: Float32 image features; each item has shape ``(batch, channels, height, width)``.

        Returns:
            Three float32 tensor lists for recurrent state, recurrent input, and spatial attention.
        """
        context_output: ContextNetOutput = self.cnet(features_left[0])
        context_pairs: list[Float32[Tensor, "b hidden h w"]] = list(context_output[0])
        net_list: TensorPyramid = [torch.tanh(context_pairs[0])]
        input_list: TensorPyramid = [torch.relu(context_pairs[1])]
        enhanced_input_list: TensorPyramid = [self.cam(input_bchw) * input_bchw for input_bchw in input_list]
        attention_list: TensorPyramid = [self.sam(input_bchw) for input_bchw in enhanced_input_list]
        return net_list, enhanced_input_list, attention_list

    def forward(
        self,
        left_b3hw: Float32[Tensor, "b 3 h w"],
        right_b3hw: Float32[Tensor, "b 3 h w"],
        max_disp: int | None = None,
        iters: int | None = None,
        test_mode: bool = False,
    ) -> Float32[Tensor, "b 1 h w"]:
        """Predict left-view disparity for a float32 stereo pair.

        Args:
            left_b3hw: Float32 left RGB tensor with shape ``(batch, 3, height, width)`` and values in ``[0, 255]``.
            right_b3hw: Float32 right RGB tensor with shape ``(batch, 3, height, width)`` and values in ``[0, 255]``.
            max_disp: Full-resolution disparity range, or None for the construction-time value.
            iters: Number of recurrent updates, or None for ``valid_iters``.
            test_mode: Must be true because this owned fork retains inference outputs only.

        Returns:
            Float32 disparity tensor with shape ``(batch, 1, height, width)``.

        Raises:
            ValueError: If training output is requested, no iterations are requested, or the disparity architecture would change.
        """
        if not test_mode:
            raise ValueError("The owned LiteAnyStereo V2 fork supports inference only; pass test_mode=True.")
        resolved_max_disp: int = self.max_disp if max_disp is None else max_disp
        iteration_count: int = self.valid_iters if iters is None else iters
        if iteration_count < 1:
            raise ValueError("LiteAnyStereoH inference requires at least one iteration.")
        if resolved_max_disp // 4 != self.max_disp // 4:
            raise ValueError(
                f"LiteAnyStereoH was built for max_disp={self.max_disp}; got max_disp={resolved_max_disp}. "
                "The Fasternet cost aggregation channel count is fixed at construction time."
            )

        normalized_left_b3hw: Float32[Tensor, "b 3 h w"] = self.normalize_image(left_b3hw)
        normalized_right_b3hw: Float32[Tensor, "b 3 h w"] = self.normalize_image(right_b3hw)

        stem_2_bchw: Float32[Tensor, "b stem_channels h2 w2"] = self.stem_2(normalized_left_b3hw)
        features_left: FeaturePyramid = self.fnet(normalized_left_b3hw)
        features_right: FeaturePyramid = self.fnet(normalized_right_b3hw)

        gwc_volume_bgdhw: Float32[Tensor, "b groups disparities h4 w4"] = build_gwc_volume_fast(
            features_left[0],
            features_right[0],
            resolved_max_disp // 4,
            self.cv_group,
        )
        cost_volume_bdhw: Float32[Tensor, "b disparities h4 w4"] = gwc_volume_bgdhw.mean(dim=1)
        aggregated_cost_bdhw: Float32[Tensor, "b disparities h4 w4"] = self.cost_agg(cost_volume_bdhw, features_left)

        probability_bdhw: Float32[Tensor, "b disparities h4 w4"] = F.softmax(aggregated_cost_bdhw, dim=1)
        initial_disparity_b1hw: Float32[Tensor, "b 1 h4 w4"] = disparity_regression(probability_bdhw, resolved_max_disp // 4)
        disparity_b1hw: Float32[Tensor, "b 1 h4 w4"] = initial_disparity_b1hw

        prepared_context: PreparedContext = self._prepare_context(features_left)
        net_list: TensorPyramid = prepared_context[0]
        input_list: TensorPyramid = prepared_context[1]
        attention_list: TensorPyramid = prepared_context[2]
        geo_volume: CombinedGeoEncodingVolume = CombinedGeoEncodingVolume(
            features_left[0],
            features_right[0],
            gwc_volume_bgdhw,
            num_levels=self.corr_levels,
        )

        feature_shape: torch.Size = features_left[0].shape
        batch_size: int = feature_shape[0]
        height: int = feature_shape[2]
        width: int = feature_shape[3]
        coordinates_111w: Float32[Tensor, "1 1 w 1"] = torch.arange(
            width,
            dtype=disparity_b1hw.dtype,
            device=disparity_b1hw.device,
        ).reshape(1, 1, width, 1)
        coordinates_bhw1: Float32[Tensor, "b h w 1"] = coordinates_111w.repeat(batch_size, height, 1, 1)

        disparity_up_b1hw: Float32[Tensor, "b 1 h w"] | None = None
        for step in range(iteration_count):
            disparity_b1hw = disparity_b1hw.detach()
            geo_features_bchw: Float32[Tensor, "b correlation_features h4 w4"] = geo_volume(disparity_b1hw, coordinates_bhw1, self.dx)
            update_output: UpdateBlockOutput = self.update_block(
                net_list,
                input_list,
                geo_features_bchw,
                disparity_b1hw,
                attention_list,
            )
            net_list = update_output[0]
            mask_features_bchw: Float32[Tensor, "b mask_channels h4 w4"] = update_output[1]
            delta_disparity_b1hw: Float32[Tensor, "b 1 h4 w4"] = update_output[2]
            disparity_b1hw = disparity_b1hw + delta_disparity_b1hw
            if step == iteration_count - 1:
                disparity_up_b1hw = self.upsample_disp(disparity_b1hw, mask_features_bchw, stem_2_bchw)

        if disparity_up_b1hw is None:
            raise RuntimeError("LiteAnyStereoH did not produce an inference result.")
        return disparity_up_b1hw
