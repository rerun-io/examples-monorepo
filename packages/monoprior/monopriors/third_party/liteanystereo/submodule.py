"""Shared convolution, correlation, and upsampling layers for LiteAnyStereo V2."""

from collections.abc import Callable
from functools import partial
from typing import Any, TypeAlias

import torch
import torch.nn.functional as F
from jaxtyping import Float32
from torch import Tensor, nn

ActivationFactory: TypeAlias = Callable[[], nn.Module]
NormFactory: TypeAlias = Callable[[int], nn.Module]
SpatialArgument: TypeAlias = int | tuple[int, int]


class BasicConv2d(nn.Module):
    """Apply a 2D convolution followed by optional normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: SpatialArgument = 3,
        stride: SpatialArgument = 1,
        padding: SpatialArgument = 0,
        bias: bool = False,
        norm_layer: NormFactory | None = None,
        act_layer: ActivationFactory | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Input padding.
            bias: Whether the convolution has a bias parameter.
            norm_layer: Optional normalization-layer factory.
            act_layer: Optional activation-layer factory.
            **kwargs: Extra keyword arguments forwarded to ``nn.Conv2d``.
        """
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                **kwargs,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())

        self.block: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x_bchw: Float32[Tensor, "b c_in h w"]) -> Float32[Tensor, "b c_out h_out w_out"]:
        """Transform one float32 feature map.

        Args:
            x_bchw: Float32 input tensor with shape ``(batch, input_channels, height, width)``.

        Returns:
            Float32 output tensor with shape ``(batch, output_channels, output_height, output_width)``.
        """
        output_bchw: Float32[Tensor, "b c_out h_out w_out"] = self.block(x_bchw)
        return output_bchw


class BasicDeconv2d(nn.Module):
    """Apply a 2D transposed convolution followed by optional normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: SpatialArgument,
        stride: SpatialArgument = 1,
        padding: SpatialArgument = 0,
        bias: bool = False,
        norm_layer: NormFactory | None = None,
        act_layer: ActivationFactory | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the transposed-convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Transposed-convolution kernel size.
            stride: Transposed-convolution stride.
            padding: Input padding.
            bias: Whether the convolution has a bias parameter.
            norm_layer: Optional normalization-layer factory.
            act_layer: Optional activation-layer factory.
            **kwargs: Extra keyword arguments forwarded to ``nn.ConvTranspose2d``.
        """
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                **kwargs,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())

        self.block: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x_bchw: Float32[Tensor, "b c_in h w"]) -> Float32[Tensor, "b c_out h_out w_out"]:
        """Upsample one float32 feature map.

        Args:
            x_bchw: Float32 input tensor with shape ``(batch, input_channels, height, width)``.

        Returns:
            Float32 output tensor with shape ``(batch, output_channels, output_height, output_width)``.
        """
        output_bchw: Float32[Tensor, "b c_out h_out w_out"] = self.block(x_bchw)
        return output_bchw


class FPNLayer(nn.Module):
    """Fuse one low-resolution feature map with a higher-resolution skip feature."""

    def __init__(self, chan_low: int, chan_high: int) -> None:
        """Initialize the feature-pyramid layer.

        Args:
            chan_low: Number of low-resolution input channels.
            chan_high: Number of skip and output channels.
        """
        super().__init__()
        self.deconv: BasicDeconv2d = BasicDeconv2d(
            chan_low,
            chan_high,
            kernel_size=4,
            stride=2,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
        )
        self.conv: BasicConv2d = BasicConv2d(
            chan_high * 2,
            chan_high,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True),
        )

    def forward(
        self,
        low_bchw: Float32[Tensor, "b c_low h_low w_low"],
        high_bchw: Float32[Tensor, "b c_high h_high w_high"],
    ) -> Float32[Tensor, "b c_high h_high w_high"]:
        """Fuse low- and high-resolution float32 feature maps.

        Args:
            low_bchw: Float32 low-resolution tensor with shape ``(batch, low_channels, low_height, low_width)``.
            high_bchw: Float32 skip tensor with shape ``(batch, high_channels, high_height, high_width)``.

        Returns:
            Float32 fused tensor with shape ``(batch, high_channels, high_height, high_width)``.
        """
        upsampled_bchw: Float32[Tensor, "b c_high h_high w_high"] = self.deconv(low_bchw)
        concatenated_bchw: Float32[Tensor, "b two_c_high h_high w_high"] = torch.cat([high_bchw, upsampled_bchw], 1)
        fused_bchw: Float32[Tensor, "b c_high h_high w_high"] = self.conv(concatenated_bchw)
        return fused_bchw


def disparity_regression(probability_bdhw: Float32[Tensor, "b disparities h w"], max_disp: int) -> Float32[Tensor, "b 1 h w"]:
    """Compute expected disparity from a float32 probability volume.

    Args:
        probability_bdhw: Float32 disparity probabilities with shape ``(batch, disparities, height, width)``.
        max_disp: Number of discrete disparity values.

    Returns:
        Float32 expected disparity with shape ``(batch, 1, height, width)``.
    """
    assert probability_bdhw.ndim == 4
    disparity_values_d: Float32[Tensor, "disparities"] = torch.arange(
        0,
        max_disp,
        dtype=probability_bdhw.dtype,
        device=probability_bdhw.device,
    )
    disparity_values_1d11: Float32[Tensor, "1 disparities 1 1"] = disparity_values_d.view(1, max_disp, 1, 1)
    disparity_b1hw: Float32[Tensor, "b 1 h w"] = torch.sum(probability_bdhw * disparity_values_1d11, 1, keepdim=True)
    return disparity_b1hw


def build_gwc_volume_fast(
    reference_bchw: Float32[Tensor, "b channels h w"],
    target_bchw: Float32[Tensor, "b channels h w"],
    max_disp: int,
    num_groups: int,
) -> Float32[Tensor, "b groups disparities h w"]:
    """Build the groupwise-correlation cost volume used by LAS2-H.

    Args:
        reference_bchw: Float32 left features with shape ``(batch, channels, height, width)``.
        target_bchw: Float32 right features with shape ``(batch, channels, height, width)``.
        max_disp: Number of feature-space disparity candidates.
        num_groups: Number of channel groups.

    Returns:
        Float32 correlation volume with shape ``(batch, groups, disparities, height, width)``.
    """
    shape_bchw: torch.Size = reference_bchw.shape
    batch_size: int = shape_bchw[0]
    channels: int = shape_bchw[1]
    height: int = shape_bchw[2]
    width: int = shape_bchw[3]
    assert channels % num_groups == 0
    channels_per_group: int = channels // num_groups

    reference_bcdhw: Float32[Tensor, "b channels disparities h w"] = reference_bchw.unsqueeze(2).expand(
        batch_size, channels, max_disp, height, width
    )
    padded_target_bchw: Float32[Tensor, "b channels h padded_w"] = F.pad(target_bchw, (max_disp - 1, 0, 0, 0))
    unfolded_target_bchdw: Float32[Tensor, "b channels h disparities w"] = padded_target_bchw.unfold(3, width, 1)
    target_bcdhw: Float32[Tensor, "b channels disparities h w"] = torch.flip(unfolded_target_bchdw, [3]).permute(0, 1, 3, 2, 4)

    grouped_reference_bgcdhw: Float32[Tensor, "b groups channels_per_group disparities h w"] = reference_bcdhw.view(
        batch_size, num_groups, channels_per_group, max_disp, height, width
    )
    grouped_target_bgcdhw: Float32[Tensor, "b groups channels_per_group disparities h w"] = target_bcdhw.view(
        batch_size, num_groups, channels_per_group, max_disp, height, width
    )
    volume_bgdhw: Float32[Tensor, "b groups disparities h w"] = (grouped_reference_bgcdhw * grouped_target_bgcdhw).mean(dim=2)
    return volume_bgdhw.contiguous()


def build_correlation_volume(
    left_feature_bchw: Float32[Tensor, "b channels h w"],
    right_feature_bchw: Float32[Tensor, "b channels h w"],
    max_disp: int,
) -> Float32[Tensor, "b disparities h w"]:
    """Build the scalar-correlation cost volume used by LAS2-S/M/L.

    Args:
        left_feature_bchw: Float32 left features with shape ``(batch, channels, height, width)``.
        right_feature_bchw: Float32 right features with shape ``(batch, channels, height, width)``.
        max_disp: Number of feature-space disparity candidates.

    Returns:
        Float32 cost volume with shape ``(batch, disparities, height, width)``.
    """
    shape_bchw: torch.Size = left_feature_bchw.shape
    batch_size: int = shape_bchw[0]
    channels: int = shape_bchw[1]
    height: int = shape_bchw[2]
    width: int = shape_bchw[3]

    left_volume_bcdhw: Float32[Tensor, "b channels disparities h w"] = left_feature_bchw.unsqueeze(2).expand(
        batch_size, channels, max_disp, height, width
    )
    padded_right_bchw: Float32[Tensor, "b channels h padded_w"] = F.pad(right_feature_bchw, (max_disp - 1, 0, 0, 0))
    unfolded_right_bchdw: Float32[Tensor, "b channels h disparities w"] = padded_right_bchw.unfold(3, width, 1)
    right_volume_bcdhw: Float32[Tensor, "b channels disparities h w"] = torch.flip(unfolded_right_bchdw, [3]).permute(0, 1, 3, 2, 4)

    cost_volume_bdhw: Float32[Tensor, "b disparities h w"] = (left_volume_bcdhw * right_volume_bcdhw).mean(dim=1)
    return cost_volume_bdhw.contiguous()


def context_upsample(
    depth_low_b1hw: Float32[Tensor, "b 1 h w"],
    up_weights_b9hw: Float32[Tensor, "b 9 h4 w4"],
) -> Float32[Tensor, "b 1 h4 w4"]:
    """Upsample disparity fourfold with learned 3-by-3 context weights.

    Args:
        depth_low_b1hw: Float32 low-resolution disparity with shape ``(batch, 1, height, width)``.
        up_weights_b9hw: Float32 weights with shape ``(batch, 9, 4 * height, 4 * width)``.

    Returns:
        Float32 upsampled disparity with shape ``(batch, 1, 4 * height, 4 * width)``.
    """
    shape_b1hw: torch.Size = depth_low_b1hw.shape
    batch_size: int = shape_b1hw[0]
    channels: int = shape_b1hw[1]
    height: int = shape_b1hw[2]
    width: int = shape_b1hw[3]

    depth_unfold_b9hw: Float32[Tensor, "b 9 h w"] = F.unfold(depth_low_b1hw.reshape(batch_size, channels, height, width), 3, 1, 1).reshape(
        batch_size, -1, height, width
    )
    depth_unfold_b9h4w4: Float32[Tensor, "b 9 h4 w4"] = F.interpolate(
        depth_unfold_b9hw,
        (height * 4, width * 4),
        mode="nearest",
    ).reshape(batch_size, 9, height * 4, width * 4)
    depth_b1h4w4: Float32[Tensor, "b 1 h4 w4"] = torch.sum(depth_unfold_b9h4w4 * up_weights_b9hw, dim=1, keepdim=True)
    return depth_b1h4w4
