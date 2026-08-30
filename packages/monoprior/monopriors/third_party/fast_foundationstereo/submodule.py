"""Convolution, attention, cost-volume, and upsampling layers for Fast-FoundationStereo."""

import math
import os
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from monopriors.third_party.fast_foundationstereo.cost_volume_triton import (
    _create_gwc_triton_kernel as _create_gwc_triton_kernel,
)
from monopriors.third_party.fast_foundationstereo.cost_volume_triton import (
    build_gwc_volume_triton as build_gwc_volume_triton,
)

ActivationFactory: TypeAlias = Callable[[], nn.Module]
NormFactory: TypeAlias = Callable[[int], nn.Module]
NormChoice: TypeAlias = Literal["batch", "instance"]
SpatialArgument: TypeAlias = int | tuple[int, ...]


def _is_contiguous(tensor: Tensor) -> bool:
    """Return whether a tensor uses contiguous memory format.

    Args:
        tensor: Tensor with arbitrary dtype and shape.

    Returns:
        Whether the tensor is contiguous.
    """
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    """Apply channel-wise layer normalization to a 2D feature map."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        """Initialize channels-first layer normalization.

        Args:
            normalized_shape: Channel count.
            eps: Numerical-stability constant.
        """
        super().__init__(normalized_shape, eps=eps)

    def forward(self, input: Float[Tensor, "b channels h w"]) -> Float[Tensor, "b channels h w"]:
        """Normalize a floating-point feature map over channels.

        Args:
            input: Features with shape ``(batch, channels, height, width)``.

        Returns:
            Normalized features with shape ``(batch, channels, height, width)``.
        """
        x_bchw: Float[Tensor, "b channels h w"] = input
        if _is_contiguous(x_bchw):
            channels_last_bhwc: Float[Tensor, "b h w channels"] = rearrange(x_bchw, "b c h w -> b h w c")
            normalized_bhwc: Float[Tensor, "b h w channels"] = F.layer_norm(
                channels_last_bhwc,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
            output_bchw: Float[Tensor, "b channels h w"] = rearrange(normalized_bhwc, "b h w c -> b c h w").contiguous()
            return output_bchw
        variance_b1hw: Float[Tensor, "b 1 h w"]
        mean_b1hw: Float[Tensor, "b 1 h w"]
        variance_b1hw, mean_b1hw = torch.var_mean(x_bchw, dim=1, keepdim=True)
        normalized_bchw = (x_bchw - mean_b1hw) * torch.rsqrt(variance_b1hw + self.eps)
        output_bchw = normalized_bchw * self.weight[:, None, None] + self.bias[:, None, None]
        return output_bchw


class BasicConv(nn.Module):
    """Apply a 2D or 3D convolution with optional normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        bn: bool = True,
        relu: bool = True,
        norm: NormChoice = "batch",
        **kwargs: Any,
    ) -> None:
        """Initialize the convolution block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            deconv: Whether to use a transposed convolution.
            is_3d: Whether to use volumetric rather than image convolution.
            bn: Whether to apply normalization.
            relu: Whether to apply LeakyReLU.
            norm: Batch or instance normalization.
            **kwargs: Convolution arguments such as kernel size, stride, and padding.
        """
        super().__init__()
        self.relu: nn.Module | bool = nn.LeakyReLU(inplace=True) if relu else nn.Identity()
        self.use_bn: bool = bn
        self.bn: nn.Module = nn.Identity()
        if is_3d:
            self.conv: nn.Module = (
                nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
                if deconv
                else nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            )
            if self.use_bn:
                self.bn = nn.BatchNorm3d(out_channels) if norm == "batch" else nn.InstanceNorm3d(out_channels)
        else:
            self.conv = (
                nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
                if deconv
                else nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            )
            if self.use_bn:
                self.bn = nn.BatchNorm2d(out_channels) if norm == "batch" else nn.InstanceNorm2d(out_channels)

    def forward(self, x: Float[Tensor, "b channels ..."]) -> Float[Tensor, "b output_channels ..."]:
        """Transform a floating-point image or cost volume.

        Args:
            x: Tensor with shape ``(batch, channels, spatial...)``.

        Returns:
            Tensor with shape ``(batch, output_channels, output_spatial...)``.
        """
        output: Float[Tensor, "b output_channels ..."] = self.conv(x)
        if self.use_bn:
            output = self.bn(output)
        if isinstance(self.relu, bool):
            self.relu = nn.LeakyReLU(inplace=True) if self.relu else nn.Identity()
        output = self.relu(output)
        return output


class Conv3dNormActReduced(nn.Module):
    """Factor a 3D convolution into spatial and disparity convolutions."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        hidden: int | None = None,
        kernel_size: int = 3,
        kernel_disp: int | None = None,
        stride: int = 1,
        norm: NormFactory = nn.BatchNorm3d,
    ) -> None:
        """Initialize the reduced volumetric convolution.

        Args:
            C_in: Input channel count.
            C_out: Output channel count.
            hidden: Intermediate channel count; defaults to ``C_out``.
            kernel_size: Spatial kernel size.
            kernel_disp: Disparity kernel size; defaults to ``kernel_size``.
            stride: Spatial and disparity stride.
            norm: Normalization-layer factory.
        """
        super().__init__()
        resolved_kernel_disp: int = kernel_size if kernel_disp is None else kernel_disp
        resolved_hidden: int = C_out if hidden is None else hidden
        self.conv1: nn.Sequential = nn.Sequential(
            nn.Conv3d(
                C_in,
                resolved_hidden,
                kernel_size=(1, kernel_size, kernel_size),
                padding=(0, kernel_size // 2, kernel_size // 2),
                stride=(1, stride, stride),
            ),
            norm(resolved_hidden),
            nn.ReLU(),
        )
        self.conv2: nn.Sequential = nn.Sequential(
            nn.Conv3d(
                resolved_hidden,
                C_out,
                kernel_size=(resolved_kernel_disp, 1, 1),
                padding=(resolved_kernel_disp // 2, 0, 0),
                stride=(stride, 1, 1),
            ),
            norm(C_out),
            nn.ReLU(),
        )

    def forward(self, x_bcdhw: Float[Tensor, "b channels disparities h w"]) -> Float[Tensor, "b output_channels output_d output_h output_w"]:
        """Transform a floating-point cost volume.

        Args:
            x_bcdhw: Cost volume with shape ``(batch, channels, disparities, height, width)``.

        Returns:
            Cost volume with shape ``(batch, output_channels, output_disparities, output_height, output_width)``.
        """
        spatial_bcdhw: Float[Tensor, "b hidden disparities output_h output_w"] = self.conv1(x_bcdhw)
        output_bcdhw: Float[Tensor, "b output_channels output_d output_h output_w"] = self.conv2(spatial_bcdhw)
        return output_bcdhw


class ResnetBasicBlock(nn.Module):
    """Two-layer residual block for 2D feature maps."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: NormFactory | None = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        """Initialize a 2D residual block.

        Args:
            inplanes: Input channel count.
            planes: Output channel count.
            kernel_size: Convolution kernel size.
            stride: First-convolution stride.
            padding: Convolution padding.
            downsample: Optional residual projection.
            groups: Must be one.
            base_width: Must be 64.
            dilation: Must be one.
            norm_layer: Optional normalization-layer factory.
            bias: Whether convolutions use bias.
        """
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.norm_layer: NormFactory | None = norm_layer
        self.conv1: nn.Conv2d = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        self.bn1: nn.Module | None = norm_layer(planes) if norm_layer is not None else None
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv2d = nn.Conv2d(planes, planes, kernel_size=kernel_size, bias=bias, padding=padding)
        self.bn2: nn.Module | None = norm_layer(planes) if norm_layer is not None else None
        self.downsample: nn.Module | None = downsample
        self.stride: int = stride

    def forward(self, x_bchw: Float[Tensor, "b channels h w"]) -> Float[Tensor, "b channels h_out w_out"]:
        """Transform a floating-point feature map with a residual.

        Args:
            x_bchw: Features with shape ``(batch, channels, height, width)``.

        Returns:
            Features with shape ``(batch, channels, output_height, output_width)``.
        """
        identity_bchw: Float[Tensor, "b channels h_out w_out"] = x_bchw
        output_bchw: Float[Tensor, "b channels h_out w_out"] = self.conv1(x_bchw)
        if self.bn1 is not None:
            output_bchw = self.bn1(output_bchw)
        output_bchw = self.relu(output_bchw)
        output_bchw = self.conv2(output_bchw)
        if self.bn2 is not None:
            output_bchw = self.bn2(output_bchw)
        if self.downsample is not None:
            identity_bchw = self.downsample(x_bchw)
        output_bchw += identity_bchw
        output_bchw = self.relu(output_bchw)
        return output_bchw


class ResnetBasicBlock3D(nn.Module):
    """Two-layer residual block for 3D cost volumes."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: NormFactory | None = nn.BatchNorm3d,
        bias: bool = False,
    ) -> None:
        """Initialize a 3D residual block.

        Args:
            inplanes: Input channel count.
            planes: Output channel count.
            kernel_size: Convolution kernel size.
            stride: First-convolution stride.
            padding: Convolution padding.
            downsample: Optional residual projection.
            groups: Must be one.
            base_width: Must be 64.
            dilation: Must be one.
            norm_layer: Optional normalization-layer factory.
            bias: Whether convolutions use bias.
        """
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.norm_layer: NormFactory | None = norm_layer
        self.conv1: nn.Conv3d = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        self.bn1: nn.Module | None = norm_layer(planes) if norm_layer is not None else None
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.conv2: nn.Conv3d = nn.Conv3d(planes, planes, kernel_size=kernel_size, bias=bias, padding=padding)
        self.bn2: nn.Module | None = norm_layer(planes) if norm_layer is not None else None
        self.downsample: nn.Module | None = downsample
        self.stride: int = stride

    def forward(self, x_bcdhw: Float[Tensor, "b channels disparities h w"]) -> Float[Tensor, "b channels output_d output_h output_w"]:
        """Transform a floating-point cost volume with a residual.

        Args:
            x_bcdhw: Volume with shape ``(batch, channels, disparities, height, width)``.

        Returns:
            Volume with shape ``(batch, channels, output_disparities, output_height, output_width)``.
        """
        identity_bcdhw: Float[Tensor, "b channels output_d output_h output_w"] = x_bcdhw
        output_bcdhw: Float[Tensor, "b channels output_d output_h output_w"] = self.conv1(x_bcdhw)
        if self.bn1 is not None:
            output_bcdhw = self.bn1(output_bcdhw)
        output_bcdhw = self.relu(output_bcdhw)
        output_bcdhw = self.conv2(output_bcdhw)
        if self.bn2 is not None:
            output_bcdhw = self.bn2(output_bcdhw)
        if self.downsample is not None:
            identity_bcdhw = self.downsample(x_bcdhw)
        output_bcdhw += identity_bcdhw
        output_bcdhw = self.relu(output_bcdhw)
        return output_bcdhw


class FlashMultiheadAttention(nn.Module):
    """Projection wrapper around scaled dot-product attention."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Initialize multi-head attention.

        Args:
            embed_dim: Embedding width.
            num_heads: Attention-head count.
        """
        super().__init__()
        self.num_heads: int = num_heads
        self.embed_dim: int = embed_dim
        self.head_dim: int = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.q_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.k_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.v_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query_blc: Float[Tensor, "b length channels"],
        key_blc: Float[Tensor, "b length channels"],
        value_blc: Float[Tensor, "b length channels"],
        attn_mask: Tensor | None = None,
        window_size: tuple[int, int] = (-1, -1),
    ) -> Float[Tensor, "b length channels"]:
        """Apply the released attention layout.

        Args:
            query_blc: Query with shape ``(batch, length, channels)``.
            key_blc: Key with shape ``(batch, length, channels)``.
            value_blc: Value with shape ``(batch, length, channels)``.
            attn_mask: Unused upstream compatibility argument.
            window_size: Unused upstream compatibility argument.

        Returns:
            Attention output with shape ``(batch, length, channels)``.
        """
        queries_blhd: Float[Tensor, "b length heads head_dim"] = rearrange(
            self.q_proj(query_blc),
            "b length (heads head_dim) -> b length heads head_dim",
            heads=self.num_heads,
        )
        keys_blhd: Float[Tensor, "b length heads head_dim"] = rearrange(
            self.k_proj(key_blc),
            "b length (heads head_dim) -> b length heads head_dim",
            heads=self.num_heads,
        )
        values_blhd: Float[Tensor, "b length heads head_dim"] = rearrange(
            self.v_proj(value_blc),
            "b length (heads head_dim) -> b length heads head_dim",
            heads=self.num_heads,
        )
        attention_blhd: Float[Tensor, "b length heads head_dim"] = F.scaled_dot_product_attention(
            queries_blhd,
            keys_blhd,
            values_blhd,
        )
        attention_blc: Float[Tensor, "b length channels"] = rearrange(attention_blhd, "b length heads head_dim -> b length (heads head_dim)")
        output_blc: Float[Tensor, "b length channels"] = self.out_proj(attention_blc)
        return output_blc


class FlashAttentionTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer using the released attention wrapper."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        act: ActivationFactory = nn.GELU,
        norm: NormFactory = nn.LayerNorm,
    ) -> None:
        """Initialize the transformer layer.

        Args:
            embed_dim: Embedding width.
            num_heads: Attention-head count.
            dim_feedforward: Feed-forward hidden width.
            dropout: Dropout probability.
            act: Activation-layer factory.
            norm: Normalization-layer factory.
        """
        super().__init__()
        self.self_attn: FlashMultiheadAttention = FlashMultiheadAttention(embed_dim, num_heads)
        self.act: nn.Module = act()
        self.linear1: nn.Linear = nn.Linear(embed_dim, dim_feedforward)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.linear2: nn.Linear = nn.Linear(dim_feedforward, embed_dim)
        self.norm1: nn.Module = norm(embed_dim)
        self.norm2: nn.Module = norm(embed_dim)
        self.dropout1: nn.Dropout = nn.Dropout(dropout)
        self.dropout2: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        src_blc: Float[Tensor, "b length channels"],
        src_mask: Tensor | None = None,
        window_size: tuple[int, int] = (-1, -1),
    ) -> Float[Tensor, "b length channels"]:
        """Transform one floating-point token sequence.

        Args:
            src_blc: Tokens with shape ``(batch, length, channels)``.
            src_mask: Optional attention mask retained by the upstream interface.
            window_size: Window selector retained by the upstream interface.

        Returns:
            Tokens with shape ``(batch, length, channels)``.
        """
        input_dtype: torch.dtype = src_blc.dtype
        attention_blc: Float[Tensor, "b length channels"] = self.self_attn(
            src_blc,
            src_blc,
            src_blc,
            src_mask,
            window_size=window_size,
        )
        output_blc: Float[Tensor, "b length channels"] = src_blc + self.dropout1(attention_blc)
        output_blc = self.norm1(output_blc).to(input_dtype)
        feedforward_blc: Float[Tensor, "b length channels"] = self.linear2(self.dropout(self.act(self.linear1(output_blc))))
        output_blc = output_blc + self.dropout2(feedforward_blc)
        output_blc = self.norm2(output_blc).to(input_dtype)
        return output_blc


class Conv2x(nn.Module):
    """Resize a 2D feature map by two, merge a skip, and convolve."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        concat: bool = True,
        keep_concat: bool = True,
        bn: bool = True,
        relu: bool = True,
        keep_dispc: bool = False,
    ) -> None:
        """Initialize the released 2D resize-and-fuse block.

        Args:
            in_channels: Main input channel count.
            out_channels: Resized feature channel count.
            deconv: Whether to upsample instead of downsample.
            is_3d: Must be false for the released inference architecture.
            concat: Whether to concatenate rather than add the skip.
            keep_concat: Whether concatenation doubles the output channels.
            bn: Whether to apply batch normalization.
            relu: Whether to apply LeakyReLU in the fusion convolution.
            keep_dispc: Must be false for the released inference architecture.

        Raises:
            ValueError: If an unused 3D branch is requested.
        """
        super().__init__()
        if is_3d or keep_dispc:
            raise ValueError("The owned Fast-FoundationStereo fork retains only the released 2D Conv2x path.")
        self.concat: bool = concat
        self.is_3d: bool = False
        kernel_size: int = 4 if deconv else 3
        self.conv1: BasicConv = BasicConv(
            in_channels,
            out_channels,
            deconv=deconv,
            bn=bn,
            relu=True,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        if self.concat:
            multiplier: int = 2 if keep_concat else 1
            self.conv2: BasicConv = BasicConv(
                out_channels * 2,
                out_channels * multiplier,
                bn=bn,
                relu=relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv2 = BasicConv(out_channels, out_channels, bn=bn, relu=relu, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        x_bchw: Float[Tensor, "b channels h w"],
        rem_bchw: Float[Tensor, "b rem_channels rem_h rem_w"],
    ) -> Float[Tensor, "b output_channels rem_h rem_w"]:
        """Resize and merge two floating-point feature maps.

        Args:
            x_bchw: Main features with shape ``(batch, channels, height, width)``.
            rem_bchw: Skip features with shape ``(batch, rem_channels, rem_height, rem_width)``.

        Returns:
            Merged features with shape ``(batch, output_channels, rem_height, rem_width)``.
        """
        resized_bchw: Float[Tensor, "b resized_channels resized_h resized_w"] = self.conv1(x_bchw)
        if resized_bchw.shape != rem_bchw.shape:
            resized_bchw = F.interpolate(resized_bchw, size=(rem_bchw.shape[-2], rem_bchw.shape[-1]), mode="bilinear")
        combined_bchw: Float[Tensor, "b combined_channels rem_h rem_w"] = (
            torch.cat((resized_bchw, rem_bchw), dim=1) if self.concat else resized_bchw + rem_bchw
        )
        output_bchw: Float[Tensor, "b output_channels rem_h rem_w"] = self.conv2(combined_bchw)
        return output_bchw


class BasicConv_IN(nn.Module):
    """Apply a 2D convolution with optional instance normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        IN: bool = True,
        relu: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the instance-normalized convolution.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            deconv: Whether to use a transposed convolution.
            is_3d: Must be false for the released inference architecture.
            IN: Whether to apply instance normalization.
            relu: Whether to apply LeakyReLU.
            **kwargs: 2D convolution arguments.

        Raises:
            ValueError: If the unused 3D branch is requested.
        """
        super().__init__()
        if is_3d:
            raise ValueError("The owned Fast-FoundationStereo fork retains only the released 2D BasicConv_IN path.")
        self.relu: nn.Module | bool = nn.LeakyReLU(inplace=True) if relu else nn.Identity()
        self.use_in: bool = IN
        self.conv: nn.Module = (
            nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            if deconv
            else nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        )
        self.IN: nn.InstanceNorm2d = nn.InstanceNorm2d(out_channels)

    def forward(self, x_bchw: Float[Tensor, "b channels h w"]) -> Float[Tensor, "b output_channels output_h output_w"]:
        """Transform a floating-point feature map.

        Args:
            x_bchw: Features with shape ``(batch, channels, height, width)``.

        Returns:
            Features with shape ``(batch, output_channels, output_height, output_width)``.
        """
        output_bchw: Float[Tensor, "b output_channels output_h output_w"] = self.conv(x_bchw)
        if self.use_in:
            output_bchw = self.IN(output_bchw)
        if isinstance(self.relu, bool):
            self.relu = nn.LeakyReLU(inplace=True) if self.relu else nn.Identity()
        output_bchw = self.relu(output_bchw)
        return output_bchw


class Conv2x_IN(nn.Module):
    """Resize and fuse 2D feature maps with instance normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        c_middle: int | None = None,
        deconv: bool = False,
        is_3d: bool = False,
        concat: bool = True,
        keep_concat: bool = True,
        IN: bool = True,
        relu: bool = True,
        keep_dispc: bool = False,
    ) -> None:
        """Initialize the instance-normalized resize-and-fuse block.

        Args:
            in_channels: Main input channel count.
            out_channels: Skip feature channel count.
            c_middle: Resized feature channel count; defaults to ``out_channels``.
            deconv: Whether to upsample instead of downsample.
            is_3d: Must be false for the released inference architecture.
            concat: Whether to concatenate rather than add the skip.
            keep_concat: Whether concatenation doubles the output channels.
            IN: Whether to apply instance normalization.
            relu: Whether to apply LeakyReLU in the additive path.
            keep_dispc: Must be false for the released inference architecture.

        Raises:
            ValueError: If an unused 3D branch is requested.
        """
        super().__init__()
        if is_3d or keep_dispc:
            raise ValueError("The owned Fast-FoundationStereo fork retains only the released 2D Conv2x_IN path.")
        self.concat: bool = concat
        self.is_3d: bool = False
        middle_channels: int = out_channels if c_middle is None else c_middle
        kernel_size: int = 4 if deconv else 3
        self.conv1: BasicConv_IN = BasicConv_IN(
            in_channels,
            middle_channels,
            deconv=deconv,
            IN=True,
            relu=True,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        if self.concat:
            multiplier: int = 2 if keep_concat else 1
            self.conv2: nn.Module = ResnetBasicBlock(
                out_channels * 2,
                out_channels * multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=nn.InstanceNorm2d,
            )
        else:
            self.conv2 = BasicConv_IN(
                middle_channels,
                out_channels,
                IN=IN,
                relu=relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(
        self,
        x_bchw: Float[Tensor, "b channels h w"],
        rem_bchw: Float[Tensor, "b rem_channels rem_h rem_w"],
    ) -> Float[Tensor, "b output_channels rem_h rem_w"]:
        """Resize and merge two floating-point feature maps.

        Args:
            x_bchw: Main features with shape ``(batch, channels, height, width)``.
            rem_bchw: Skip features with shape ``(batch, rem_channels, rem_height, rem_width)``.

        Returns:
            Merged features with shape ``(batch, output_channels, rem_height, rem_width)``.
        """
        resized_bchw: Float[Tensor, "b resized_channels resized_h resized_w"] = self.conv1(x_bchw)
        if resized_bchw.shape != rem_bchw.shape:
            resized_bchw = F.interpolate(resized_bchw, size=(rem_bchw.shape[-2], rem_bchw.shape[-1]), mode="bilinear")
        combined_bchw: Float[Tensor, "b combined_channels rem_h rem_w"] = (
            torch.cat((resized_bchw, rem_bchw), dim=1) if self.concat else resized_bchw + rem_bchw
        )
        output_bchw: Float[Tensor, "b output_channels rem_h rem_w"] = self.conv2(combined_bchw)
        return output_bchw


def _build_gwc_volume_optimized_pytorch1(
    refimg_fea_bchw: Float[Tensor, "b channels h w"],
    targetimg_fea_bchw: Float[Tensor, "b channels h w"],
    maxdisp: int,
    num_groups: int,
    normalize: bool = True,
) -> Float[Tensor, "b groups disparities h w"]:
    """Run the compiler-compatible groupwise-correlation operations."""
    input_dtype: torch.dtype = refimg_fea_bchw.dtype
    batch_size: int = refimg_fea_bchw.shape[0]
    channels: int = refimg_fea_bchw.shape[1]
    height: int = refimg_fea_bchw.shape[2]
    width: int = refimg_fea_bchw.shape[3]
    channels_per_group: int = channels // num_groups
    ref_volume_bcdhw: Float[Tensor, "b channels disparities h w"] = refimg_fea_bchw.unsqueeze(2).expand(
        batch_size,
        channels,
        maxdisp,
        height,
        width,
    )
    padded_target_bchw: Float[Tensor, "b channels h padded_w"] = F.pad(targetimg_fea_bchw, (maxdisp - 1, 0, 0, 0))
    unfolded_target_bchdw: Float[Tensor, "b channels h disparities w"] = padded_target_bchw.unfold(3, width, 1)
    target_volume_bcdhw: Float[Tensor, "b channels disparities h w"] = rearrange(
        torch.flip(unfolded_target_bchdw, [3]),
        "b channels h disparities w -> b channels disparities h w",
    )
    grouped_ref_bgkdhw: Float[Tensor, "b groups channels_per_group disparities h w"] = ref_volume_bcdhw.view(
        batch_size,
        num_groups,
        channels_per_group,
        maxdisp,
        height,
        width,
    )
    grouped_target_bgkdhw: Float[Tensor, "b groups channels_per_group disparities h w"] = target_volume_bcdhw.view(
        batch_size,
        num_groups,
        channels_per_group,
        maxdisp,
        height,
        width,
    )
    if normalize:
        grouped_ref_bgkdhw = F.normalize(grouped_ref_bgkdhw.float(), dim=2).to(input_dtype)
        grouped_target_bgkdhw = F.normalize(grouped_target_bgkdhw.float(), dim=2).to(input_dtype)
    cost_volume_bgdhw: Float[Tensor, "b groups disparities h w"] = (grouped_ref_bgkdhw * grouped_target_bgkdhw).sum(dim=2)
    return cost_volume_bgdhw.contiguous()


build_gwc_volume_optimized_pytorch1_eager = _build_gwc_volume_optimized_pytorch1
build_gwc_volume_optimized_pytorch1 = torch.compile(
    build_gwc_volume_optimized_pytorch1_eager,
    disable=os.environ.get("PIXI_DEV_MODE") == "1",
)


def _build_concat_volume_optimized_pytorch1(
    refimg_fea_bchw: Float[Tensor, "b channels h w"],
    targetimg_fea_bchw: Float[Tensor, "b channels h w"],
    maxdisp: int,
) -> Float[Tensor, "b double_channels disparities h w"]:
    """Run the compiler-compatible concatenated-volume operations."""
    batch_size: int = refimg_fea_bchw.shape[0]
    channels: int = refimg_fea_bchw.shape[1]
    height: int = refimg_fea_bchw.shape[2]
    width: int = refimg_fea_bchw.shape[3]
    ref_volume_bcdhw: Float[Tensor, "b channels disparities h w"] = refimg_fea_bchw.unsqueeze(2).expand(
        batch_size,
        channels,
        maxdisp,
        height,
        width,
    )
    padded_target_bchw: Float[Tensor, "b channels h padded_w"] = F.pad(targetimg_fea_bchw, (maxdisp - 1, 0, 0, 0))
    unfolded_target_bchdw: Float[Tensor, "b channels h disparities w"] = padded_target_bchw.unfold(dimension=3, size=width, step=1)
    target_volume_bcdhw: Float[Tensor, "b channels disparities h w"] = rearrange(
        torch.flip(unfolded_target_bchdw, [3]),
        "b channels h disparities w -> b channels disparities h w",
    )
    volume_bcdhw: Float[Tensor, "b double_channels disparities h w"] = torch.cat((ref_volume_bcdhw, target_volume_bcdhw), dim=1)
    return volume_bcdhw.contiguous()


build_concat_volume_optimized_pytorch1_eager = _build_concat_volume_optimized_pytorch1
build_concat_volume_optimized_pytorch1 = torch.compile(
    build_concat_volume_optimized_pytorch1_eager,
    disable=os.environ.get("PIXI_DEV_MODE") == "1",
)


def disparity_regression(probability_bdhw: Float[Tensor, "b disparities h w"], maxdisp: int) -> Float[Tensor, "b 1 h w"]:
    """Compute expected disparity from a probability volume.

    Args:
        probability_bdhw: Disparity probabilities with shape ``(batch, disparities, height, width)``.
        maxdisp: Number of discrete disparity candidates.

    Returns:
        Expected disparity with shape ``(batch, 1, height, width)``.
    """
    assert probability_bdhw.ndim == 4
    disparity_values_d: Float[Tensor, "disparities"] = torch.arange(
        0,
        maxdisp,
        dtype=probability_bdhw.dtype,
        device=probability_bdhw.device,
    )
    disparity_values_1d11: Float[Tensor, "1 disparities 1 1"] = disparity_values_d.reshape(1, maxdisp, 1, 1)
    disparity_b1hw: Float[Tensor, "b 1 h w"] = torch.sum(probability_bdhw * disparity_values_1d11, 1, keepdim=True)
    return disparity_b1hw


class FeatureAtt(nn.Module):
    """Modulate a cost volume with image-conditioned channel attention."""

    def __init__(self, cv_chan: int, feat_chan: int) -> None:
        """Initialize feature attention.

        Args:
            cv_chan: Cost-volume channel count.
            feat_chan: Image-feature channel count.
        """
        super().__init__()
        self.feat_att: nn.Sequential = nn.Sequential(
            BasicConv(feat_chan, feat_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1),
        )

    def forward(
        self,
        cost_volume_bcdhw: Float[Tensor, "b channels disparities h w"],
        features_bchw: Float[Tensor, "b feature_channels h w"],
    ) -> Float[Tensor, "b channels disparities h w"]:
        """Apply image-conditioned sigmoid attention.

        Args:
            cost_volume_bcdhw: Cost volume with shape ``(batch, channels, disparities, height, width)``.
            features_bchw: Image features with shape ``(batch, feature_channels, height, width)``.

        Returns:
            Modulated cost volume with shape ``(batch, channels, disparities, height, width)``.
        """
        attention_bc1hw: Float[Tensor, "b channels 1 h w"] = self.feat_att(features_bchw).unsqueeze(2)
        output_bcdhw: Float[Tensor, "b channels disparities h w"] = torch.sigmoid(attention_bc1hw) * cost_volume_bcdhw
        return output_bcdhw


def context_upsample(
    disp_low_b1hw: Float[Tensor, "b 1 h w"],
    up_weights_b9hw: Float[Tensor, "b 9 h4 w4"],
) -> Float[Tensor, "b h4 w4"]:
    """Upsample disparity fourfold with learned 3-by-3 context weights.

    Args:
        disp_low_b1hw: Low-resolution disparity with shape ``(batch, 1, height, width)``.
        up_weights_b9hw: Weights with shape ``(batch, 9, 4 * height, 4 * width)``.

    Returns:
        Upsampled disparity with shape ``(batch, 4 * height, 4 * width)``.
    """
    height: int = disp_low_b1hw.shape[2]
    width: int = disp_low_b1hw.shape[3]
    unfolded_b9hw: Float[Tensor, "b 9 h w"] = rearrange(
        F.unfold(disp_low_b1hw, 3, 1, 1),
        "b neighbors (h w) -> b neighbors h w",
        h=height,
        w=width,
    )
    upsampled_b9hw: Float[Tensor, "b 9 h4 w4"] = F.interpolate(
        unfolded_b9hw,
        (height * 4, width * 4),
        mode="nearest",
    )
    disparity_bhw: Float[Tensor, "b h4 w4"] = (upsampled_b9hw * up_weights_b9hw).sum(1)
    return disparity_bhw


class PositionalEmbedding(nn.Module):
    """Add fixed sinusoidal disparity-position embeddings."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        """Initialize positional embeddings.

        Args:
            d_model: Embedding width.
            max_len: Maximum disparity-token count.
        """
        super().__init__()
        embedding_lc: Float[Tensor, "length channels"] = torch.zeros(max_len, d_model, dtype=torch.float32)
        positions_l1: Float[Tensor, "length 1"] = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        frequencies_1c: Float[Tensor, "1 half_channels"] = (
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        ).exp()[None]
        embedding_lc[:, 0::2] = torch.sin(positions_l1 * frequencies_1c)
        embedding_lc[:, 1::2] = torch.cos(positions_l1 * frequencies_1c)
        self.pe: Float[Tensor, "1 length channels"] = embedding_lc.unsqueeze(0)

    def forward(
        self,
        x_blc: Float[Tensor, "b length channels"],
        resize_embed: bool = False,
    ) -> Float[Tensor, "b length channels"]:
        """Add position embeddings to a token sequence.

        Args:
            x_blc: Tokens with shape ``(batch, length, channels)``.
            resize_embed: Whether to linearly extend embeddings for a longer sequence.

        Returns:
            Position-encoded tokens with shape ``(batch, length, channels)``.
        """
        input_dtype: torch.dtype = x_blc.dtype
        self.pe = self.pe.to(device=x_blc.device, dtype=x_blc.dtype)
        embedding_blc: Float[Tensor, "1 length channels"] = self.pe
        if embedding_blc.shape[1] < x_blc.shape[1]:
            if not resize_embed:
                raise RuntimeError(f"x:{x_blc.shape}, pe:{embedding_blc.shape}")
            embedding_bcl: Float[Tensor, "1 channels length"] = rearrange(embedding_blc, "b length channels -> b channels length")
            resized_bcl: Float[Tensor, "1 channels new_length"] = F.interpolate(
                embedding_bcl,
                size=x_blc.shape[1],
                mode="linear",
                align_corners=True,
            )
            embedding_blc = rearrange(resized_bcl, "b channels length -> b length channels")
        output_blc: Float[Tensor, "b length channels"] = (x_blc + embedding_blc[:, : x_blc.size(1)]).to(input_dtype)
        return output_blc


class CostVolumeDisparityAttention(nn.Module):
    """Apply transformer attention along a cost volume's disparity axis."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        act: ActivationFactory = nn.GELU,
        norm_first: bool = False,
        num_transformer: int = 6,
        max_len: int = 512,
        resize_embed: bool = False,
    ) -> None:
        """Initialize disparity attention.

        Args:
            d_model: Cost-volume channel count.
            nhead: Attention-head count.
            dim_feedforward: Feed-forward hidden width.
            dropout: Dropout probability.
            act: Activation-layer factory.
            norm_first: Upstream compatibility field; the released layer is post-normalized.
            num_transformer: Number of transformer layers.
            max_len: Positional-embedding length.
            resize_embed: Whether embeddings may be resized.
        """
        super().__init__()
        self.resize_embed: bool = resize_embed
        self.sa: nn.ModuleList = nn.ModuleList(
            [
                FlashAttentionTransformerEncoderLayer(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dim_feedforward=dim_feedforward,
                    act=act,
                    dropout=dropout,
                )
                for _ in range(num_transformer)
            ]
        )
        self.pos_embed0: PositionalEmbedding = PositionalEmbedding(d_model, max_len=max_len)

    def forward(
        self,
        cost_volume_bcdhw: Float[Tensor, "b channels disparities h w"],
        window_size: tuple[int, int] = (-1, -1),
    ) -> Float[Tensor, "b channels disparities h w"]:
        """Transform each pixel's disparity sequence.

        Args:
            cost_volume_bcdhw: Volume with shape ``(batch, channels, disparities, height, width)``.
            window_size: Attention window retained by the upstream interface.

        Returns:
            Volume with shape ``(batch, channels, disparities, height, width)``.
        """
        batch_size: int = cost_volume_bcdhw.shape[0]
        height: int = cost_volume_bcdhw.shape[3]
        width: int = cost_volume_bcdhw.shape[4]
        tokens_ndc: Float[Tensor, "pixels disparities channels"] = rearrange(
            cost_volume_bcdhw,
            "b channels disparities h w -> (b h w) disparities channels",
        )
        tokens_ndc = self.pos_embed0(tokens_ndc, resize_embed=self.resize_embed)
        for layer in self.sa:
            tokens_ndc = layer(tokens_ndc, window_size=window_size)
        output_bcdhw: Float[Tensor, "b channels disparities h w"] = rearrange(
            tokens_ndc,
            "(b h w) disparities channels -> b channels disparities h w",
            b=batch_size,
            h=height,
            w=width,
        )
        return output_bcdhw


class ChannelAttentionEnhancement(nn.Module):
    """Predict per-channel recurrent-context weights."""

    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        """Initialize channel attention.

        Args:
            in_planes: Input and output channel count.
            ratio: Hidden-channel reduction ratio.
        """
        super().__init__()
        hidden_channels: int = in_planes // ratio
        self.avg_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.max_pool: nn.AdaptiveMaxPool2d = nn.AdaptiveMaxPool2d(1)
        self.fc: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_planes, hidden_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_planes, 1, bias=False),
        )
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x_bchw: Float[Tensor, "b channels h w"]) -> Float[Tensor, "b channels 1 1"]:
        """Predict floating-point channel weights.

        Args:
            x_bchw: Features with shape ``(batch, channels, height, width)``.

        Returns:
            Weights with shape ``(batch, channels, 1, 1)``.
        """
        average_bchw: Float[Tensor, "b channels 1 1"] = self.fc(self.avg_pool(x_bchw))
        maximum_bchw: Float[Tensor, "b channels 1 1"] = self.fc(self.max_pool(x_bchw))
        weights_bchw: Float[Tensor, "b channels 1 1"] = self.sigmoid(average_bchw + maximum_bchw)
        return weights_bchw


class SpatialAttentionExtractor(nn.Module):
    """Predict per-pixel recurrent update weights."""

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize spatial attention.

        Args:
            kernel_size: Spatial convolution kernel size.
        """
        super().__init__()
        self.samconv: nn.Conv2d = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x_bchw: Float[Tensor, "b channels h w"]) -> Float[Tensor, "b 1 h w"]:
        """Predict floating-point spatial weights.

        Args:
            x_bchw: Features with shape ``(batch, channels, height, width)``.

        Returns:
            Weights with shape ``(batch, 1, height, width)``.
        """
        average_b1hw: Float[Tensor, "b 1 h w"] = torch.mean(x_bchw, dim=1, keepdim=True)
        maximum_b1hw: Float[Tensor, "b 1 h w"] = torch.max(x_bchw, dim=1, keepdim=True)[0]
        pooled_b2hw: Float[Tensor, "b 2 h w"] = torch.cat([average_b1hw, maximum_b1hw], dim=1)
        weights_b1hw: Float[Tensor, "b 1 h w"] = self.sigmoid(self.samconv(pooled_b2hw))
        return weights_b1hw


class EdgeNextConvEncoder(nn.Module):
    """EdgeNeXt depthwise convolution and pointwise MLP residual block."""

    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        expan_ratio: int = 4,
        kernel_size: int = 7,
        norm: Literal["layer", "batch"] | None = "layer",
    ) -> None:
        """Initialize the convolution encoder.

        Args:
            dim: Input and output channel count.
            layer_scale_init_value: Initial residual-branch scale.
            expan_ratio: Pointwise MLP expansion ratio.
            kernel_size: Depthwise convolution kernel size.
            norm: Layer normalization, batch normalization, or no normalization.
        """
        super().__init__()
        self.dwconv: nn.Conv2d = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        if norm == "layer":
            self.norm: nn.Module = LayerNorm2d(dim, eps=1e-6)
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.norm = nn.Identity()
        self.pwconv1: nn.Linear = nn.Linear(dim, expan_ratio * dim)
        self.act: nn.GELU = nn.GELU()
        self.pwconv2: nn.Linear = nn.Linear(expan_ratio * dim, dim)
        self.gamma: nn.Parameter | None = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0.0 else None
        )

    def forward(self, x_bchw: Float[Tensor, "b channels h w"]) -> Float[Tensor, "b channels h w"]:
        """Transform a floating-point feature map with a residual.

        Args:
            x_bchw: Features with shape ``(batch, channels, height, width)``.

        Returns:
            Features with shape ``(batch, channels, height, width)``.
        """
        shortcut_bchw: Float[Tensor, "b channels h w"] = x_bchw
        output_bchw: Float[Tensor, "b channels h w"] = self.norm(self.dwconv(x_bchw))
        output_bhwc: Float[Tensor, "b h w channels"] = rearrange(output_bchw, "b c h w -> b h w c")
        output_bhwc = self.pwconv2(self.act(self.pwconv1(output_bhwc)))
        if self.gamma is not None:
            output_bhwc = self.gamma * output_bhwc
        output_bchw = rearrange(output_bhwc, "b h w c -> b c h w")
        output_bchw = shortcut_bchw + output_bchw
        return output_bchw
