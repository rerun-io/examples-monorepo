"""ZipDepth model architecture with runtime-checkable tensor annotations."""

from typing import Self, TypeAlias, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class ModelConfig(TypedDict):
    """Configuration for one ZipDepth model variant."""

    dims: list[int]
    depths: list[int]
    heads: int
    dec_ch: int
    half_dec_ch: int
    use_global: bool


class ModelInfo(TypedDict):
    """Summary information returned for a ZipDepth model."""

    variant: str
    dims: list[int]
    depths: list[int]
    dec_ch: int
    half_dec_ch: int
    parameters_M: float
    global_mode: str


FeaturePyramid: TypeAlias = list[Float[torch.Tensor, "b _channels _height _width"]]
EncoderOutput: TypeAlias = tuple[Float[torch.Tensor, "b c_half h_half w_half"], FeaturePyramid]
ConvBNFusion: TypeAlias = tuple[
    Float[torch.Tensor, "c_out c_in_per_group kernel_h kernel_w"],
    Float[torch.Tensor, "c_out"],
]

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": {
        "dims": [24, 48, 96, 192],
        "depths": [2, 2, 4, 2],
        "heads": 4,
        "dec_ch": 32,
        "half_dec_ch": 24,
        "use_global": True,
    },
    "base": {
        "dims": [48, 96, 192, 384],
        "depths": [2, 2, 6, 2],
        "heads": 4,
        "dec_ch": 96,
        "half_dec_ch": 32,
        "use_global": True,
    },
    "large": {
        "dims": [64, 128, 256, 384],
        "depths": [2, 4, 10, 4],
        "heads": 8,
        "dec_ch": 192,
        "half_dec_ch": 48,
        "use_global": True,
    },
    "giant": {
        "dims": [96, 192, 384, 512],
        "depths": [2, 4, 14, 6],
        "heads": 8,
        "dec_ch": 288,
        "half_dec_ch": 64,
        "use_global": True,
    },
}


# =============================================================================
# CORE UTILITIES
# =============================================================================
def count_parameters(model: nn.Module) -> float:
    """Count trainable parameters.

    Args:
        model: Model whose trainable parameters are counted.

    Returns:
        Number of trainable parameters, in millions.
    """
    parameter_count: int = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return parameter_count / 1e6


class ConvBN(nn.Module):
    """Apply a convolution, batch normalization, and optional ReLU activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = (k + (k - 1) * (d - 1)) // 2
        self.conv: nn.Conv2d = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=g, bias=False)
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_ch)
        self.act: nn.Module = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Float[torch.Tensor, "b c_in h w"]) -> Float[torch.Tensor, "b c_out h_out w_out"]:
        """Transform one feature map.

        Args:
            x: Float tensor with shape ``(batch, input_channels, height, width)``.

        Returns:
            Float tensor with shape ``(batch, output_channels, output_height, output_width)``.
        """
        out_bchw: Float[torch.Tensor, "b c_out h_out w_out"] = self.act(self.bn(self.conv(x)))
        return out_bchw


# =============================================================================
# REPARAMETERIZABLE BLOCKS
# =============================================================================
class QARepBlock(nn.Module):
    """Apply a reparameterizable RepVGG-style block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1, act: bool = True) -> None:
        super().__init__()
        self.in_ch: int = in_ch
        self.out_ch: int = out_ch
        self.stride: int = stride
        self.groups: int = groups
        self.has_identity: bool = in_ch == out_ch and stride == 1

        self.branch_3x3: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.branch_1x1: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, 0, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act: nn.Module = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Float[torch.Tensor, "b c_in h w"]) -> Float[torch.Tensor, "b c_out h_out w_out"]:
        """Transform one feature map through the training or fused path.

        Args:
            x: Float tensor with shape ``(batch, input_channels, height, width)``.

        Returns:
            Float tensor with shape ``(batch, output_channels, output_height, output_width)``.
        """
        if hasattr(self, "fused_conv"):
            fused_out_bchw: Float[torch.Tensor, "b c_out h_out w_out"] = self.act(self.fused_conv(x))
            return fused_out_bchw

        out_bchw: Float[torch.Tensor, "b c_out h_out w_out"] = self.branch_3x3(x) + self.branch_1x1(x)
        if self.has_identity:
            out_bchw = out_bchw + x
        activated_bchw: Float[torch.Tensor, "b c_out h_out w_out"] = self.act(out_bchw)
        return activated_bchw

    def fuse(self) -> None:
        """Replace the training branches with one equivalent convolution."""
        if hasattr(self, "fused_conv"):
            return

        branch_3x3_conv_module: nn.Module = self.branch_3x3[0]
        branch_3x3_bn_module: nn.Module = self.branch_3x3[1]
        branch_1x1_conv_module: nn.Module = self.branch_1x1[0]
        branch_1x1_bn_module: nn.Module = self.branch_1x1[1]
        if not isinstance(branch_3x3_conv_module, nn.Conv2d) or not isinstance(branch_1x1_conv_module, nn.Conv2d):
            raise TypeError("QARepBlock convolution branches have unexpected module types")
        if not isinstance(branch_3x3_bn_module, nn.BatchNorm2d) or not isinstance(branch_1x1_bn_module, nn.BatchNorm2d):
            raise TypeError("QARepBlock batch-normalization branches have unexpected module types")
        branch_3x3_conv: nn.Conv2d = branch_3x3_conv_module
        branch_3x3_bn: nn.BatchNorm2d = branch_3x3_bn_module
        branch_1x1_conv: nn.Conv2d = branch_1x1_conv_module
        branch_1x1_bn: nn.BatchNorm2d = branch_1x1_bn_module
        fused_3x3: ConvBNFusion = self._fuse_conv_bn(branch_3x3_conv, branch_3x3_bn)
        kernel_3x3: Float[torch.Tensor, "c_out c_in_per_group 3 3"] = fused_3x3[0]
        bias_3x3: Float[torch.Tensor, "c_out"] = fused_3x3[1]
        fused_1x1: ConvBNFusion = self._fuse_conv_bn(branch_1x1_conv, branch_1x1_bn)
        kernel_1x1: Float[torch.Tensor, "c_out c_in_per_group 1 1"] = fused_1x1[0]
        bias_1x1: Float[torch.Tensor, "c_out"] = fused_1x1[1]
        kernel_1x1_padded: Float[torch.Tensor, "c_out c_in_per_group 3 3"] = F.pad(kernel_1x1, [1, 1, 1, 1])

        kernel: Float[torch.Tensor, "c_out c_in_per_group 3 3"] = kernel_3x3 + kernel_1x1_padded
        bias: Float[torch.Tensor, "c_out"] = bias_3x3 + bias_1x1

        if self.has_identity:
            identity_kernel: Float[torch.Tensor, "c_out c_in_per_group 3 3"] = torch.zeros_like(kernel)
            for i in range(self.in_ch):
                identity_kernel[i, i % (self.in_ch // self.groups), 1, 1] = 1.0
            kernel = kernel + identity_kernel

        self.fused_conv: nn.Conv2d = nn.Conv2d(self.in_ch, self.out_ch, 3, self.stride, 1, groups=self.groups, bias=True)
        self.fused_conv.weight.data = kernel
        if self.fused_conv.bias is None:
            raise RuntimeError("fused QARepBlock convolution unexpectedly has no bias")
        self.fused_conv.bias.data = bias

        del self.branch_3x3, self.branch_1x1

    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> ConvBNFusion:
        """Combine convolution and batch-normalization tensors.

        Args:
            conv: Convolution whose float weights have shape
                ``(output_channels, input_channels_per_group, kernel_height, kernel_width)``.
            bn: Batch normalization with one float statistic per output channel.

        Returns:
            Fused float weights and bias with shapes
            ``(output_channels, input_channels_per_group, kernel_height, kernel_width)`` and
            ``(output_channels,)``.
        """
        if bn.running_mean is None or bn.running_var is None:
            raise ValueError("BatchNorm without running statistics cannot be fused")
        if bn.weight is None or bn.bias is None:
            raise ValueError("BatchNorm without affine parameters cannot be fused")
        weights: Float[torch.Tensor, "c_out c_in_per_group kernel_h kernel_w"] = conv.weight
        running_mean: Float[torch.Tensor, "c_out"] = bn.running_mean
        running_var: Float[torch.Tensor, "c_out"] = bn.running_var
        gamma: Float[torch.Tensor, "c_out"] = bn.weight
        beta: Float[torch.Tensor, "c_out"] = bn.bias
        epsilon: float = bn.eps
        std: Float[torch.Tensor, "c_out"] = (running_var + epsilon).sqrt()
        scale: Float[torch.Tensor, "c_out 1 1 1"] = (gamma / std).reshape(-1, 1, 1, 1)
        fused_weights: Float[torch.Tensor, "c_out c_in_per_group kernel_h kernel_w"] = weights * scale
        fused_bias: Float[torch.Tensor, "c_out"] = beta - running_mean * gamma / std
        return fused_weights, fused_bias


# =============================================================================
# CHANNEL ATTENTION
# =============================================================================
class ChannelAttention(nn.Module):
    """Apply squeeze-and-excitation channel attention."""

    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        hidden: int = max(dim // reduction, 4)
        self.pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.fc: nn.Sequential = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Float[torch.Tensor, "b c h w"]) -> Float[torch.Tensor, "b c h w"]:
        """Reweight the input channels.

        Args:
            x: Float tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float tensor with shape ``(batch, channels, height, width)``.
        """
        channel_weights_bc11: Float[torch.Tensor, "b c 1 1"] = self.fc(self.pool(x))
        out_bchw: Float[torch.Tensor, "b c h w"] = x * channel_weights_bc11
        return out_bchw


# =============================================================================
# EfficientGlobalAttention
# =============================================================================
class EfficientGlobalAttention(nn.Module):
    """Apply global attention through learnable tokens."""

    def __init__(self, dim: int, num_tokens: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        self.num_tokens: int = num_tokens
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        self.scale: float = self.head_dim**-0.5

        self.tokens: nn.Parameter = nn.Parameter(torch.randn(1, num_tokens, dim))
        nn.init.trunc_normal_(self.tokens, std=0.02)

        self.q_tokens: nn.Linear = nn.Linear(dim, dim, bias=False)
        self.kv_spatial: nn.Conv2d = nn.Conv2d(dim, dim * 2, 1, bias=False)

        self.q_spatial: nn.Conv2d = nn.Conv2d(dim, dim, 1, groups=num_heads, bias=False)
        self.k_proj_tokens: nn.Linear = nn.Linear(dim, dim, bias=False)

        self.proj_out: nn.Conv2d = nn.Conv2d(dim, dim, 1)
        self.norm: nn.BatchNorm2d = nn.BatchNorm2d(dim)

    def forward(self, x: Float[torch.Tensor, "b c h w"]) -> Float[torch.Tensor, "b c h w"]:
        """Exchange information between spatial positions and global tokens.

        Args:
            x: Float tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float tensor with shape ``(batch, channels, height, width)``.
        """
        batch_size: int = x.shape[0]
        channels: int = x.shape[1]
        height: int = x.shape[2]
        width: int = x.shape[3]
        spatial_tokens: int = height * width

        # ---- Step 1: tokens attend spatial (aggregate) ----
        key_values_2bcn: Float[torch.Tensor, "2 b c spatial"] = (
            self.kv_spatial(x).reshape(batch_size, 2, channels, spatial_tokens).permute(1, 0, 2, 3)
        )
        spatial_keys_bcn: Float[torch.Tensor, "b c spatial"] = key_values_2bcn[0]
        spatial_values_bcn: Float[torch.Tensor, "b c spatial"] = key_values_2bcn[1]

        spatial_keys_bhnd: Float[torch.Tensor, "b heads spatial head_dim"] = spatial_keys_bcn.view(
            batch_size, self.num_heads, self.head_dim, spatial_tokens
        ).transpose(-1, -2)
        spatial_values_bhnd: Float[torch.Tensor, "b heads spatial head_dim"] = spatial_values_bcn.view(
            batch_size, self.num_heads, self.head_dim, spatial_tokens
        ).transpose(-1, -2)

        token_queries_bhtd: Float[torch.Tensor, "b heads tokens head_dim"] = (
            self.q_tokens(self.tokens).reshape(1, self.num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3).expand(batch_size, -1, -1, -1)
        )

        token_to_spatial_attention_bhtn: Float[torch.Tensor, "b heads tokens spatial"] = (
            token_queries_bhtd @ spatial_keys_bhnd.transpose(-2, -1)
        ) * self.scale
        tokens_updated_bhtd: Float[torch.Tensor, "b heads tokens head_dim"] = F.softmax(token_to_spatial_attention_bhtn, dim=-1) @ spatial_values_bhnd

        spatial_queries_bhnd: Float[torch.Tensor, "b heads spatial head_dim"] = (
            self.q_spatial(x)
            .reshape(batch_size, channels, spatial_tokens)
            .view(batch_size, self.num_heads, self.head_dim, spatial_tokens)
            .transpose(-1, -2)
        )

        token_keys_bhtd: Float[torch.Tensor, "b heads tokens head_dim"] = (
            self.k_proj_tokens(self.tokens)
            .reshape(1, self.num_tokens, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .expand(batch_size, -1, -1, -1)
        )
        token_values_bhtd: Float[torch.Tensor, "b heads tokens head_dim"] = tokens_updated_bhtd

        spatial_to_token_attention_bhnt: Float[torch.Tensor, "b heads spatial tokens"] = F.softmax(
            (spatial_queries_bhnd @ token_keys_bhtd.transpose(-2, -1)) * self.scale, dim=-1
        )
        attended_bhnd: Float[torch.Tensor, "b heads spatial head_dim"] = spatial_to_token_attention_bhnt @ token_values_bhtd

        attended_bchw: Float[torch.Tensor, "b c h w"] = (
            attended_bhnd.transpose(1, 2).reshape(batch_size, spatial_tokens, channels).transpose(1, 2).reshape(batch_size, channels, height, width)
        )
        out_bchw: Float[torch.Tensor, "b c h w"] = x + self.norm(self.proj_out(attended_bchw))
        return out_bchw


# =============================================================================
# StripPoolingAttention
# =============================================================================
class StripPoolingAttention(nn.Module):
    """Apply strip-pooling attention along the two spatial axes."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(dim, dim, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Float[torch.Tensor, "b c h w"]) -> Float[torch.Tensor, "b c h w"]:
        """Reweight a feature map from horizontal and vertical summaries.

        Args:
            x: Float tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float tensor with shape ``(batch, channels, height, width)``.
        """
        h_strip_bch1: Float[torch.Tensor, "b c h 1"] = x.mean(dim=3, keepdim=True)
        w_strip_bc1w: Float[torch.Tensor, "b c 1 w"] = x.mean(dim=2, keepdim=True)

        gate_bchw: Float[torch.Tensor, "b c h w"] = self.gate_conv(h_strip_bch1 + w_strip_bc1w)
        out_bchw: Float[torch.Tensor, "b c h w"] = x * gate_bchw
        return out_bchw


# =============================================================================
# GlobalContextBlock
# =============================================================================
class GlobalContextBlock(nn.Module):
    """GCNet-style global context. BN instead of LN for stability with small batches."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        self.context_weight: nn.Conv2d = nn.Conv2d(dim, 1, 1)

        hidden: int = max(dim // reduction, 8)
        self.transform: nn.Sequential = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x: Float[torch.Tensor, "b c h w"]) -> Float[torch.Tensor, "b c h w"]:
        """Add a learned global context vector to every spatial position.

        Args:
            x: Float tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float tensor with shape ``(batch, channels, height, width)``.
        """
        batch_size: int = x.shape[0]
        channels: int = x.shape[1]
        height: int = x.shape[2]
        width: int = x.shape[3]

        context_mask_b1n: Float[torch.Tensor, "b 1 spatial"] = self.context_weight(x).view(batch_size, 1, height * width)
        context_mask_b1n = F.softmax(context_mask_b1n, dim=-1)

        x_flat_bcn: Float[torch.Tensor, "b c spatial"] = x.view(batch_size, channels, height * width)
        context_bc11: Float[torch.Tensor, "b c 1 1"] = torch.bmm(x_flat_bcn, context_mask_b1n.transpose(1, 2)).unsqueeze(-1)
        transformed_context_bc11: Float[torch.Tensor, "b c 1 1"] = self.transform(context_bc11)

        out_bchw: Float[torch.Tensor, "b c h w"] = x + transformed_context_bc11
        return out_bchw


# =============================================================================
# MULTI-SCALE CONTEXT
# =============================================================================
class MinimalMultiScale(nn.Module):
    """Lightweight multi-scale context with 2 dilation rates."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.branch1: nn.Conv2d = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.branch2: nn.Conv2d = nn.Conv2d(dim, dim, 3, 1, 2, dilation=2, groups=dim, bias=False)
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(dim)

    def forward(self, x: Float[torch.Tensor, "b c h w"]) -> Float[torch.Tensor, "b c h w"]:
        """Add depthwise context from two dilation rates.

        Args:
            x: Float tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float tensor with shape ``(batch, channels, height, width)``.
        """
        context_bchw: Float[torch.Tensor, "b c h w"] = self.bn(self.branch1(x) + self.branch2(x))
        out_bchw: Float[torch.Tensor, "b c h w"] = x + context_bchw
        return out_bchw


# =============================================================================
# CROSS-SCALE
# =============================================================================
def _pick_groups(in_ch: int, out_ch: int, max_g: int = 4) -> int:
    """Choose the largest supported group count that divides both channel counts.

    Args:
        in_ch: Input channel count.
        out_ch: Output channel count.
        max_g: Largest group count to try.

    Returns:
        A group count that divides both channel counts.
    """
    for g in (max_g, 2, 1):
        if in_ch % g == 0 and out_ch % g == 0:
            return g
    return 1


class MinimalCrossScale(nn.Module):
    """Exchange projected features between adjacent encoder scales."""

    def __init__(self, dim_high: int, dim_low: int) -> None:
        super().__init__()
        g_h: int = _pick_groups(dim_low, dim_high, 4)
        g_l: int = _pick_groups(dim_high, dim_low, 4)

        self.low_to_high: nn.Conv2d = nn.Conv2d(dim_low, dim_high, 1, groups=g_h, bias=False)
        self.high_to_low: nn.Conv2d = nn.Conv2d(dim_high, dim_low, 1, groups=g_l, bias=False)

    def forward(
        self,
        x_high: Float[torch.Tensor, "b c_high h_high w_high"],
        x_low: Float[torch.Tensor, "b c_low h_low w_low"],
    ) -> tuple[
        Float[torch.Tensor, "b c_high h_high w_high"],
        Float[torch.Tensor, "b c_low h_low w_low"],
    ]:
        """Exchange context between high- and low-resolution feature maps.

        Args:
            x_high: Float tensor with shape ``(batch, high_channels, high_height, high_width)``.
            x_low: Float tensor with shape ``(batch, low_channels, low_height, low_width)``.

        Returns:
            A pair of float tensors with shapes ``(batch, high_channels, high_height, high_width)`` and
            ``(batch, low_channels, low_height, low_width)``.
        """
        high_size: tuple[int, int] = (x_high.shape[2], x_high.shape[3])
        low_size: tuple[int, int] = (x_low.shape[2], x_low.shape[3])
        low_up_bchw: Float[torch.Tensor, "b c_high h_high w_high"] = F.interpolate(self.low_to_high(x_low), size=high_size, mode="nearest")
        high_down_bchw: Float[torch.Tensor, "b c_low h_low w_low"] = F.adaptive_avg_pool2d(self.high_to_low(x_high), low_size)
        high_out_bchw: Float[torch.Tensor, "b c_high h_high w_high"] = x_high + low_up_bchw * 0.3
        low_out_bchw: Float[torch.Tensor, "b c_low h_low w_low"] = x_low + high_down_bchw * 0.3
        return high_out_bchw, low_out_bchw


# =============================================================================
# SPPF
# =============================================================================
class LightweightSPPF(nn.Module):
    """SPPF with reduced hidden channels for lightweight deployment."""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_hidden: int = c1 // 4
        self.cv1: ConvBN = ConvBN(c1, c_hidden, 1)
        self.cv2: ConvBN = ConvBN(c_hidden * 4, c2, 1)
        self.m: nn.MaxPool2d = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Float[torch.Tensor, "b c_in h w"]) -> Float[torch.Tensor, "b c_out h w"]:
        """Pool and concatenate one feature map at four receptive-field sizes.

        Args:
            x: Float tensor with shape ``(batch, input_channels, height, width)``.

        Returns:
            Float tensor with shape ``(batch, output_channels, height, width)``.
        """
        x_hidden_bchw: Float[torch.Tensor, "b c_hidden h w"] = self.cv1(x)
        pooled_1_bchw: Float[torch.Tensor, "b c_hidden h w"] = self.m(x_hidden_bchw)
        pooled_2_bchw: Float[torch.Tensor, "b c_hidden h w"] = self.m(pooled_1_bchw)
        pooled_3_bchw: Float[torch.Tensor, "b c_hidden h w"] = self.m(pooled_2_bchw)
        concatenated_bchw: Float[torch.Tensor, "b four_c_hidden h w"] = torch.cat((x_hidden_bchw, pooled_1_bchw, pooled_2_bchw, pooled_3_bchw), 1)
        out_bchw: Float[torch.Tensor, "b c_out h w"] = self.cv2(concatenated_bchw)
        return out_bchw


# =============================================================================
# DECODER FUSION
# =============================================================================
class UltraLightFusion(nn.Module):
    """Fuse adjacent decoder feature maps with grouped projections."""

    def __init__(self, high_ch: int, low_ch: int, out_ch: int) -> None:
        super().__init__()
        g_high: int = _pick_groups(high_ch, out_ch, 4)
        g_low: int = _pick_groups(low_ch, out_ch, 4)

        self.proj_high: nn.Conv2d = nn.Conv2d(high_ch, out_ch, 1, groups=g_high, bias=False)
        self.proj_low: nn.Conv2d = nn.Conv2d(low_ch, out_ch, 1, groups=g_low, bias=False)
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_ch)
        self.act: nn.ReLU = nn.ReLU(inplace=True)

    def forward(
        self,
        x_high: Float[torch.Tensor, "b c_high h_high w_high"],
        x_low: Float[torch.Tensor, "b c_low h_low w_low"],
    ) -> Float[torch.Tensor, "b c_out h_high w_high"]:
        """Project and sum a high-resolution map with an upsampled low-resolution map.

        Args:
            x_high: Float tensor with shape ``(batch, high_channels, high_height, high_width)``.
            x_low: Float tensor with shape ``(batch, low_channels, low_height, low_width)``.

        Returns:
            Float tensor with shape ``(batch, output_channels, high_height, high_width)``.
        """
        high_size: tuple[int, int] = (x_high.shape[2], x_high.shape[3])
        x_low_up_bchw: Float[torch.Tensor, "b c_low h_high w_high"] = F.interpolate(x_low, size=high_size, mode="bilinear", align_corners=False)
        fused_bchw: Float[torch.Tensor, "b c_out h_high w_high"] = self.proj_high(x_high) + self.proj_low(x_low_up_bchw)
        out_bchw: Float[torch.Tensor, "b c_out h_high w_high"] = self.act(self.bn(fused_bchw))
        return out_bchw


# =============================================================================
# FastConvexUpsample
# =============================================================================
class FastConvexUpsample(nn.Module):
    """Upsample depth with convex weights or an NPU-compatible blend."""

    def __init__(
        self,
        feat_ch: int,
        scale: int = 4,
        temperature: float = 1.0,
        use_unfold: bool = True,
    ) -> None:
        super().__init__()
        self.scale: int = scale
        self.temperature: float = temperature
        self.use_unfold: bool = use_unfold

        if use_unfold:
            # --- GPU / TensorRT path ---
            hidden: int = max(feat_ch // 4, 8)
            self.mask_pred: nn.Sequential = nn.Sequential(
                nn.Conv2d(feat_ch, hidden, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 9 * scale * scale, 1),
            )
        else:
            # --- NPU path ---
            where_hidden: int = max(feat_ch // 2, 8)
            self.where_conv: nn.Sequential = nn.Sequential(
                nn.Conv2d(feat_ch, where_hidden, 1, bias=False),
                nn.BatchNorm2d(where_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(where_hidden, where_hidden, 5, padding=2, groups=where_hidden, bias=False),
                nn.BatchNorm2d(where_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(where_hidden, 1, 1, bias=False),
            )

    def forward(
        self,
        feat: Float[torch.Tensor, "b c_feat h_low w_low"],
        depth: Float[torch.Tensor, "b 1 h_low w_low"],
    ) -> Float[torch.Tensor, "b 1 h_up w_up"]:
        """Upsample a low-resolution depth map.

        Args:
            feat: Float feature tensor with shape ``(batch, feature_channels, height, width)``.
            depth: Float depth tensor with shape ``(batch, 1, height, width)``.

        Returns:
            Nonnegative float depth tensor with shape ``(batch, 1, upsampled_height, upsampled_width)``.
        """
        if self.use_unfold:
            unfold_depth_b1hw: Float[torch.Tensor, "b 1 h_up w_up"] = self._forward_unfold(feat, depth)
            return F.relu(unfold_depth_b1hw)

        scale: int = self.scale

        depth_nn_b1hw: Float[torch.Tensor, "b 1 h_up w_up"] = F.interpolate(depth, scale_factor=scale, mode="nearest")
        depth_bilinear_b1hw: Float[torch.Tensor, "b 1 h_up w_up"] = F.interpolate(depth, scale_factor=scale, mode="bilinear", align_corners=False)

        alpha_b1h_low_w_low: Float[torch.Tensor, "b 1 h_low w_low"] = self.where_conv(feat)
        alpha_up_b1h_up_w_up: Float[torch.Tensor, "b 1 h_up w_up"] = F.interpolate(
            alpha_b1h_low_w_low, scale_factor=scale, mode="bilinear", align_corners=False
        )
        alpha_up_b1h_up_w_up = torch.sigmoid(alpha_up_b1h_up_w_up)

        out_b1hw: Float[torch.Tensor, "b 1 h_up w_up"] = alpha_up_b1h_up_w_up * depth_nn_b1hw + (1.0 - alpha_up_b1h_up_w_up) * depth_bilinear_b1hw

        return F.relu(out_b1hw)

    def _forward_unfold(
        self,
        feat: Float[torch.Tensor, "b c_feat h_low w_low"],
        depth: Float[torch.Tensor, "b 1 h_low w_low"],
    ) -> Float[torch.Tensor, "b 1 h_up w_up"]:
        """Upsample depth from learned convex neighborhood weights.

        Args:
            feat: Float feature tensor with shape
                ``(batch, feature_channels, low_height, low_width)``.
            depth: Float depth tensor with shape ``(batch, 1, low_height, low_width)``.

        Returns:
            Nonnegative float depth tensor with shape
            ``(batch, 1, upsampled_height, upsampled_width)``.
        """
        batch_size: int = depth.shape[0]
        height: int = depth.shape[2]
        width: int = depth.shape[3]
        scale: int = self.scale

        mask_raw_bchw: Float[torch.Tensor, "b mask_channels h w"] = self.mask_pred(feat)
        mask_b9sh_low_w_low: Float[torch.Tensor, "b 9 subpixels h_low w_low"] = mask_raw_bchw.view(batch_size, 9, scale * scale, height, width)
        mask_b9sh_low_w_low = F.softmax(mask_b9sh_low_w_low / self.temperature, dim=1)

        depth_pad_b1hw: Float[torch.Tensor, "b 1 h_pad w_pad"] = F.pad(depth, (1, 1, 1, 1), mode="replicate")
        neighbors_b91hw: Float[torch.Tensor, "b 9 1 h w"] = F.unfold(depth_pad_b1hw, 3).view(batch_size, 9, 1, height, width)

        up_bsh_low_w_low: Float[torch.Tensor, "b subpixels h_low w_low"] = (mask_b9sh_low_w_low * neighbors_b91hw).sum(1)
        up_b1h_up_w_up: Float[torch.Tensor, "b 1 h_up w_up"] = F.pixel_shuffle(up_bsh_low_w_low.view(batch_size, scale * scale, height, width), scale)

        return up_b1h_up_w_up


# =============================================================================
# Decoder
# =============================================================================
class ZipDepthDecoder(nn.Module):
    """Decode a ZipDepth feature pyramid into full-resolution depth."""

    def __init__(
        self,
        enc_dims: list[int],
        half_ch: int,
        dec_ch: int,
        half_dec_ch: int = 16,
        upsample_unfold: bool = True,
    ) -> None:
        super().__init__()
        c1: int = enc_dims[0]
        c2: int = enc_dims[1]
        c3: int = enc_dims[2]
        c4: int = enc_dims[3]
        ch4: int = dec_ch * 3
        ch3: int = dec_ch * 2
        ch2: int = int(dec_ch * 1.5)
        ch1: int = dec_ch
        self.proj4: ConvBN = ConvBN(c4, ch4, 1)
        self.fuse3: UltraLightFusion = UltraLightFusion(c3, ch4, ch3)
        self.fuse2: UltraLightFusion = UltraLightFusion(c2, ch3, ch2)
        self.fuse1: UltraLightFusion = UltraLightFusion(c1, ch2, ch1)

        ch_half: int = half_dec_ch
        self.fuse_half: UltraLightFusion = UltraLightFusion(high_ch=half_ch, low_ch=ch1, out_ch=ch_half)
        self.head_half: nn.Conv2d = nn.Conv2d(ch_half, 1, 3, padding=1)
        nn.init.kaiming_normal_(self.head_half.weight, mode="fan_out", nonlinearity="relu")
        if self.head_half.bias is not None:
            nn.init.constant_(self.head_half.bias, 0.5)
        self.convex_up: FastConvexUpsample = FastConvexUpsample(feat_ch=ch_half, scale=2, use_unfold=upsample_unfold)

    def forward(
        self,
        s_half: Float[torch.Tensor, "b c_half h_half w_half"],
        feats: FeaturePyramid,
    ) -> Float[torch.Tensor, "b 1 h_out w_out"]:
        """Decode multi-scale encoder features.

        Args:
            s_half: Float stem tensor with shape ``(batch, half_channels, half_height, half_width)``.
            feats: Four float tensors at quarter, eighth, sixteenth, and thirty-second resolution.

        Returns:
            Nonnegative float depth tensor with shape ``(batch, 1, output_height, output_width)``.
        """
        c1_bchw: Float[torch.Tensor, "b c1 h_quarter w_quarter"] = feats[0]
        c2_bchw: Float[torch.Tensor, "b c2 h_eighth w_eighth"] = feats[1]
        c3_bchw: Float[torch.Tensor, "b c3 h_sixteenth w_sixteenth"] = feats[2]
        c4_bchw: Float[torch.Tensor, "b c4 h_thirtysecond w_thirtysecond"] = feats[3]
        f4_bchw: Float[torch.Tensor, "b dec4 h_thirtysecond w_thirtysecond"] = self.proj4(c4_bchw)
        f3_bchw: Float[torch.Tensor, "b dec3 h_sixteenth w_sixteenth"] = self.fuse3(c3_bchw, f4_bchw)
        f2_bchw: Float[torch.Tensor, "b dec2 h_eighth w_eighth"] = self.fuse2(c2_bchw, f3_bchw)
        f1_bchw: Float[torch.Tensor, "b dec1 h_quarter w_quarter"] = self.fuse1(c1_bchw, f2_bchw)
        f_half_bchw: Float[torch.Tensor, "b dec_half h_half w_half"] = self.fuse_half(s_half, f1_bchw)
        depth_half_b1hw: Float[torch.Tensor, "b 1 h_half w_half"] = self.head_half(f_half_bchw)
        depth_b1hw: Float[torch.Tensor, "b 1 h_out w_out"] = self.convex_up(f_half_bchw, depth_half_b1hw)

        return depth_b1hw

    def fuse(self) -> None:
        """Do nothing; the decoder has no reparameterizable blocks and keeps this method for encoder symmetry."""
        pass


# =============================================================================
# Encoder
# =============================================================================
class ZipDepthEncoder(nn.Module):
    """Encode an RGB image into a four-level feature pyramid."""

    def __init__(
        self,
        in_ch: int,
        dims: list[int],
        depths: list[int],
        num_heads: int = 4,
        use_global: bool = True,
        global_mode: str = "balanced",
    ) -> None:
        super().__init__()
        self.use_global: bool = use_global
        self.global_mode: str = global_mode

        self.stem_half: ConvBN = ConvBN(in_ch, dims[0] // 2, k=3, s=2)  # -> H/2
        self.stem_quarter: ConvBN = ConvBN(dims[0] // 2, dims[0], k=3, s=2)  # -> H/4

        # Stage 1
        stage1_blocks: list[nn.Module] = [QARepBlock(dims[0], dims[0]) for _ in range(depths[0])]
        self.stage1: nn.Sequential = nn.Sequential(*stage1_blocks)

        # Stage 2
        self.down2: QARepBlock = QARepBlock(dims[0], dims[1], stride=2)
        stage2_blocks: list[nn.Module] = []
        for i in range(depths[1]):
            stage2_blocks.append(QARepBlock(dims[1], dims[1]))
            if i == depths[1] - 1:
                stage2_blocks.append(MinimalMultiScale(dims[1]))
                if use_global and global_mode in ["balanced", "full"]:
                    stage2_blocks.append(StripPoolingAttention(dims[1]))
        self.stage2: nn.Sequential = nn.Sequential(*stage2_blocks)

        # Stage 3
        self.down3: QARepBlock = QARepBlock(dims[1], dims[2], stride=2)
        stage3_blocks: list[nn.Module] = []
        for i in range(depths[2]):
            stage3_blocks.append(QARepBlock(dims[2], dims[2]))
            if i == depths[2] - 1:
                stage3_blocks.append(ChannelAttention(dims[2], reduction=8))
                if use_global:
                    stage3_blocks.append(GlobalContextBlock(dims[2]))
        self.stage3: nn.Sequential = nn.Sequential(*stage3_blocks)

        # Stage 4
        self.down4: QARepBlock = QARepBlock(dims[2], dims[3], stride=2)
        stage4_blocks: list[nn.Module] = [QARepBlock(dims[3], dims[3]) for _ in range(depths[3])]
        if use_global and global_mode == "full":
            stage4_blocks.append(EfficientGlobalAttention(dims[3], num_tokens=8, num_heads=num_heads))
        self.stage4: nn.Sequential = nn.Sequential(*stage4_blocks)

        # SPPF + Cross-scale
        self.spp: LightweightSPPF = LightweightSPPF(dims[3], dims[3])
        self.cross_scale: MinimalCrossScale = MinimalCrossScale(dims[2], dims[3])

    def forward(self, x: Float[torch.Tensor, "b c_in h w"]) -> EncoderOutput:
        """Build half-resolution stem features and a four-level pyramid.

        Args:
            x: Float tensor with shape ``(batch, input_channels, height, width)``.

        Returns:
            A half-resolution float tensor and four float tensors at quarter, eighth, sixteenth, and thirty-second resolution.
        """
        s_half_bchw: Float[torch.Tensor, "b c_half h_half w_half"] = self.stem_half(x)
        s_quarter_bchw: Float[torch.Tensor, "b c1 h_quarter w_quarter"] = self.stem_quarter(s_half_bchw)

        s1_bchw: Float[torch.Tensor, "b c1 h_quarter w_quarter"] = self.stage1(s_quarter_bchw)
        s2_bchw: Float[torch.Tensor, "b c2 h_eighth w_eighth"] = self.stage2(self.down2(s1_bchw))
        s3_bchw: Float[torch.Tensor, "b c3 h_sixteenth w_sixteenth"] = self.stage3(self.down3(s2_bchw))
        s4_bchw: Float[torch.Tensor, "b c4 h_thirtysecond w_thirtysecond"] = self.stage4(self.down4(s3_bchw))

        s4_bchw = self.spp(s4_bchw)
        cross_scale_output: tuple[
            Float[torch.Tensor, "b c3 h_sixteenth w_sixteenth"],
            Float[torch.Tensor, "b c4 h_thirtysecond w_thirtysecond"],
        ] = self.cross_scale(s3_bchw, s4_bchw)
        s3_bchw = cross_scale_output[0]
        s4_bchw = cross_scale_output[1]

        pyramid: FeaturePyramid = [s1_bchw, s2_bchw, s3_bchw, s4_bchw]
        return s_half_bchw, pyramid

    def fuse(self) -> None:
        """Fuse every reparameterizable encoder block in place."""
        for m in self.modules():
            if isinstance(m, QARepBlock):
                m.fuse()


# =============================================================================
# MAIN MODEL
# =============================================================================
class ZipDepth(nn.Module):
    """Estimate relative depth from normalized RGB images."""

    def __init__(
        self,
        variant: str = "base",
        global_mode: str = "balanced",
        pretrained: bool = False,
        upsample_unfold: bool = True,
    ) -> None:
        super().__init__()

        if variant not in MODEL_CONFIGS:
            raise ValueError(f"unknown ZipDepth variant {variant!r}; expected one of {sorted(MODEL_CONFIGS)}")
        cfg: ModelConfig = MODEL_CONFIGS[variant]
        self.variant: str = variant
        self.global_mode: str = global_mode

        use_global: bool = cfg["use_global"] and global_mode != "none"

        self.encoder: ZipDepthEncoder = ZipDepthEncoder(
            in_ch=3,
            dims=cfg["dims"],
            depths=cfg["depths"],
            num_heads=cfg["heads"],
            use_global=use_global,
            global_mode=global_mode,
        )

        self.decoder: ZipDepthDecoder = ZipDepthDecoder(
            enc_dims=cfg["dims"],
            half_ch=cfg["dims"][0] // 2,
            dec_ch=cfg["dec_ch"],
            half_dec_ch=cfg["half_dec_ch"],
            upsample_unfold=upsample_unfold,
        )

        mean_1311: Float[torch.Tensor, "1 3 1 1"] = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std_1311: Float[torch.Tensor, "1 3 1 1"] = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.mean: Float[torch.Tensor, "1 3 1 1"]
        self.std: Float[torch.Tensor, "1 3 1 1"]
        self.register_buffer("mean", mean_1311)
        self.register_buffer("std", std_1311)

        self.apply(self._init_weights)

        if pretrained:
            print("[Warning] Pretrained weights not available yet.")

    # ------------------------------------------------------------------
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize one child module with the model's original scheme.

        Args:
            m: Child module to initialize.

        Returns:
            None.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: Float[torch.Tensor, "b 3 h w"]) -> Float[torch.Tensor, "b 1 h_out w_out"]:
        """Predict nonnegative relative depth.

        Args:
            x: Float RGB tensor with shape ``(batch, 3, height, width)``.

        Returns:
            Float depth tensor with shape ``(batch, 1, output_height, output_width)``.
        """
        x_norm_bchw: Float[torch.Tensor, "b 3 h w"] = (x - self.mean) / self.std

        encoder_output: EncoderOutput = self.encoder(x_norm_bchw)
        s_half_bchw: Float[torch.Tensor, "b c_half h_half w_half"] = encoder_output[0]
        enc_feats: FeaturePyramid = encoder_output[1]

        depth_b1hw: Float[torch.Tensor, "b 1 h_out w_out"] = self.decoder(s_half_bchw, enc_feats)
        return depth_b1hw

    # ------------------------------------------------------------------
    def fuse_for_inference(self) -> Self:
        """Fuse reparameterizable blocks and switch the model to evaluation mode.

        Returns:
            This model after in-place fusion.
        """
        self.eval()
        self.encoder.fuse()
        self.decoder.fuse()
        return self

    # ------------------------------------------------------------------
    def get_model_info(self) -> ModelInfo:
        """Return model configuration and parameter-count information."""
        cfg: ModelConfig = MODEL_CONFIGS[self.variant]
        return {
            "variant": self.variant,
            "dims": cfg["dims"],
            "depths": cfg["depths"],
            "dec_ch": cfg["dec_ch"],
            "half_dec_ch": cfg["half_dec_ch"],
            "parameters_M": count_parameters(self),
            "global_mode": self.global_mode,
        }

    def print_model_summary(self) -> None:
        """Print a compact model summary to standard output."""
        info: ModelInfo = self.get_model_info()
        print(f"\n{'=' * 60}")
        print(f"ZipDepth-{self.variant.upper()}")
        print(f"{'=' * 60}")
        print(f"Parameters:  {info['parameters_M']:.2f}M")
        print(f"Dims:        {info['dims']}")
        print(f"Depths:      {info['depths']}")
        print(f"Decoder Ch:  {info['dec_ch']}")
        print(f"Global Mode: {info['global_mode']}")
        print(f"{'=' * 60}\n")


# =============================================================================
# API
# =============================================================================


def create_model(
    variant: str = "base",
    *,
    global_mode: str = "balanced",
    pretrained: bool = False,
    upsample_unfold: bool = True,
) -> ZipDepth:
    """Create a ZipDepth model.

    Args:
        variant: Model-size name. Hyphen and known prefix spelling variants are normalized.
        global_mode: Global-attention mode passed to the encoder.
        pretrained: Whether to request pretrained weights. The vendored model only emits a warning when true.
        upsample_unfold: Whether to use the unfold-based convex upsampling path.

    Returns:
        A newly initialized ZipDepth model.
    """
    variant = variant.lower().replace("-", "_").replace("zip_", "").replace("depth_", "")
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"unknown ZipDepth variant {variant!r}; expected one of {sorted(MODEL_CONFIGS)}")
    model: ZipDepth = ZipDepth(
        variant=variant,
        global_mode=global_mode,
        pretrained=pretrained,
        upsample_unfold=upsample_unfold,
    )
    return model
