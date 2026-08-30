"""FasterNet cost-volume aggregation layers for LiteAnyStereo V2."""

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn.functional as F
from jaxtyping import Float32
from torch import Tensor, nn

ActivationFactory: TypeAlias = Callable[[], nn.Module]
FeaturePyramid: TypeAlias = list[Float32[Tensor, "b _channels _height _width"]]


class Aggregation(nn.Module):
    """Aggregate a disparity cost volume with a FasterNet encoder-decoder."""

    def __init__(self, in_channels: int, left_att: bool, blocks: list[int], expanse_ratio: int, backbone_channels: list[int]) -> None:
        """Initialize the aggregation network.

        Args:
            in_channels: Number of disparity channels in the input cost volume.
            left_att: Whether to modulate cost features with left-image features.
            blocks: Residual-block counts for the three encoder resolutions.
            expanse_ratio: FasterNet MLP expansion ratio.
            backbone_channels: Channel counts for the image feature pyramid.
        """
        super().__init__()
        self.left_att: bool = left_att
        self.expanse_ratio: int = expanse_ratio

        conv0: list[nn.Module] = [
            FasterNetResidual(in_channels, in_channels, stride=1, mlp_ratio=self.expanse_ratio) for _ in range(blocks[0])
        ]
        self.conv0: nn.Sequential = nn.Sequential(*conv0)

        self.conv1: FasterNetResidual = FasterNetResidual(in_channels, in_channels * 2, stride=2, mlp_ratio=self.expanse_ratio)
        conv2_add: list[nn.Module] = [
            FasterNetResidual(in_channels * 2, in_channels * 2, stride=1, mlp_ratio=self.expanse_ratio) for _ in range(blocks[1] - 1)
        ]
        self.conv2: nn.Sequential = nn.Sequential(*conv2_add)

        self.conv3: FasterNetResidual = FasterNetResidual(in_channels * 2, in_channels * 4, stride=2, mlp_ratio=self.expanse_ratio)
        conv4_add: list[nn.Module] = [
            FasterNetResidual(in_channels * 4, in_channels * 4, stride=1, mlp_ratio=self.expanse_ratio) for _ in range(blocks[2] - 1)
        ]
        self.conv4: nn.Sequential = nn.Sequential(*conv4_add)

        self.conv5: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2),
        )
        self.conv6: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.redir1: FasterNetResidual = FasterNetResidual(in_channels, in_channels, stride=1, mlp_ratio=self.expanse_ratio)
        self.redir2: FasterNetResidual = FasterNetResidual(in_channels * 2, in_channels * 2, stride=1, mlp_ratio=self.expanse_ratio)

        self.att0: AttentionModule | None = AttentionModule(in_channels, backbone_channels[0]) if self.left_att else None
        self.att2: AttentionModule | None = AttentionModule(in_channels * 2, backbone_channels[1]) if self.left_att else None
        self.att4: AttentionModule | None = AttentionModule(in_channels * 4, backbone_channels[2]) if self.left_att else None

    def forward(
        self,
        cost_bdhw: Float32[Tensor, "b disparities h w"],
        features_left: FeaturePyramid,
    ) -> Float32[Tensor, "b disparities h w"]:
        """Aggregate a float32 cost volume.

        Args:
            cost_bdhw: Float32 cost volume with shape ``(batch, disparities, height, width)``.
            features_left: Float32 image features; each item has shape ``(batch, channels, height, width)``.

        Returns:
            Float32 aggregated cost volume with shape ``(batch, disparities, height, width)``.
        """
        conv0_bdhw: Float32[Tensor, "b disparities h w"] = self.conv0(cost_bdhw)
        if self.left_att:
            assert self.att0 is not None
            conv0_bdhw = self.att0(conv0_bdhw, features_left[0])

        conv1_bdhw: Float32[Tensor, "b disparities_2 h2 w2"] = self.conv1(conv0_bdhw)
        conv2_bdhw: Float32[Tensor, "b disparities_2 h2 w2"] = self.conv2(conv1_bdhw)
        if self.left_att:
            assert self.att2 is not None
            conv2_bdhw = self.att2(conv2_bdhw, features_left[1])

        conv3_bdhw: Float32[Tensor, "b disparities_4 h4 w4"] = self.conv3(conv2_bdhw)
        conv4_bdhw: Float32[Tensor, "b disparities_4 h4 w4"] = self.conv4(conv3_bdhw)
        if self.left_att:
            assert self.att4 is not None
            conv4_bdhw = self.att4(conv4_bdhw, features_left[2])

        conv5_bdhw: Float32[Tensor, "b disparities_2 h2 w2"] = F.relu(self.conv5(conv4_bdhw) + self.redir2(conv2_bdhw), inplace=True)
        conv6_bdhw: Float32[Tensor, "b disparities h w"] = F.relu(self.conv6(conv5_bdhw) + self.redir1(conv0_bdhw), inplace=True)
        return conv6_bdhw


class FasterNetResidual(nn.Module):
    """Apply FasterNet spatial mixing and an MLP after an optional projection."""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        mlp_ratio: int,
        n_div: int = 4,
        act_layer: ActivationFactory = nn.GELU,
    ) -> None:
        """Initialize one residual block.

        Args:
            inp: Number of input channels.
            oup: Number of output channels.
            stride: Spatial stride, either 1 or 2.
            mlp_ratio: MLP expansion ratio.
            n_div: Fractional divisor for partial spatial convolution.
            act_layer: Activation-layer factory.
        """
        super().__init__()
        assert stride in (1, 2)

        if stride == 1 and inp == oup:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))

        self.block: FasterNetBlock = FasterNetBlock(oup, mlp_ratio=mlp_ratio, n_div=n_div, act_layer=act_layer)

    def forward(self, x_bchw: Float32[Tensor, "b c_in h w"]) -> Float32[Tensor, "b c_out h_out w_out"]:
        """Transform one float32 feature map.

        Args:
            x_bchw: Float32 tensor with shape ``(batch, input_channels, height, width)``.

        Returns:
            Float32 tensor with shape ``(batch, output_channels, output_height, output_width)``.
        """
        projected_bchw: Float32[Tensor, "b c_out h_out w_out"] = self.proj(x_bchw)
        output_bchw: Float32[Tensor, "b c_out h_out w_out"] = self.block(projected_bchw)
        return output_bchw


class FasterNetBlock(nn.Module):
    """Apply partial spatial convolution and a pointwise MLP with a residual."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        n_div: int = 4,
        act_layer: ActivationFactory = nn.GELU,
        layer_scale_init_value: float = 0.0,
    ) -> None:
        """Initialize one FasterNet block.

        Args:
            dim: Input and output channel count.
            mlp_ratio: MLP expansion ratio.
            n_div: Fractional divisor for partial spatial convolution.
            act_layer: Activation-layer factory.
            layer_scale_init_value: Optional initial residual-branch scale.
        """
        super().__init__()
        hidden_dim: int = int(dim * mlp_ratio)

        self.spatial_mixing: PartialConv3 = PartialConv3(dim, n_div)
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_layer(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
        )

        if layer_scale_init_value > 0.0:
            self.layer_scale: Float32[Tensor, "channels"] | None = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True
            )
        else:
            self.layer_scale = None

    def forward(self, x_bchw: Float32[Tensor, "b channels h w"]) -> Float32[Tensor, "b channels h w"]:
        """Transform one float32 feature map.

        Args:
            x_bchw: Float32 tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float32 tensor with shape ``(batch, channels, height, width)``.
        """
        shortcut_bchw: Float32[Tensor, "b channels h w"] = x_bchw
        mixed_bchw: Float32[Tensor, "b channels h w"] = self.spatial_mixing(x_bchw)
        residual_bchw: Float32[Tensor, "b channels h w"] = self.mlp(mixed_bchw)
        if self.layer_scale is not None:
            residual_bchw = self.layer_scale.view(1, -1, 1, 1) * residual_bchw
        output_bchw: Float32[Tensor, "b channels h w"] = shortcut_bchw + residual_bchw
        return output_bchw


class PartialConv3(nn.Module):
    """Apply a 3-by-3 convolution to one channel partition."""

    def __init__(self, dim: int, n_div: int = 4) -> None:
        """Initialize the partial convolution.

        Args:
            dim: Total channel count.
            n_div: Divisor that selects the convolved channel count.
        """
        super().__init__()
        self.dim_conv3: int = dim // n_div
        self.dim_untouched: int = dim - self.dim_conv3
        self.partial_conv3: nn.Conv2d = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x_bchw: Float32[Tensor, "b channels h w"]) -> Float32[Tensor, "b channels h w"]:
        """Transform one float32 feature map.

        Args:
            x_bchw: Float32 tensor with shape ``(batch, channels, height, width)``.

        Returns:
            Float32 tensor with shape ``(batch, channels, height, width)``.
        """
        split_features: tuple[Float32[Tensor, "b _channels h w"], ...] = torch.split(
            x_bchw, [self.dim_conv3, self.dim_untouched], dim=1
        )
        convolved_bchw: Float32[Tensor, "b convolved_channels h w"] = self.partial_conv3(split_features[0])
        untouched_bchw: Float32[Tensor, "b untouched_channels h w"] = split_features[1]
        output_bchw: Float32[Tensor, "b channels h w"] = torch.cat((convolved_bchw, untouched_bchw), dim=1)
        return output_bchw


class AttentionModule(nn.Module):
    """Modulate cost features with multi-scale spatial attention from image features."""

    def __init__(self, dim: int, img_feat_dim: int) -> None:
        """Initialize the attention module.

        Args:
            dim: Cost-volume channel count.
            img_feat_dim: Image-feature channel count.
        """
        super().__init__()
        self.conv0: nn.Conv2d = nn.Conv2d(img_feat_dim, dim, 1)
        self.conv0_1: nn.Conv2d = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2: nn.Conv2d = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1: nn.Conv2d = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2: nn.Conv2d = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1: nn.Conv2d = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2: nn.Conv2d = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3: nn.Conv2d = nn.Conv2d(dim, dim, 1)

    def forward(
        self,
        cost_bchw: Float32[Tensor, "b channels h w"],
        image_features_bchw: Float32[Tensor, "b image_channels h w"],
    ) -> Float32[Tensor, "b channels h w"]:
        """Apply float32 image-conditioned spatial attention.

        Args:
            cost_bchw: Float32 cost features with shape ``(batch, channels, height, width)``.
            image_features_bchw: Float32 image features with shape ``(batch, image_channels, height, width)``.

        Returns:
            Float32 modulated cost features with shape ``(batch, channels, height, width)``.
        """
        attention_bchw: Float32[Tensor, "b channels h w"] = self.conv0(image_features_bchw)
        attention0_bchw: Float32[Tensor, "b channels h w"] = self.conv0_2(self.conv0_1(attention_bchw))
        attention1_bchw: Float32[Tensor, "b channels h w"] = self.conv1_2(self.conv1_1(attention_bchw))
        attention2_bchw: Float32[Tensor, "b channels h w"] = self.conv2_2(self.conv2_1(attention_bchw))
        combined_attention_bchw: Float32[Tensor, "b channels h w"] = attention_bchw + attention0_bchw + attention1_bchw + attention2_bchw
        projected_attention_bchw: Float32[Tensor, "b channels h w"] = self.conv3(combined_attention_bchw)
        output_bchw: Float32[Tensor, "b channels h w"] = projected_attention_bchw * cost_bchw
        return output_bchw
