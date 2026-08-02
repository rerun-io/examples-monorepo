import functools
import itertools
from collections.abc import Callable, Iterator, Sequence
from typing import Literal, TypeAlias, cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

from monopriors.third_party.dinov2 import dinov2_vitb14, dinov2_vitl14, dinov2_vits14
from monopriors.third_party.dinov2.vision_transformer import DinoVisionTransformer

ActivationMode: TypeAlias = Literal["relu", "leaky_relu", "silu", "elu"]
NormalizationMode: TypeAlias = Literal["group_norm", "layer_norm", "instance_norm", "none"]
ResamplerMode: TypeAlias = Literal["bilinear", "conv_transpose"]
EncoderLayer: TypeAlias = tuple[Float[Tensor, "b n c"], Float[Tensor, "b c"]]

DINOV2_FACTORIES: dict[str, Callable[..., DinoVisionTransformer]] = {
    "dinov2_vits14": dinov2_vits14,
    "dinov2_vitb14": dinov2_vitb14,
    "dinov2_vitl14": dinov2_vitl14,
}


class ResidualConvBlock(nn.Module):
    """Two-convolution residual block used by the MoGe v2 decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        activation: ActivationMode = "relu",
        in_norm: NormalizationMode = "layer_norm",
        hidden_norm: NormalizationMode = "group_norm",
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        activation_factory: Callable[[], nn.Module]
        if activation == "relu":
            activation_factory = nn.ReLU
        elif activation == "leaky_relu":
            activation_factory = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        elif activation == "silu":
            activation_factory = nn.SiLU
        elif activation == "elu":
            activation_factory = nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        input_norm: nn.Module
        if in_norm == "group_norm":
            input_norm = nn.GroupNorm(in_channels // 32, in_channels)
        elif in_norm == "layer_norm":
            input_norm = nn.GroupNorm(1, in_channels)
        elif in_norm == "instance_norm":
            input_norm = nn.InstanceNorm2d(in_channels)
        else:
            input_norm = nn.Identity()

        hidden_normalization: nn.Module
        if hidden_norm == "group_norm":
            hidden_normalization = nn.GroupNorm(hidden_channels // 32, hidden_channels)
        elif hidden_norm == "layer_norm":
            hidden_normalization = nn.GroupNorm(1, hidden_channels)
        elif hidden_norm == "instance_norm":
            hidden_normalization = nn.InstanceNorm2d(hidden_channels)
        else:
            hidden_normalization = nn.Identity()

        self.layers: nn.Sequential = nn.Sequential(
            input_norm,
            activation_factory(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode),
            hidden_normalization,
            activation_factory(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode),
        )

        self.skip_connection: nn.Module = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x_bchw: Float[Tensor, "b c h w"]) -> Float[Tensor, "b out_c h w"]:
        """Apply the residual block.

        Args:
            x_bchw: Float feature tensor shaped ``b c h w``.

        Returns:
            Float feature tensor shaped ``b out_c h w``.
        """
        skip_bchw: Float[Tensor, "b out_c h w"] = self.skip_connection(x_bchw)
        output_bchw: Float[Tensor, "b out_c h w"] = self.layers(x_bchw)
        output_bchw = output_bchw + skip_bchw
        return output_bchw


class DINOv2Encoder(nn.Module):
    """DINOv2 encoder wrapper used by MoGe v2."""

    backbone: DinoVisionTransformer
    image_mean: Float[Tensor, "1 3 1 1"]
    image_std: Float[Tensor, "1 3 1 1"]
    dim_features: int

    def __init__(self, backbone: str, intermediate_layers: int | Sequence[int], dim_out: int) -> None:
        super().__init__()

        self.intermediate_layers: int | tuple[int, ...] = intermediate_layers if isinstance(intermediate_layers, int) else tuple(intermediate_layers)
        self.backbone = DINOV2_FACTORIES[backbone](pretrained=False)
        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features: int = self.intermediate_layers if isinstance(self.intermediate_layers, int) else len(self.intermediate_layers)
        self.output_projections: nn.ModuleList = nn.ModuleList(
            [nn.Conv2d(in_channels=self.dim_features, out_channels=dim_out, kernel_size=1, stride=1, padding=0) for _ in range(self.num_features)]
        )

        image_mean_1311: Float[Tensor, "1 3 1 1"] = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std_1311: Float[Tensor, "1 3 1 1"] = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean_1311)
        self.register_buffer("image_std", image_std_1311)

    @property
    def onnx_compatible_mode(self) -> bool:
        """Whether ONNX-safe interpolation is enabled."""
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool) -> None:
        self._onnx_compatible_mode: bool = value
        self.backbone.onnx_compatible_mode = value

    def forward(
        self,
        image_b3hw: Float[Tensor, "b 3 h w"],
        token_rows: int | Int[Tensor, ""],
        token_cols: int | Int[Tensor, ""],
        return_class_token: bool = False,
    ) -> Float[Tensor, "b c token_h token_w"] | tuple[Float[Tensor, "b c token_h token_w"], Float[Tensor, "b c"]]:
        """Encode an image at the requested token-grid size.

        Args:
            image_b3hw: Float RGB image tensor shaped ``b 3 h w``.
            token_rows: Number of patch-token rows as an integer or scalar integer tensor.
            token_cols: Number of patch-token columns as an integer or scalar integer tensor.
            return_class_token: Whether to return the final class token with the feature map.

        Returns:
            Float feature map shaped ``b c token_h token_w``, optionally paired with a float class-token tensor shaped ``b c``.
        """
        image_14_b3hw: Float[Tensor, "b 3 token_h token_w"] = F.interpolate(
            image_b3hw,
            (token_rows * 14, token_cols * 14),
            mode="bilinear",
            align_corners=False,
            antialias=not self.onnx_compatible_mode,
        )
        image_14_b3hw = (image_14_b3hw - self.image_mean) / self.image_std

        raw_features: object = self.backbone.get_intermediate_layers(
            image_14_b3hw,
            n=self.intermediate_layers,
            return_class_token=True,
        )
        features: tuple[EncoderLayer, ...] = cast(tuple[EncoderLayer, ...], raw_features)
        projected_features: list[Float[Tensor, "b c token_h token_w"]] = [
            projection(feature_bnc.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous())
            for projection, (feature_bnc, _class_token_bc) in zip(self.output_projections, features, strict=True)
        ]
        output_bchw: Float[Tensor, "b c token_h token_w"] = torch.stack(projected_features, dim=1).sum(dim=1)

        if return_class_token:
            class_token_bc: Float[Tensor, "b c"] = features[-1][1]
            return output_bchw, class_token_bc
        return output_bchw


class Resampler(nn.Sequential):
    """Upsample one feature-pyramid level and change its channel count."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        type_: ResamplerMode,
        scale_factor: int = 2,
    ) -> None:
        if type_ == "bilinear":
            nn.Sequential.__init__(
                self,
                nn.Upsample(scale_factor=scale_factor, mode=type_, align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            )
        elif type_ == "conv_transpose":
            nn.Sequential.__init__(
                self,
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            )
            self[0].weight.data[:] = self[0].weight.data[:, :, :1, :1]
        else:
            raise ValueError(f"Unsupported resampler type: {type_}")


class MLP(nn.Sequential):
    """ReLU MLP used for the MoGe metric-scale head."""

    def __init__(self, dims: Sequence[int]) -> None:
        nn.Sequential.__init__(
            self,
            *itertools.chain(*[(nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)) for dim_in, dim_out in zip(dims[:-2], dims[1:-1], strict=True)]),
            nn.Linear(dims[-2], dims[-1]),
        )


class ConvStack(nn.Module):
    """Decode a five-level MoGe feature pyramid."""

    def __init__(
        self,
        dim_in: Sequence[int | None],
        dim_res_blocks: Sequence[int],
        dim_out: Sequence[int | None] | None,
        resamplers: ResamplerMode | Sequence[ResamplerMode],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int | Sequence[int] = 1,
        res_block_in_norm: NormalizationMode = "layer_norm",
        res_block_hidden_norm: NormalizationMode = "group_norm",
        activation: ActivationMode = "relu",
    ) -> None:
        super().__init__()
        self.input_blocks: nn.ModuleList = nn.ModuleList(
            [
                nn.Conv2d(dim_in_level, dim_res_block, kernel_size=1, stride=1, padding=0) if dim_in_level is not None else nn.Identity()
                for dim_in_level, dim_res_block in zip(dim_in, dim_res_blocks, strict=True)
            ]
        )
        resampler_values: Sequence[ResamplerMode] | Iterator[ResamplerMode] = (
            resamplers if isinstance(resamplers, Sequence) and not isinstance(resamplers, str) else itertools.repeat(resamplers)
        )
        self.resamplers: nn.ModuleList = nn.ModuleList(
            [
                Resampler(dim_previous, dim_successor, scale_factor=2, type_=resampler)
                for dim_previous, dim_successor, resampler in zip(
                    dim_res_blocks[:-1],
                    dim_res_blocks[1:],
                    resampler_values,
                    strict=False,
                )
            ]
        )
        self.res_blocks: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        ResidualConvBlock(
                            dim_res_block,
                            dim_res_block,
                            dim_times_res_block_hidden * dim_res_block,
                            activation=activation,
                            in_norm=res_block_in_norm,
                            hidden_norm=res_block_hidden_norm,
                        )
                        for _ in range(num_res_blocks[level] if isinstance(num_res_blocks, Sequence) else num_res_blocks)
                    )
                )
                for level, dim_res_block in enumerate(dim_res_blocks)
            ]
        )
        output_dimensions: Sequence[int | None] | Iterator[None] = dim_out if dim_out is not None else itertools.repeat(None)
        self.output_blocks: nn.ModuleList = nn.ModuleList(
            [
                nn.Conv2d(dim_res_block, dim_out_level, kernel_size=1, stride=1, padding=0) if dim_out_level is not None else nn.Identity()
                for dim_out_level, dim_res_block in zip(output_dimensions, dim_res_blocks, strict=False)
            ]
        )

    def forward(self, in_features: Sequence[Float[Tensor, "b c h w"]]) -> list[Float[Tensor, "b c h w"]]:
        """Decode a feature pyramid.

        Args:
            in_features: Float feature tensors shaped ``b c h w``, ordered from coarse to fine.

        Returns:
            Float decoded feature tensors shaped ``b c h w`` at each level.
        """
        out_features: list[Float[Tensor, "b c h w"]] = []
        x_bchw: Float[Tensor, "b c h w"]
        for level in range(len(self.res_blocks)):
            feature_bchw: Float[Tensor, "b c h w"] = self.input_blocks[level](in_features[level])
            x_bchw = feature_bchw if level == 0 else x_bchw + feature_bchw
            x_bchw = self.res_blocks[level](x_bchw)
            output_bchw: Float[Tensor, "b c h w"] = self.output_blocks[level](x_bchw)
            out_features.append(output_bchw)
            if level < len(self.res_blocks) - 1:
                x_bchw = self.resamplers[level](x_bchw)
        return out_features
