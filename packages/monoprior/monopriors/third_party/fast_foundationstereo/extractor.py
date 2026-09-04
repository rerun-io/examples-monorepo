"""EdgeNeXt feature and recurrent-context extractors for Fast-FoundationStereo."""

from collections.abc import Sequence
from typing import Any, Literal, NotRequired, Protocol, TypeAlias, TypedDict, cast, runtime_checkable

import timm
from jaxtyping import Float
from torch import Tensor, nn

from monopriors.third_party.fast_foundationstereo.submodule import Conv2x_IN

FeaturePyramid: TypeAlias = list[Float[Tensor, "b _channels _height _width"]]
ContextNetOutput: TypeAlias = tuple[list[Float[Tensor, "b hidden h w"]]]


class FastFoundationStereoConfig(TypedDict):
    """Configuration fields used by the Fast-FoundationStereo inference architecture."""

    corr_levels: int
    corr_radius: int
    hidden_dims: list[int]
    low_memory: int
    max_disp: int
    mixed_precision: bool
    n_downsample: int
    n_gru_layers: int
    slow_fast_gru: bool
    valid_iters: int
    vit_size: Literal["vitl", "vitb", "vits"]
    normalize: bool
    image_size: NotRequired[list[int]]
    cv_group: NotRequired[int]
    volume_dim: NotRequired[int]


@runtime_checkable
class ConfigMapping(Protocol):
    """Structural interface shared by plain typed dictionaries and OmegaConf mappings."""

    def __getitem__(self, key: str) -> Any:
        """Return one configuration value."""
        ...

    def get(self, key: str, default: object = None) -> Any:
        """Return one optional configuration value."""
        ...


ConfigLike: TypeAlias = FastFoundationStereoConfig | ConfigMapping


class ContextNetSharedBackbone(nn.Module):
    """Project quarter-resolution image features into recurrent hidden and input states."""

    def __init__(
        self,
        args: ConfigLike,
        c04: int,
        c08: int,
        c16: int,
        output_dim: Sequence[Sequence[int]] = ((128, 128, 128), (128, 128, 128)),
        norm_fn: str = "batch",
        downsample: int = 3,
    ) -> None:
        """Initialize the context projections.

        Args:
            args: Model configuration retained for checkpoint compatibility.
            c04: Quarter-resolution input channel count.
            c08: Eighth-resolution channel count retained by the upstream signature.
            c16: Sixteenth-resolution channel count retained by the upstream signature.
            output_dim: Hidden and input channel pyramids.
            norm_fn: Upstream normalization selector retained by the constructor contract.
            downsample: Upstream context depth retained by the constructor contract.
        """
        super().__init__()
        self.args: ConfigLike = args
        self.conv04: nn.ModuleList = nn.ModuleList(
            [
                nn.Conv2d(c04, output_dim[0][0], kernel_size=3, padding=1),
                nn.Conv2d(c04, output_dim[1][0], kernel_size=3, padding=1),
            ]
        )

    def forward(
        self,
        x4_bchw: Float[Tensor, "b channels h4 w4"],
        x8_bchw: Float[Tensor, "b channels8 h8 w8"],
        x16_bchw: Float[Tensor, "b channels16 h16 w16"],
    ) -> ContextNetOutput:
        """Create the one-level recurrent context used by released inference models.

        Args:
            x4_bchw: Floating-point quarter-resolution features with shape ``(batch, channels, height, width)``.
            x8_bchw: Floating-point eighth-resolution features retained by the upstream interface.
            x16_bchw: Floating-point sixteenth-resolution features retained by the upstream interface.

        Returns:
            One list containing floating-point hidden and input states with shape ``(batch, hidden, height, width)``.
        """
        outputs04: list[Float[Tensor, "b hidden h4 w4"]] = [conv(x4_bchw) for conv in self.conv04]
        return (outputs04,)


class DepthAnythingModelConfig(TypedDict):
    """DepthAnything feature dimensions retained by the released model selector."""

    encoder: str
    features: int
    out_channels: list[int]


class DepthAnythingFeature:
    """Static feature dimensions selected by ``vit_size`` in the upstream configuration."""

    model_configs: dict[str, DepthAnythingModelConfig] = {
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    }


class Feature(nn.Module):
    """Extract and fuse the four EdgeNeXt feature levels used by stereo inference."""

    def __init__(self, args: ConfigLike) -> None:
        """Initialize the EdgeNeXt feature pyramid.

        Args:
            args: Fast-FoundationStereo inference configuration.
        """
        super().__init__()
        self.args: ConfigLike = args
        model: Any = timm.create_model("edgenext_small", pretrained=True, features_only=False)
        self.stem: nn.Sequential = cast(nn.Sequential, model.stem)
        self.stages: nn.Sequential = cast(nn.Sequential, model.stages)
        self.chans: list[int] = [48, 96, 160, 304]
        vit_size: str = str(args["vit_size"])
        vit_feat_dim: int = DepthAnythingFeature.model_configs[vit_size]["features"] // 2

        self.deconv32_16: Conv2x_IN = Conv2x_IN(self.chans[3], self.chans[2], deconv=True, concat=True)
        self.deconv16_8: Conv2x_IN = Conv2x_IN(self.chans[2] * 2, self.chans[1], deconv=True, concat=True)
        self.deconv8_4: Conv2x_IN = Conv2x_IN(self.chans[1] * 2, self.chans[0], deconv=True, concat=True)
        self.conv4: nn.Conv2d = nn.Conv2d(self.chans[0] * 2, self.chans[0] * 2 + vit_feat_dim, kernel_size=1, stride=1, padding=0)
        self.d_out: list[int] = [self.chans[0] * 2 + vit_feat_dim, self.chans[1] * 2, self.chans[2] * 2, self.chans[3]]

    def forward(self, x_b3hw: Float[Tensor, "b 3 h w"]) -> FeaturePyramid:
        """Extract a four-level floating-point feature pyramid.

        Args:
            x_b3hw: Normalized RGB tensor with shape ``(batch, 3, height, width)``.

        Returns:
            Four floating-point tensors from quarter through thirty-second resolution.
        """
        stem_bchw: Float[Tensor, "b stem_channels h2 w2"] = self.stem(x_b3hw)
        x4_bchw: Float[Tensor, "b stage4_channels h4 w4"] = self.stages[0](stem_bchw)
        x8_bchw: Float[Tensor, "b stage8_channels h8 w8"] = self.stages[1](x4_bchw)
        x16_bchw: Float[Tensor, "b stage16_channels h16 w16"] = self.stages[2](x8_bchw)
        x32_bchw: Float[Tensor, "b stage32_channels h32 w32"] = self.stages[3](x16_bchw)

        fused16_bchw: Float[Tensor, "b fused16_channels h16 w16"] = self.deconv32_16(x32_bchw, x16_bchw)
        fused8_bchw: Float[Tensor, "b fused8_channels h8 w8"] = self.deconv16_8(fused16_bchw, x8_bchw)
        fused4_bchw: Float[Tensor, "b fused4_channels h4 w4"] = self.deconv8_4(fused8_bchw, x4_bchw)
        output4_bchw: Float[Tensor, "b output_channels h4 w4"] = self.conv4(fused4_bchw)
        return [output4_bchw, fused8_bchw, fused16_bchw, x32_bchw]
