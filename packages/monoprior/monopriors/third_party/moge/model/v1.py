import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self, TypeAlias, cast

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from jaxtyping import Bool, Float
from torch import Tensor, nn

from monopriors.third_party.dinov2 import dinov2_vitl14
from monopriors.third_party.dinov2.vision_transformer import DinoVisionTransformer
from monopriors.third_party.moge.model._inference import (
    CameraRecovery,
    InferenceInput,
    InferenceOutput,
    MaskedGeometry,
    finalize_inference_output,
    inference_autocast,
    mask_depth_and_points,
    prepare_inference_input,
    recover_shift_and_intrinsics,
    select_num_tokens,
)
from monopriors.third_party.moge.utils.geometry_torch import depth_map_to_point_map, normalized_view_plane_uv

ActivationMode: TypeAlias = Literal["relu", "leaky_relu", "silu", "elu"]
NormalizationMode: TypeAlias = Literal["group_norm", "layer_norm"]
EncoderLayer: TypeAlias = tuple[Float[Tensor, "b n c"], Float[Tensor, "b c"]]
ForwardOutput: TypeAlias = dict[str, Float[Tensor, "*shape"]]


def _require_checkpoint_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"MoGe v1 checkpoint field {field!r} must be a mapping")
    return value


def _require_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"MoGe v1 config field {field!r} must be an integer")
    return value


def _require_int_tuple(value: object, *, field: str, length: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"MoGe v1 config field {field!r} must contain integers")
    parsed: tuple[int, ...] = tuple(value)
    if length is not None and len(parsed) != length:
        raise ValueError(f"MoGe v1 config field {field!r} must contain {length} integers")
    return parsed


@dataclass(frozen=True, slots=True)
class MoGeV1Config:
    """Normalized configuration for the sole supported MoGe v1 checkpoint."""

    encoder: Literal["dinov2_vitl14"]
    """DINOv2 encoder shipped by the checkpoint."""
    remap_output: Literal["exp"]
    """Checkpoint point-remapping mode."""
    intermediate_layers: int
    """Number of intermediate encoder layers consumed by the head."""
    dim_upsample: tuple[int, ...]
    """Channel counts for successive upsampling blocks."""
    dim_times_res_block_hidden: int
    """Residual-block hidden-channel multiplier."""
    num_res_blocks: int
    """Residual blocks per upsampling stage."""
    num_tokens_range: tuple[int, int]
    """Normalized inference token range translated from trained pixel area."""
    last_conv_channels: int
    """Channels in the final prediction block."""
    last_conv_size: int
    """Kernel size of the final prediction convolution."""

    @classmethod
    def from_checkpoint_config(cls, config: Mapping[str, object]) -> "MoGeV1Config":
        """Validate and normalize the Ruicheng/moge-vitl checkpoint config.

        Args:
            config: Raw checkpoint configuration mapping.

        Returns:
            Strict normalized model configuration.

        Raises:
            ValueError: If a key or checkpoint-family invariant is unsupported.
        """
        expected_keys: frozenset[str] = frozenset(
            {
                "encoder",
                "remap_output",
                "output_mask",
                "split_head",
                "intermediate_layers",
                "dim_upsample",
                "dim_times_res_block_hidden",
                "num_res_blocks",
                "trained_area_range",
                "last_conv_channels",
                "last_conv_size",
            }
        )
        unknown_keys: set[str] = set(config) - expected_keys
        missing_keys: frozenset[str] = expected_keys - set(config)
        if unknown_keys:
            raise ValueError(f"Unsupported MoGe v1 config keys: {sorted(unknown_keys)}")
        if missing_keys:
            raise ValueError(f"Missing MoGe v1 config keys: {sorted(missing_keys)}")
        if config["encoder"] != "dinov2_vitl14":
            raise ValueError(f"Unsupported MoGe v1 encoder: {config['encoder']!r}")
        if config["remap_output"] != "exp":
            raise ValueError(f"Unsupported MoGe v1 output remap: {config['remap_output']!r}")
        if config["output_mask"] is not True or config["split_head"] is not True:
            raise ValueError("MoGe v1 requires output_mask=True and split_head=True")

        trained_area_range: tuple[int, ...] = _require_int_tuple(config["trained_area_range"], field="trained_area_range", length=2)
        num_tokens_range: tuple[int, int] = (trained_area_range[0] // 14**2, trained_area_range[1] // 14**2)
        return cls(
            encoder="dinov2_vitl14",
            remap_output="exp",
            intermediate_layers=_require_int(config["intermediate_layers"], field="intermediate_layers"),
            dim_upsample=_require_int_tuple(config["dim_upsample"], field="dim_upsample"),
            dim_times_res_block_hidden=_require_int(config["dim_times_res_block_hidden"], field="dim_times_res_block_hidden"),
            num_res_blocks=_require_int(config["num_res_blocks"], field="num_res_blocks"),
            num_tokens_range=num_tokens_range,
            last_conv_channels=_require_int(config["last_conv_channels"], field="last_conv_channels"),
            last_conv_size=_require_int(config["last_conv_size"], field="last_conv_size"),
        )


class ResidualConvBlock(nn.Module):
    """Two-convolution residual block used by the MoGe v1 decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        padding_mode: str = "replicate",
        activation: ActivationMode = "relu",
        norm: NormalizationMode = "group_norm",
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        activation_factory: Callable[[], nn.Module]
        if activation == "relu":
            activation_factory = functools.partial(nn.ReLU, inplace=True)
        elif activation == "leaky_relu":
            activation_factory = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
        elif activation == "silu":
            activation_factory = functools.partial(nn.SiLU, inplace=True)
        elif activation == "elu":
            activation_factory = functools.partial(nn.ELU, inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers: nn.Sequential = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_factory(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32 if norm == "group_norm" else 1, hidden_channels),
            activation_factory(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
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


class Head(nn.Module):
    """MoGe v1 multi-scale point and mask prediction head."""

    def __init__(
        self,
        num_features: int,
        dim_in: int,
        dim_out: Sequence[int],
        dim_proj: int = 512,
        dim_upsample: tuple[int, ...] = (256, 128, 128),
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: NormalizationMode = "group_norm",
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
    ) -> None:
        super().__init__()
        self.projects: nn.ModuleList = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0) for _ in range(num_features)]
        )
        self.upsample_blocks: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_upsampler(in_channels + 2, out_channels),
                    *(
                        ResidualConvBlock(
                            out_channels,
                            out_channels,
                            dim_times_res_block_hidden * out_channels,
                            activation="relu",
                            norm=res_block_norm,
                        )
                        for _ in range(num_res_blocks)
                    ),
                )
                for in_channels, out_channels in zip((dim_proj,) + dim_upsample[:-1], dim_upsample, strict=True)
            ]
        )
        self.output_block: nn.ModuleList = nn.ModuleList(
            [
                self._make_output_block(
                    dim_upsample[-1] + 2,
                    output_channels,
                    dim_times_res_block_hidden,
                    last_res_blocks,
                    last_conv_channels,
                    last_conv_size,
                    res_block_norm,
                )
                for output_channels in dim_out
            ]
        )

    def _make_upsampler(self, in_channels: int, out_channels: int) -> nn.Sequential:
        upsampler: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(
        self,
        dim_in: int,
        dim_out: int,
        dim_times_res_block_hidden: int,
        last_res_blocks: int,
        last_conv_channels: int,
        last_conv_size: int,
        res_block_norm: NormalizationMode,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            *(
                ResidualConvBlock(
                    last_conv_channels,
                    last_conv_channels,
                    dim_times_res_block_hidden * last_conv_channels,
                    activation="relu",
                    norm=res_block_norm,
                )
                for _ in range(last_res_blocks)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_conv_channels,
                dim_out,
                kernel_size=last_conv_size,
                stride=1,
                padding=last_conv_size // 2,
                padding_mode="replicate",
            ),
        )

    def forward(
        self,
        hidden_states: Sequence[EncoderLayer],
        image_b3hw: Float[Tensor, "b 3 h w"],
    ) -> list[Float[Tensor, "b c h w"]]:
        """Decode DINOv2 features into point and mask maps.

        Args:
            hidden_states: Encoder feature/class-token pairs with shapes ``b n c`` and ``b c``.
            image_b3hw: Float RGB image tensor shaped ``b 3 h w``.

        Returns:
            Float point and mask tensors, each shaped ``b c h w``.
        """
        image_height: int = image_b3hw.shape[-2]
        image_width: int = image_b3hw.shape[-1]
        patch_height: int = image_height // 14
        patch_width: int = image_width // 14
        projected_features: list[Float[Tensor, "b c patch_h patch_w"]] = [
            projection(feature_bnc.permute(0, 2, 1).unflatten(2, (patch_height, patch_width)).contiguous())
            for projection, (feature_bnc, _class_token_bc) in zip(self.projects, hidden_states, strict=True)
        ]
        x_bchw: Float[Tensor, "b c h w"] = torch.stack(projected_features, dim=1).sum(dim=1)

        for block in self.upsample_blocks:
            uv_hw2: Float[Tensor, "h w 2"] = normalized_view_plane_uv(
                width=x_bchw.shape[-1],
                height=x_bchw.shape[-2],
                aspect_ratio=image_width / image_height,
                dtype=x_bchw.dtype,
                device=x_bchw.device,
            )
            uv_b2hw: Float[Tensor, "b 2 h w"] = uv_hw2.permute(2, 0, 1).unsqueeze(0).expand(x_bchw.shape[0], -1, -1, -1)
            x_bchw = torch.cat([x_bchw, uv_b2hw], dim=1)
            for layer in block:
                x_bchw = layer(x_bchw)

        x_bchw = F.interpolate(x_bchw, (image_height, image_width), mode="bilinear", align_corners=False)
        uv_hw2 = normalized_view_plane_uv(
            width=x_bchw.shape[-1],
            height=x_bchw.shape[-2],
            aspect_ratio=image_width / image_height,
            dtype=x_bchw.dtype,
            device=x_bchw.device,
        )
        uv_b2hw = uv_hw2.permute(2, 0, 1).unsqueeze(0).expand(x_bchw.shape[0], -1, -1, -1)
        x_bchw = torch.cat([x_bchw, uv_b2hw], dim=1)
        output: list[Float[Tensor, "b c h w"]] = [block(x_bchw) for block in self.output_block]
        return output


class MoGeModel(nn.Module):
    """Inference-only adapter for the Ruicheng/moge-vitl checkpoint."""

    image_mean: Float[Tensor, "1 3 1 1"]
    image_std: Float[Tensor, "1 3 1 1"]

    def __init__(self, config: MoGeV1Config) -> None:
        super().__init__()
        self.encoder: Literal["dinov2_vitl14"] = config.encoder
        self.remap_output: Literal["exp"] = config.remap_output
        self.intermediate_layers: int = config.intermediate_layers
        self.num_tokens_range: tuple[int, int] = config.num_tokens_range
        self.mask_threshold: float = 0.5

        self.backbone: DinoVisionTransformer = dinov2_vitl14(pretrained=False)
        dim_feature: int = self.backbone.blocks[0].attn.qkv.in_features
        self.head: Head = Head(
            num_features=config.intermediate_layers,
            dim_in=dim_feature,
            dim_out=[3, 1],
            dim_proj=512,
            dim_upsample=config.dim_upsample,
            dim_times_res_block_hidden=config.dim_times_res_block_hidden,
            num_res_blocks=config.num_res_blocks,
            res_block_norm="group_norm",
            last_res_blocks=0,
            last_conv_channels=config.last_conv_channels,
            last_conv_size=config.last_conv_size,
        )

        image_mean_1311: Float[Tensor, "1 3 1 1"] = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std_1311: Float[Tensor, "1 3 1 1"] = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean_1311)
        self.register_buffer("image_std", image_std_1311)

    @property
    def device(self) -> torch.device:
        """Device holding the model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype used by the model parameters."""
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str) -> Self:
        """Load the pinned Ruicheng/moge-vitl checkpoint from Hugging Face.

        Args:
            repo_id: Hugging Face model repository ID. Only ``Ruicheng/moge-vitl`` is supported.
            revision: Immutable Hugging Face repository commit SHA.

        Returns:
            Model loaded with exact state-dict matching.

        Raises:
            ValueError: If ``repo_id`` or its checkpoint configuration is unsupported.
            RuntimeError: If checkpoint parameters do not match the model exactly.
        """
        if repo_id != "Ruicheng/moge-vitl":
            raise ValueError(f"Unsupported MoGe v1 checkpoint repository: {repo_id!r}")
        cached_checkpoint_path: str = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="model.pt",
            revision=revision,
        )
        checkpoint: dict[str, object] = torch.load(cached_checkpoint_path, map_location="cpu", weights_only=True)
        raw_config: Mapping[str, object] = _require_checkpoint_mapping(checkpoint.get("model_config"), field="model_config")
        state_dict: dict[str, Float[Tensor, "*shape"]] = cast(
            dict[str, Float[Tensor, "*shape"]],
            _require_checkpoint_mapping(checkpoint.get("model"), field="model"),
        )
        model: MoGeModel = cls(MoGeV1Config.from_checkpoint_config(raw_config))
        model.load_state_dict(state_dict, strict=True)
        return model

    def _remap_points(self, points_bhw3: Float[Tensor, "b h w 3"]) -> Float[Tensor, "b h w 3"]:
        xy_bhw2: Float[Tensor, "b h w 2"]
        z_bhw1: Float[Tensor, "b h w 1"]
        xy_bhw2, z_bhw1 = points_bhw3.split([2, 1], dim=-1)
        z_bhw1 = torch.exp(z_bhw1)
        points_bhw3 = torch.cat([xy_bhw2 * z_bhw1, z_bhw1], dim=-1)
        return points_bhw3

    def forward(self, image_b3hw: Float[Tensor, "b 3 h w"], num_tokens: int) -> ForwardOutput:
        """Predict an affine point map and validity logits.

        Args:
            image_b3hw: Float RGB image tensor shaped ``b 3 h w`` in the range ``[0, 1]``.
            num_tokens: Approximate number of DINOv2 patch tokens.

        Returns:
            Float point tensor shaped ``b h w 3`` and mask-logit tensor shaped ``b h w``.
        """
        original_height: int = image_b3hw.shape[-2]
        original_width: int = image_b3hw.shape[-1]
        resize_factor: float = ((num_tokens * 14**2) / (original_height * original_width)) ** 0.5
        resized_width: int = int(original_width * resize_factor)
        resized_height: int = int(original_height * resize_factor)
        image_b3hw = F.interpolate(
            image_b3hw,
            (resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        image_b3hw = (image_b3hw - self.image_mean) / self.image_std
        image_14_b3hw: Float[Tensor, "b 3 patch_h patch_w"] = F.interpolate(
            image_b3hw,
            (resized_height // 14 * 14, resized_width // 14 * 14),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        raw_features: object = self.backbone.get_intermediate_layers(
            image_14_b3hw,
            self.intermediate_layers,
            return_class_token=True,
        )
        features: tuple[EncoderLayer, ...] = cast(tuple[EncoderLayer, ...], raw_features)
        output: list[Float[Tensor, "b c h w"]] = self.head(features, image_b3hw)
        points_b3hw: Float[Tensor, "b 3 h w"] = output[0]
        mask_b1hw: Float[Tensor, "b 1 h w"] = output[1]

        with torch.autocast(device_type=image_b3hw.device.type, dtype=torch.float32):
            points_b3hw = F.interpolate(
                points_b3hw,
                (original_height, original_width),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            mask_b1hw = F.interpolate(
                mask_b1hw,
                (original_height, original_width),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
            points_bhw3: Float[Tensor, "b h w 3"] = points_b3hw.permute(0, 2, 3, 1)
            mask_bhw: Float[Tensor, "b h w"] = mask_b1hw.squeeze(1)
            points_bhw3 = self._remap_points(points_bhw3)

        return {"points": points_bhw3, "mask": mask_bhw}

    @torch.inference_mode()
    def infer(
        self,
        image: Float[Tensor, "3 h w"] | Float[Tensor, "b 3 h w"],
        resolution_level: int = 9,
        num_tokens: int | None = None,
        apply_mask: bool = True,
        force_projection: bool = True,
        use_fp16: bool = True,
    ) -> InferenceOutput:
        """Run user-facing MoGe v1 inference.

        Args:
            image: Float RGB image tensor shaped ``3 h w`` or ``b 3 h w``.
            resolution_level: Detail level from 0 through 9 used when ``num_tokens`` is absent.
            num_tokens: Explicit DINOv2 token count, or ``None`` to derive it.
            apply_mask: Whether invalid point and depth values become infinity.
            force_projection: Whether to reproject depth into a pinhole-consistent point map.
            use_fp16: Whether to use float16 mixed precision for the network forward pass.

        Returns:
            Tensor dictionary containing points, depth, normalized intrinsics, and a bool mask. Batched inputs retain their batch dimension.
        """
        inference_input: InferenceInput = prepare_inference_input(image, device=self.device, dtype=self.dtype)
        selected_num_tokens: int = select_num_tokens(
            self.num_tokens_range,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
        )

        with inference_autocast(device=self.device, dtype=self.dtype, use_fp16=use_fp16):
            output: ForwardOutput = self.forward(inference_input.image_bchw, selected_num_tokens)
        points_bhw3: Float[Tensor, "b h w 3"] = output["points"]
        mask_bhw: Float[Tensor, "b h w"] = output["mask"]

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            points_bhw3 = points_bhw3.float()
            mask_bhw = mask_bhw.float()

            mask_binary_bhw: Bool[Tensor, "b h w"] = mask_bhw > self.mask_threshold
            camera: CameraRecovery = recover_shift_and_intrinsics(
                points_bhw3,
                mask_binary_bhw,
                aspect_ratio=inference_input.aspect_ratio,
            )
            depth_bhw: Float[Tensor, "b h w"] = points_bhw3[..., 2] + camera.shift_b[..., None, None]

            if force_projection:
                points_bhw3 = depth_map_to_point_map(depth_bhw, intrinsics=camera.intrinsics_b33)
            else:
                points_bhw3 = (
                    points_bhw3
                    + torch.stack(
                        [torch.zeros_like(camera.shift_b), torch.zeros_like(camera.shift_b), camera.shift_b],
                        dim=-1,
                    )[..., None, None, :]
                )

            if apply_mask:
                masked_geometry: MaskedGeometry = mask_depth_and_points(points_bhw3, depth_bhw, mask_binary_bhw)
                points_bhw3 = masked_geometry.points_bhw3
                depth_bhw = masked_geometry.depth_bhw

        return finalize_inference_output(
            {
                "points": points_bhw3,
                "intrinsics": camera.intrinsics_b33,
                "depth": depth_bhw,
                "mask": mask_binary_bhw,
            },
            omit_batch_dim=inference_input.omit_batch_dim,
        )
