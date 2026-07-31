from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal, Self, TypeAlias, cast

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

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
from monopriors.third_party.moge.model.modules import MLP, ConvStack, DINOv2Encoder
from monopriors.third_party.moge.utils.geometry_torch import depth_map_to_point_map, normalized_view_plane_uv

BackboneName: TypeAlias = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"]
NormalizationMode: TypeAlias = Literal["layer_norm", "group_norm", "instance_norm", "none"]
ResamplerMode: TypeAlias = Literal["bilinear", "conv_transpose"]
ActivationMode: TypeAlias = Literal["relu", "leaky_relu", "silu", "elu"]
RemapMode: TypeAlias = Literal["linear", "sinh", "exp", "sinh_exp"]
OutputHead: TypeAlias = Literal["points", "normal", "mask", "scale"]
ForwardOutput: TypeAlias = dict[str, Float[Tensor, "*shape"]]


def _validate_keys(
    config: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    unknown_keys: set[str] = set(config) - required - optional
    missing_keys: frozenset[str] = required - set(config)
    if unknown_keys:
        raise ValueError(f"Unsupported {context} config keys: {sorted(unknown_keys)}")
    if missing_keys:
        raise ValueError(f"Missing {context} config keys: {sorted(missing_keys)}")


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_int_tuple(value: object, *, context: str, allow_none: bool = False) -> tuple[int | None, ...]:
    if not isinstance(value, list) or not all(
        (isinstance(item, int) and not isinstance(item, bool)) or (allow_none and item is None) for item in value
    ):
        qualifier: str = "integers or None" if allow_none else "integers"
        raise ValueError(f"{context} must contain {qualifier}")
    return tuple(value)


def _require_string_tuple(value: object, *, context: str, choices: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item in choices for item in value):
        raise ValueError(f"{context} contains an unsupported value")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class DINOv2EncoderConfig:
    """Typed MoGe v2 encoder configuration."""

    backbone: BackboneName
    """Vendored DINOv2 backbone factory name."""
    intermediate_layers: int | tuple[int, ...]
    """Encoder block indices or trailing-layer count used by MoGe."""
    dim_out: int
    """Projected feature-channel count."""

    @classmethod
    def from_checkpoint_config(cls, config: Mapping[str, object]) -> "DINOv2EncoderConfig":
        """Validate an encoder checkpoint mapping.

        Args:
            config: Raw nested encoder configuration.

        Returns:
            Typed encoder configuration.
        """
        _validate_keys(config, required=frozenset({"backbone", "intermediate_layers", "dim_out"}), context="MoGe v2 encoder")
        backbone: object = config["backbone"]
        if backbone not in {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"}:
            raise ValueError(f"Unsupported MoGe v2 backbone: {backbone!r}")
        intermediate_layers_value: object = config["intermediate_layers"]
        intermediate_layers: int | tuple[int, ...]
        if isinstance(intermediate_layers_value, int) and not isinstance(intermediate_layers_value, bool):
            intermediate_layers = intermediate_layers_value
        else:
            parsed_layers: tuple[int | None, ...] = _require_int_tuple(
                intermediate_layers_value,
                context="MoGe v2 encoder intermediate_layers",
            )
            intermediate_layers = tuple(cast(int, layer) for layer in parsed_layers)
        dim_out: object = config["dim_out"]
        if not isinstance(dim_out, int) or isinstance(dim_out, bool):
            raise ValueError("MoGe v2 encoder dim_out must be an integer")
        return cls(
            backbone=cast(BackboneName, backbone),
            intermediate_layers=intermediate_layers,
            dim_out=dim_out,
        )


@dataclass(frozen=True, slots=True)
class ConvStackConfig:
    """Typed configuration for a MoGe v2 multi-resolution convolution stack."""

    dim_in: tuple[int | None, ...]
    """Input-channel count at each stack level."""
    dim_res_blocks: tuple[int, ...]
    """Residual-block channel count at each stack level."""
    dim_out: tuple[int | None, ...] | None
    """Output-channel count at each stack level, or no projections for the neck."""
    resamplers: tuple[ResamplerMode, ...]
    """Upsampling operation between successive levels."""
    dim_times_res_block_hidden: int
    """Residual-block hidden-channel multiplier."""
    num_res_blocks: int | tuple[int, ...]
    """Residual-block count shared by or specified for each level."""
    res_block_in_norm: NormalizationMode
    """Normalization applied at each residual-block input."""
    res_block_hidden_norm: NormalizationMode
    """Normalization applied to residual-block hidden channels."""
    activation: ActivationMode
    """Residual-block activation function."""

    @classmethod
    def from_checkpoint_config(cls, config: Mapping[str, object], *, context: str) -> "ConvStackConfig":
        """Validate a convolution-stack checkpoint mapping.

        Args:
            config: Raw nested stack configuration.
            context: Human-readable field name for validation errors.

        Returns:
            Typed convolution-stack configuration.
        """
        _validate_keys(
            config,
            required=frozenset(
                {
                    "dim_in",
                    "dim_res_blocks",
                    "dim_out",
                    "resamplers",
                    "num_res_blocks",
                    "res_block_in_norm",
                    "res_block_hidden_norm",
                }
            ),
            optional=frozenset({"dim_times_res_block_hidden", "activation"}),
            context=context,
        )
        parsed_dim_in: tuple[int | None, ...] = _require_int_tuple(config["dim_in"], context=f"{context} dim_in", allow_none=True)
        parsed_dim_res_blocks: tuple[int | None, ...] = _require_int_tuple(config["dim_res_blocks"], context=f"{context} dim_res_blocks")
        dim_out_value: object = config["dim_out"]
        parsed_dim_out: tuple[int | None, ...] | None = (
            None if dim_out_value is None else _require_int_tuple(dim_out_value, context=f"{context} dim_out", allow_none=True)
        )
        parsed_resamplers: tuple[str, ...] = _require_string_tuple(
            config["resamplers"],
            context=f"{context} resamplers",
            choices=frozenset({"bilinear", "conv_transpose"}),
        )
        num_res_blocks_value: object = config["num_res_blocks"]
        num_res_blocks: int | tuple[int, ...]
        if isinstance(num_res_blocks_value, int) and not isinstance(num_res_blocks_value, bool):
            num_res_blocks = num_res_blocks_value
        else:
            parsed_num_res_blocks: tuple[int | None, ...] = _require_int_tuple(
                num_res_blocks_value,
                context=f"{context} num_res_blocks",
            )
            num_res_blocks = tuple(cast(int, value) for value in parsed_num_res_blocks)
        in_norm: object = config["res_block_in_norm"]
        hidden_norm: object = config["res_block_hidden_norm"]
        activation: object = config.get("activation", "relu")
        norm_choices: set[str] = {"layer_norm", "group_norm", "instance_norm", "none"}
        if in_norm not in norm_choices or hidden_norm not in norm_choices:
            raise ValueError(f"{context} contains an unsupported normalization mode")
        if activation not in {"relu", "leaky_relu", "silu", "elu"}:
            raise ValueError(f"{context} contains an unsupported activation")
        hidden_multiplier: object = config.get("dim_times_res_block_hidden", 1)
        if not isinstance(hidden_multiplier, int) or isinstance(hidden_multiplier, bool):
            raise ValueError(f"{context} dim_times_res_block_hidden must be an integer")
        return cls(
            dim_in=parsed_dim_in,
            dim_res_blocks=tuple(cast(int, value) for value in parsed_dim_res_blocks),
            dim_out=parsed_dim_out,
            resamplers=cast(tuple[ResamplerMode, ...], parsed_resamplers),
            dim_times_res_block_hidden=hidden_multiplier,
            num_res_blocks=num_res_blocks,
            res_block_in_norm=cast(NormalizationMode, in_norm),
            res_block_hidden_norm=cast(NormalizationMode, hidden_norm),
            activation=cast(ActivationMode, activation),
        )


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Typed configuration for the metric-scale MLP head."""

    dims: tuple[int, ...]
    """Feature dimensions for successive linear layers."""

    @classmethod
    def from_checkpoint_config(cls, config: Mapping[str, object]) -> "MLPConfig":
        """Validate a metric-scale head checkpoint mapping.

        Args:
            config: Raw nested MLP configuration.

        Returns:
            Typed MLP configuration.
        """
        _validate_keys(config, required=frozenset({"dims"}), context="MoGe v2 scale_head")
        parsed_dims: tuple[int | None, ...] = _require_int_tuple(config["dims"], context="MoGe v2 scale_head dims")
        return cls(dims=tuple(cast(int, value) for value in parsed_dims))


@dataclass(frozen=True, slots=True)
class MoGeV2Config:
    """Normalized configuration shared by the four supported MoGe v2 checkpoints."""

    encoder: DINOv2EncoderConfig
    """DINOv2 encoder configuration."""
    neck: ConvStackConfig
    """Shared multi-resolution feature neck."""
    points_head: ConvStackConfig
    """Invariant affine point-map prediction head."""
    mask_head: ConvStackConfig
    """Invariant validity-mask prediction head."""
    scale_head: MLPConfig
    """Invariant metric-scale prediction head."""
    normal_head: ConvStackConfig | None
    """Optional surface-normal prediction head."""
    remap_output: RemapMode
    """Point-map output remapping mode."""
    num_tokens_range: tuple[int, int]
    """Recommended inference token-count range."""

    @classmethod
    def from_checkpoint_config(cls, config: Mapping[str, object]) -> "MoGeV2Config":
        """Validate and normalize a supported MoGe v2 checkpoint config.

        Args:
            config: Raw checkpoint configuration mapping.

        Returns:
            Typed configuration with required points, mask, and scale heads.
        """
        _validate_keys(
            config,
            required=frozenset({"encoder", "neck", "points_head", "mask_head", "scale_head", "remap_output", "num_tokens_range"}),
            optional=frozenset({"normal_head"}),
            context="MoGe v2",
        )
        remap_output: object = config["remap_output"]
        if remap_output not in {"linear", "sinh", "exp", "sinh_exp"}:
            raise ValueError(f"Unsupported MoGe v2 output remap: {remap_output!r}")
        parsed_token_range: tuple[int | None, ...] = _require_int_tuple(config["num_tokens_range"], context="MoGe v2 num_tokens_range")
        if len(parsed_token_range) != 2:
            raise ValueError("MoGe v2 num_tokens_range must contain two integers")
        normal_head_value: object | None = config.get("normal_head")
        normal_head: ConvStackConfig | None = (
            ConvStackConfig.from_checkpoint_config(
                _require_mapping(normal_head_value, context="MoGe v2 normal_head"),
                context="MoGe v2 normal_head",
            )
            if normal_head_value is not None
            else None
        )
        return cls(
            encoder=DINOv2EncoderConfig.from_checkpoint_config(_require_mapping(config["encoder"], context="MoGe v2 encoder")),
            neck=ConvStackConfig.from_checkpoint_config(
                _require_mapping(config["neck"], context="MoGe v2 neck"),
                context="MoGe v2 neck",
            ),
            points_head=ConvStackConfig.from_checkpoint_config(
                _require_mapping(config["points_head"], context="MoGe v2 points_head"),
                context="MoGe v2 points_head",
            ),
            mask_head=ConvStackConfig.from_checkpoint_config(
                _require_mapping(config["mask_head"], context="MoGe v2 mask_head"),
                context="MoGe v2 mask_head",
            ),
            scale_head=MLPConfig.from_checkpoint_config(_require_mapping(config["scale_head"], context="MoGe v2 scale_head")),
            normal_head=normal_head,
            remap_output=cast(RemapMode, remap_output),
            num_tokens_range=(cast(int, parsed_token_range[0]), cast(int, parsed_token_range[1])),
        )


SUPPORTED_REPO_IDS: frozenset[str] = frozenset(
    {
        "Ruicheng/moge-2-vitl",
        "Ruicheng/moge-2-vitl-normal",
        "Ruicheng/moge-2-vitb-normal",
        "Ruicheng/moge-2-vits-normal",
    }
)


def _build_conv_stack(config: ConvStackConfig) -> ConvStack:
    return ConvStack(
        dim_in=config.dim_in,
        dim_res_blocks=config.dim_res_blocks,
        dim_out=config.dim_out,
        resamplers=config.resamplers,
        dim_times_res_block_hidden=config.dim_times_res_block_hidden,
        num_res_blocks=config.num_res_blocks,
        res_block_in_norm=config.res_block_in_norm,
        res_block_hidden_norm=config.res_block_hidden_norm,
        activation=config.activation,
    )


class MoGeModel(nn.Module):
    """Inference-only adapter for the four supported MoGe v2 checkpoints."""

    encoder: DINOv2Encoder
    neck: ConvStack
    points_head: ConvStack
    normal_head: ConvStack
    mask_head: ConvStack
    scale_head: MLP

    def __init__(self, config: MoGeV2Config) -> None:
        super().__init__()
        self.remap_output: RemapMode = config.remap_output
        self.num_tokens_range: tuple[int, int] = config.num_tokens_range
        self.encoder = DINOv2Encoder(
            backbone=config.encoder.backbone,
            intermediate_layers=config.encoder.intermediate_layers,
            dim_out=config.encoder.dim_out,
        )
        self.neck = _build_conv_stack(config.neck)
        self.points_head = _build_conv_stack(config.points_head)
        self.mask_head = _build_conv_stack(config.mask_head)
        if config.normal_head is not None:
            self.normal_head = _build_conv_stack(config.normal_head)
        self.scale_head = MLP(config.scale_head.dims)
        optional_heads: set[OutputHead] = {"normal"} if config.normal_head is not None else set()
        self.available_heads: frozenset[OutputHead] = frozenset({"points", "mask", "scale", *optional_heads})

    @property
    def device(self) -> torch.device:
        """Device holding the model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype used by the model parameters."""
        return next(self.parameters()).dtype

    @property
    def onnx_compatible_mode(self) -> bool:
        """Whether ONNX-safe interpolation is enabled."""
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool) -> None:
        self._onnx_compatible_mode: bool = value
        self.encoder.onnx_compatible_mode = value

    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str) -> Self:
        """Load one pinned supported MoGe v2 checkpoint from Hugging Face.

        Args:
            repo_id: Hugging Face repository ID for one of the four supported checkpoints.
            revision: Immutable Hugging Face repository commit SHA.

        Returns:
            Model loaded with exact state-dict matching.

        Raises:
            ValueError: If the repository or checkpoint configuration is unsupported.
            RuntimeError: If checkpoint parameters do not match the model exactly.
        """
        if repo_id not in SUPPORTED_REPO_IDS:
            raise ValueError(f"Unsupported MoGe v2 checkpoint repository: {repo_id!r}")
        checkpoint_path: str = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="model.pt",
            revision=revision,
        )
        checkpoint: dict[str, object] = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        raw_config: Mapping[str, object] = _require_mapping(
            checkpoint.get("model_config"),
            context="MoGe v2 checkpoint model_config",
        )
        state_dict: dict[str, Float[Tensor, "*shape"]] = cast(
            dict[str, Float[Tensor, "*shape"]],
            _require_mapping(checkpoint.get("model"), context="MoGe v2 checkpoint model"),
        )
        model: MoGeModel = cls(MoGeV2Config.from_checkpoint_config(raw_config))
        model.load_state_dict(state_dict, strict=True)
        return model

    def _remap_points(self, points_bhw3: Float[Tensor, "b h w 3"]) -> Float[Tensor, "b h w 3"]:
        if self.remap_output == "linear":
            pass
        elif self.remap_output == "sinh":
            points_bhw3 = torch.sinh(points_bhw3)
        elif self.remap_output == "exp":
            xy_bhw2: Float[Tensor, "b h w 2"]
            z_bhw1: Float[Tensor, "b h w 1"]
            xy_bhw2, z_bhw1 = points_bhw3.split([2, 1], dim=-1)
            z_bhw1 = torch.exp(z_bhw1)
            points_bhw3 = torch.cat([xy_bhw2 * z_bhw1, z_bhw1], dim=-1)
        elif self.remap_output == "sinh_exp":
            xy_bhw2, z_bhw1 = points_bhw3.split([2, 1], dim=-1)
            points_bhw3 = torch.cat([torch.sinh(xy_bhw2), torch.exp(z_bhw1)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points_bhw3

    def forward(
        self,
        image_b3hw: Float[Tensor, "b 3 h w"],
        num_tokens: int | Int[Tensor, ""],
        *,
        output_heads: Collection[OutputHead] | None = None,
    ) -> ForwardOutput:
        """Run the shared encoder, neck, and selected prediction heads.

        Args:
            image_b3hw: Float RGB image tensor shaped ``b 3 h w`` in the range ``[0, 1]``.
            num_tokens: Approximate token count as an integer or scalar integer tensor.
            output_heads: Optional subset of points, normals, mask, and scale heads. ``None`` runs every checkpoint head.

        Returns:
            Float tensor dictionary containing the requested raw head outputs.

        Raises:
            ValueError: If the checkpoint does not provide a requested head.
        """
        requested_heads: frozenset[OutputHead] = self.available_heads if output_heads is None else frozenset(output_heads)
        unsupported_heads: frozenset[OutputHead] = requested_heads - self.available_heads
        if unsupported_heads:
            raise ValueError(f"Unavailable MoGe v2 output heads: {sorted(unsupported_heads)}")

        batch_size: int = image_b3hw.shape[0]
        image_height: int = image_b3hw.shape[2]
        image_width: int = image_b3hw.shape[3]
        device: torch.device = image_b3hw.device
        dtype: torch.dtype = image_b3hw.dtype
        aspect_ratio: float = image_width / image_height
        base_height: float | Float[Tensor, ""] | Int[Tensor, ""] = (num_tokens / aspect_ratio) ** 0.5
        base_width: float | Float[Tensor, ""] | Int[Tensor, ""] = (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_height, Tensor):
            base_height = base_height.round().long()
            base_width_tensor: Float[Tensor, ""] = cast(Float[Tensor, ""], base_width)
            base_width = base_width_tensor.round().long()
        else:
            base_height = round(base_height)
            base_width = round(cast(float, base_width))

        encoder_output: tuple[Float[Tensor, "b c token_h token_w"], Float[Tensor, "b c"]] = self.encoder(
            image_b3hw,
            base_height,
            base_width,
            return_class_token=True,
        )
        encoded_bchw: Float[Tensor, "b c token_h token_w"] = encoder_output[0]
        class_token_bc: Float[Tensor, "b c"] = encoder_output[1]
        feature_pyramid: list[Float[Tensor, "b c h w"] | None] = [encoded_bchw, None, None, None, None]

        for level in range(5):
            uv_hw2: Float[Tensor, "h w 2"] = normalized_view_plane_uv(
                width=base_width * 2**level,
                height=base_height * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv_b2hw: Float[Tensor, "b 2 h w"] = uv_hw2.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            feature_bchw: Float[Tensor, "b c h w"] | None = feature_pyramid[level]
            if feature_bchw is None:
                feature_pyramid[level] = uv_b2hw
            else:
                feature_pyramid[level] = torch.concat([feature_bchw, uv_b2hw], dim=1)

        features: list[Float[Tensor, "b c h w"]] = self.neck(cast(list[Float[Tensor, "b c h w"]], feature_pyramid))
        points_b3hw: Float[Tensor, "b 3 h w"] | None = self.points_head(features)[-1] if "points" in requested_heads else None
        normal_b3hw: Float[Tensor, "b 3 h w"] | None = self.normal_head(features)[-1] if "normal" in requested_heads else None
        mask_b1hw: Float[Tensor, "b 1 h w"] | None = self.mask_head(features)[-1] if "mask" in requested_heads else None
        metric_scale_b1: Float[Tensor, "b 1"] | None = self.scale_head(class_token_bc) if "scale" in requested_heads else None

        points_b3hw = (
            F.interpolate(points_b3hw, (image_height, image_width), mode="bilinear", align_corners=False, antialias=False)
            if points_b3hw is not None
            else None
        )
        normal_b3hw = (
            F.interpolate(normal_b3hw, (image_height, image_width), mode="bilinear", align_corners=False, antialias=False)
            if normal_b3hw is not None
            else None
        )
        mask_b1hw = (
            F.interpolate(mask_b1hw, (image_height, image_width), mode="bilinear", align_corners=False, antialias=False)
            if mask_b1hw is not None
            else None
        )

        points_bhw3: Float[Tensor, "b h w 3"] | None = None
        normal_bhw3: Float[Tensor, "b h w 3"] | None = None
        mask_bhw: Float[Tensor, "b h w"] | None = None
        metric_scale_b: Float[Tensor, "b"] | None = None
        if points_b3hw is not None:
            points_bhw3 = points_b3hw.permute(0, 2, 3, 1)
            points_bhw3 = self._remap_points(points_bhw3)
        if normal_b3hw is not None:
            normal_bhw3 = normal_b3hw.permute(0, 2, 3, 1)
            normal_bhw3 = F.normalize(normal_bhw3, dim=-1)
        if mask_b1hw is not None:
            mask_bhw = mask_b1hw.squeeze(1).sigmoid()
        if metric_scale_b1 is not None:
            metric_scale_b = metric_scale_b1.squeeze(1).exp()

        optional_output: dict[str, Float[Tensor, "*shape"] | None] = {
            "points": points_bhw3,
            "normal": normal_bhw3,
            "mask": mask_bhw,
            "metric_scale": metric_scale_b,
        }
        return {key: value for key, value in optional_output.items() if value is not None}

    @torch.inference_mode()
    def infer(
        self,
        image: Float[Tensor, "3 h w"] | Float[Tensor, "b 3 h w"],
        num_tokens: int | None = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        use_fp16: bool = True,
        output_heads: Collection[OutputHead] | None = None,
    ) -> InferenceOutput:
        """Run user-facing MoGe v2 inference.

        Args:
            image: Float RGB image tensor shaped ``3 h w`` or ``b 3 h w``.
            num_tokens: Explicit DINOv2 token count, or ``None`` to derive it.
            resolution_level: Detail level from 0 through 9 used when ``num_tokens`` is absent.
            force_projection: Whether to reproject depth into a pinhole-consistent point map.
            apply_mask: Whether to mask invalid geometry and normals.
            use_fp16: Whether to use float16 mixed precision for the network forward pass.
            output_heads: Optional network heads to compute. ``None`` preserves the full default output.

        Returns:
            Tensor dictionary containing available points, depth, intrinsics, mask, and normals. Batched inputs retain their batch dimension.
        """
        inference_input: InferenceInput = prepare_inference_input(image, device=self.device, dtype=self.dtype)
        selected_num_tokens: int = select_num_tokens(
            self.num_tokens_range,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
        )

        with inference_autocast(device=self.device, dtype=self.dtype, use_fp16=use_fp16):
            output: ForwardOutput = self.forward(
                inference_input.image_bchw,
                num_tokens=selected_num_tokens,
                output_heads=output_heads,
            )
        points_bhw3: Float[Tensor, "b h w 3"] | None = output.get("points")
        normal_bhw3: Float[Tensor, "b h w 3"] | None = output.get("normal")
        mask_bhw: Float[Tensor, "b h w"] | None = output.get("mask")
        metric_scale_b: Float[Tensor, "b"] | None = output.get("metric_scale")

        points_bhw3 = points_bhw3.float() if points_bhw3 is not None else None
        normal_bhw3 = normal_bhw3.float() if normal_bhw3 is not None else None
        mask_bhw = mask_bhw.float() if mask_bhw is not None else None
        metric_scale_b = metric_scale_b.float() if metric_scale_b is not None else None
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            mask_binary_bhw: Bool[Tensor, "b h w"] | None = mask_bhw > 0.5 if mask_bhw is not None else None
            intrinsics_b33: Float[Tensor, "b 3 3"] | None
            depth_bhw: Float[Tensor, "b h w"] | None

            if points_bhw3 is not None:
                camera: CameraRecovery = recover_shift_and_intrinsics(
                    points_bhw3,
                    mask_binary_bhw,
                        aspect_ratio=inference_input.aspect_ratio,
                )
                intrinsics_b33 = camera.intrinsics_b33
                points_bhw3[..., 2] += camera.shift_b[..., None, None]
                if mask_binary_bhw is not None:
                    mask_binary_bhw &= points_bhw3[..., 2] > 0
                depth_bhw = points_bhw3[..., 2].clone()
            else:
                depth_bhw = None
                intrinsics_b33 = None

            if force_projection and depth_bhw is not None:
                points_bhw3 = depth_map_to_point_map(depth_bhw, intrinsics=intrinsics_b33)

            if metric_scale_b is not None:
                if points_bhw3 is not None:
                    points_bhw3 *= metric_scale_b[:, None, None, None]
                if depth_bhw is not None:
                    depth_bhw *= metric_scale_b[:, None, None]

            if apply_mask and mask_binary_bhw is not None:
                if points_bhw3 is not None and depth_bhw is not None:
                    masked_geometry: MaskedGeometry = mask_depth_and_points(points_bhw3, depth_bhw, mask_binary_bhw)
                    points_bhw3 = masked_geometry.points_bhw3
                    depth_bhw = masked_geometry.depth_bhw
                normal_bhw3 = torch.where(mask_binary_bhw[..., None], normal_bhw3, torch.zeros_like(normal_bhw3)) if normal_bhw3 is not None else None

        return finalize_inference_output(
            {
                "points": points_bhw3,
                "intrinsics": intrinsics_b33,
                "depth": depth_bhw,
                "mask": mask_binary_bhw,
                "normal": normal_bhw3,
            },
            omit_batch_dim=inference_input.omit_batch_dim,
        )
