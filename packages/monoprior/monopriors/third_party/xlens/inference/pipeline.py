"""Load released X-Lens checkpoints and run calibrated rig-depth inference."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
import yaml
from jaxtyping import Float32
from safetensors.torch import load_file
from torch import Tensor

from monopriors.third_party.xlens.inference.preprocess import AssembledBatch
from monopriors.third_party.xlens.models import XLensNet

logger = logging.getLogger(__name__)

BackboneName = Literal["vits", "vitb", "vitl"]
ScaleHeadMode = Literal["cls", "attn_pool"]
DistortionBiasLayers = Literal["global_only", "all"]
AmpDtype = Literal["bf16", "fp16"]


class XLensOutput(TypedDict, total=False):
    """Released X-Lens inference outputs, copied to CPU float32."""

    depth: Float32[Tensor, "batch views height width"]
    depth_metric: Float32[Tensor, "batch views height width"]
    depth_conf: Float32[Tensor, "batch views height width"]
    metric_scaling_factor: Float32[Tensor, "batch"]
    mask_logits: Float32[Tensor, "batch views height width"]
    mask: Float32[Tensor, "batch views height width"]


@dataclass(frozen=True, slots=True)
class XLensArchitectureConfig:
    """Inference architecture fields stored beside or inside a checkpoint."""

    backbone: BackboneName = "vitl"
    checkpoint_dir: str = "checkpoints"
    head_features: int = 256
    head_out_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024)
    predict_mask: bool = False
    scale_head_mode: ScaleHeadMode = "cls"
    scale_head_num_queries: int = 4
    scale_head_num_heads: int = 8
    use_dwc: bool = False
    dwc_kernel_size: int = 3
    n_cam_types: int = 0
    use_calib_tokens: bool = False
    calib_tokens_per_type: int = 4
    calib_token_inject_types: tuple[int, ...] = (0,)
    use_distortion_bias: bool = False
    distortion_bias_layers: DistortionBiasLayers = "global_only"
    distortion_bias_hidden_dim: int = 64
    distortion_bias_chunk_size: int = 1024

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> XLensArchitectureConfig:
        """Validate an untyped YAML/checkpoint mapping at the I/O boundary.

        Args:
            values: Architecture values loaded from YAML or a PyTorch checkpoint.

        Returns:
            A validated immutable architecture configuration.

        Raises:
            ValueError: If a literal, integer sequence, or scalar has an invalid value.
        """
        backbone_value = values.get("backbone", "vitl")
        if backbone_value not in ("vits", "vitb", "vitl"):
            raise ValueError(f"unsupported X-Lens backbone: {backbone_value!r}")
        scale_mode_value = values.get("scale_head_mode", "cls")
        if scale_mode_value not in ("cls", "attn_pool"):
            raise ValueError(f"unsupported scale-head mode: {scale_mode_value!r}")
        distortion_layers_value = values.get("distortion_bias_layers", "global_only")
        if distortion_layers_value not in ("global_only", "all"):
            raise ValueError(f"unsupported distortion-bias layer selection: {distortion_layers_value!r}")

        raw_head_channels = values.get("head_out_channels", (256, 512, 1024, 1024))
        if (
            not isinstance(raw_head_channels, (list, tuple))
            or len(raw_head_channels) != 4
            or not all(isinstance(value, int) for value in raw_head_channels)
        ):
            raise ValueError("head_out_channels must contain four integers")
        raw_inject_types = values.get("calib_token_inject_types", (0,))
        if not isinstance(raw_inject_types, (list, tuple)) or not all(isinstance(value, int) for value in raw_inject_types):
            raise ValueError("calib_token_inject_types must contain integers")

        def integer(name: str, default: int) -> int:
            """Read one integer field."""
            value = values.get(name, default)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            return value

        def boolean(name: str, default: bool) -> bool:
            """Read one Boolean field."""
            value = values.get(name, default)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            return value

        checkpoint_dir_value = values.get("checkpoint_dir", "checkpoints")
        if not isinstance(checkpoint_dir_value, str):
            raise ValueError("checkpoint_dir must be a string")

        return cls(
            backbone=cast(BackboneName, backbone_value),
            checkpoint_dir=checkpoint_dir_value,
            head_features=integer("head_features", 256),
            head_out_channels=cast(tuple[int, int, int, int], tuple(raw_head_channels)),
            predict_mask=boolean("predict_mask", False),
            scale_head_mode=cast(ScaleHeadMode, scale_mode_value),
            scale_head_num_queries=integer("scale_head_num_queries", 4),
            scale_head_num_heads=integer("scale_head_num_heads", 8),
            use_dwc=boolean("use_dwc", False),
            dwc_kernel_size=integer("dwc_kernel_size", 3),
            n_cam_types=integer("n_cam_types", 0),
            use_calib_tokens=boolean("use_calib_tokens", False),
            calib_tokens_per_type=integer("calib_tokens_per_type", 4),
            calib_token_inject_types=tuple(raw_inject_types),
            use_distortion_bias=boolean("use_distortion_bias", False),
            distortion_bias_layers=cast(DistortionBiasLayers, distortion_layers_value),
            distortion_bias_hidden_dim=integer("distortion_bias_hidden_dim", 64),
            distortion_bias_chunk_size=integer("distortion_bias_chunk_size", 1024),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> XLensArchitectureConfig:
        """Load and validate an architecture YAML file.

        Args:
            path: YAML configuration path.

        Returns:
            A validated architecture configuration.
        """
        loaded: object
        with Path(path).open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"X-Lens config must be a mapping, got {type(loaded).__name__}")
        return cls.from_mapping(cast(Mapping[str, object], loaded))


class XLensInference:
    """Strict checkpoint loader and inference facade for X-Lens."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
        amp_dtype: AmpDtype = "bf16",
        config: str | Path | None = None,
    ) -> None:
        """Load one released checkpoint.

        Args:
            checkpoint_path: Safetensors or PyTorch checkpoint path.
            device: Device that owns the model and inference inputs.
            amp_dtype: CUDA automatic-mixed-precision dtype.
            config: Optional architecture YAML. Bare safetensors require this file.
        """
        self.device: torch.device = torch.device(device)
        self.amp_dtype: torch.dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        checkpoint = Path(checkpoint_path)

        embedded_config: Mapping[str, object] = {}
        state: dict[str, Tensor]
        if checkpoint.suffix == ".safetensors":
            state = load_file(str(checkpoint))
        else:
            loaded: object = torch.load(checkpoint, map_location=self.device, weights_only=False)
            if not isinstance(loaded, Mapping):
                raise ValueError(f"X-Lens checkpoint must be a mapping, got {type(loaded).__name__}")
            loaded_mapping = cast(Mapping[str, object], loaded)
            raw_config = loaded_mapping.get("config", {})
            if isinstance(raw_config, Mapping):
                embedded_config = cast(Mapping[str, object], raw_config)
            raw_state = loaded_mapping.get("model_state_dict", loaded_mapping.get("model", loaded_mapping))
            if not isinstance(raw_state, Mapping) or not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in raw_state.items()):
                raise ValueError("X-Lens checkpoint does not contain a tensor state dict")
            state = cast(dict[str, Tensor], dict(raw_state))

        architecture = XLensArchitectureConfig.from_mapping(embedded_config)
        if config is not None:
            architecture = XLensArchitectureConfig.from_yaml(config)
        state = {key.removeprefix("module."): value for key, value in state.items()}

        token_tensor = state.get("backbone.pretrained.calib_tokens.tokens")
        if token_tensor is not None:
            architecture = replace(
                architecture,
                n_cam_types=token_tensor.shape[1],
                calib_tokens_per_type=token_tensor.shape[2],
            )

        model = XLensNet(
            backbone_name=architecture.backbone,
            checkpoint_dir=architecture.checkpoint_dir,
            head_features=architecture.head_features,
            head_out_channels=architecture.head_out_channels,
            predict_mask=architecture.predict_mask,
            scale_head_mode=architecture.scale_head_mode,
            scale_head_num_queries=architecture.scale_head_num_queries,
            scale_head_num_heads=architecture.scale_head_num_heads,
            use_dwc=architecture.use_dwc,
            dwc_kernel_size=architecture.dwc_kernel_size,
            n_cam_types=architecture.n_cam_types,
            use_calib_tokens=architecture.use_calib_tokens,
            calib_tokens_per_type=architecture.calib_tokens_per_type,
            calib_inject_types=architecture.calib_token_inject_types,
            use_distortion_bias=architecture.use_distortion_bias,
            distortion_bias_layers=architecture.distortion_bias_layers,
            distortion_bias_hidden_dim=architecture.distortion_bias_hidden_dim,
            distortion_bias_chunk_size=architecture.distortion_bias_chunk_size,
        ).to(self.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model: XLensNet = model
        logger.info(
            "loaded %s (%.1fM params, n_cam_types=%d)",
            checkpoint,
            sum(parameter.numel() for parameter in model.parameters()) / 1e6,
            architecture.n_cam_types,
        )

    @torch.no_grad()
    def __call__(self, batch: AssembledBatch) -> XLensOutput:
        """Run X-Lens and copy its numeric outputs to CPU float32.

        Args:
            batch: Model-ready tensors from ``assemble_batch``.

        Returns:
            Metric depth, confidence, scale, and optional mask tensors.
        """
        with torch.autocast("cuda", enabled=self.device.type == "cuda", dtype=self.amp_dtype):
            device_output = self.model(
                batch["images"],
                ray_map=batch["ray_map"],
                d_cam=batch["d_cam"],
                cam_types=batch["cam_types"],
            )
        return cast(XLensOutput, {key: value.detach().float().cpu() for key, value in device_output.items()})
