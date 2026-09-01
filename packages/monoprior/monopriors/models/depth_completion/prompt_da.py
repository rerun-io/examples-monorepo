"""Prompt Depth Anything model config and backend-neutral predictor."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import torch
from jaxtyping import Float32, UInt8
from torch import Tensor
from trtkit import (
    BackendConfig,
    TensorRtBackendConfig,
    TensorRuntime,
    TensorSpec,
    TorchBackendConfig,
    TorchRuntime,
    create_runtime_from_onnx,
)

from monopriors.models.depth_completion.base_completion_depth import (
    DEPTH_OUTPUT_NAME,
    IMAGE_INPUT_NAME,
    PROMPT_INPUT_NAME,
    BaseCompletionPredictor,
    postprocess_completion_depth,
    preprocess_completion_batch,
)

ModelType: TypeAlias = Literal["large", "small", "small-transparent"]

NAME_TO_HFNAME: dict[ModelType, str] = {
    "large": "depth-anything/prompt-depth-anything-vitl",
    "small": "depth-anything/prompt-depth-anything-vits",
    "small-transparent": "depth-anything/prompt-depth-anything-vits-transparent",
}

DEFAULT_PROMPTDA_CACHE_DIR: Path = Path(os.environ.get("PROMPTDA_TRT_CACHE", "~/.cache/prompt-da")).expanduser()
"""Cache root for portable PromptDA ONNX graphs and machine-local engines."""


def ensure_multiple_of(x: float, multiple_of: int = 14) -> int:
    """Round a size down to the nearest model patch multiple.

    Args:
        x: Size to align.
        multiple_of: Required alignment.

    Returns:
        Largest aligned integer no greater than ``x``.
    """
    return int(x // multiple_of * multiple_of)


def network_image_hw(capture_hw: tuple[int, int], max_image_size: int) -> tuple[int, int]:
    """Scale a capture to the PromptDA limit and align both sides to 14.

    Args:
        capture_hw: Capture height and width.
        max_image_size: Longest allowed network side.

    Returns:
        Network height and width, each divisible by 14.
    """
    height, width = capture_hw
    scale: float = min(1.0, max_image_size / max(height, width))
    return ensure_multiple_of(height * scale), ensure_multiple_of(width * scale)


@dataclass(frozen=True, slots=True)
class PromptDAConfig:
    """Prompt Depth Anything model and runtime configuration."""

    model_type: ModelType = "large"
    """PromptDA checkpoint variant."""
    backend: BackendConfig = field(default_factory=TorchBackendConfig)
    """Backend running the network: torch, ONNX Runtime CUDA, or TensorRT."""
    image_height: int = 756
    """Static network image height; must be divisible by 14."""
    image_width: int = 1008
    """Static network image width; must be divisible by 14."""
    cache_dir: Path = field(default_factory=lambda: DEFAULT_PROMPTDA_CACHE_DIR)
    """Portable ONNX cache root; TensorRT uses the backend's engine cache."""

    def setup(self) -> "PromptDAPredictor":
        """Build the selected runtime and return a ready predictor."""
        image_hw: tuple[int, int] = (self.image_height, self.image_width)
        image_spec = TensorSpec(IMAGE_INPUT_NAME, (3, *image_hw), torch.float32)
        prompt_spec = TensorSpec(PROMPT_INPUT_NAME, (1, 192, 256), torch.float32)
        depth_spec = TensorSpec(DEPTH_OUTPUT_NAME, (1, *image_hw), torch.float32)
        if isinstance(self.backend, TorchBackendConfig):
            from monopriors.third_party.promptda.promptda import PromptDA

            model = PromptDA.from_pretrained(NAME_TO_HFNAME[self.model_type]).to("cuda").eval()
            runtime: TensorRuntime = TorchRuntime(
                model,
                input_specs=(image_spec, prompt_spec),
                output_specs=(depth_spec,),
                max_batch_size=self.backend.max_batch_size,
                autocast_dtype=self.backend.autocast_dtype,
            )
        else:
            from monopriors.models.depth_completion.prompt_da_export import export_promptda_onnx

            onnx_path: Path = export_promptda_onnx(model_type=self.model_type, image_hw=image_hw, cache_dir=self.cache_dir)
            runtime = create_runtime_from_onnx(onnx_path, self.backend)
        return PromptDAPredictor(runtime=runtime)


class PromptDAPredictor(BaseCompletionPredictor[TensorRuntime]):
    """PromptDA completion through the shared batched GPU tensor contract."""


def preprocess_batch(
    rgb_bhw3: UInt8[Tensor, "b h w 3"],
    prompt_depth_bhw: Float32[Tensor, "b 192 256"],
    image_hw: tuple[int, int],
) -> tuple[Float32[Tensor, "b 3 nh nw"], Float32[Tensor, "b 1 192 256"]]:
    """Compatibility wrapper that retains the historical host-to-CUDA move."""
    return preprocess_completion_batch(
        rgb_bhw3.to("cuda", non_blocking=True),
        prompt_depth_bhw.to("cuda", non_blocking=True),
        image_hw,
    )


def postprocess_depth(
    depth_b1hw: Float32[Tensor, "b 1 nh nw"],
    out_hw: tuple[int, int],
) -> Float32[Tensor, "b h w"]:
    """Compatibility name for shared completion postprocessing."""
    return postprocess_completion_depth(depth_b1hw, out_hw)


class PromptDATrtPredictor(PromptDAPredictor):
    """Compatibility constructor for a TensorRT PromptDA predictor."""

    def __init__(
        self,
        model_type: ModelType = "large",
        image_hw: tuple[int, int] = (756, 1008),
        batch_size: int = 8,
        cache_dir: Path = DEFAULT_PROMPTDA_CACHE_DIR,
    ) -> None:
        """Build or load the historical dynamic-batch TensorRT runtime.

        Args:
            model_type: PromptDA checkpoint variant.
            image_hw: Static network image height and width.
            batch_size: TensorRT profile maximum and optimum batch.
            cache_dir: PromptDA ONNX and engine cache root.
        """
        configured: PromptDAPredictor = PromptDAConfig(
            model_type=model_type,
            backend=TensorRtBackendConfig(
                max_batch_size=batch_size,
                opt_batch_size=batch_size,
                cache_dir=cache_dir / "trt",
            ),
            image_height=image_hw[0],
            image_width=image_hw[1],
            cache_dir=cache_dir,
        ).setup()
        super().__init__(configured.runtime)


class PromptDATorchPredictor(PromptDAPredictor):
    """Compatibility constructor for an eager fp32 PromptDA predictor."""

    def __init__(
        self,
        model_type: ModelType = "large",
        image_hw: tuple[int, int] = (756, 1008),
    ) -> None:
        """Load the historical eager PromptDA baseline.

        Args:
            model_type: PromptDA checkpoint variant.
            image_hw: Static network image height and width.
        """
        configured: PromptDAPredictor = PromptDAConfig(
            model_type=model_type,
            backend=TorchBackendConfig(autocast="fp32"),
            image_height=image_hw[0],
            image_width=image_hw[1],
        ).setup()
        super().__init__(configured.runtime)


__all__ = (
    "DEFAULT_PROMPTDA_CACHE_DIR",
    "ModelType",
    "NAME_TO_HFNAME",
    "PromptDAConfig",
    "PromptDAPredictor",
    "PromptDATorchPredictor",
    "PromptDATrtPredictor",
    "ensure_multiple_of",
    "network_image_hw",
    "postprocess_depth",
    "preprocess_batch",
)
