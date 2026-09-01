"""Backend-neutral family contract for metric depth-completion predictors."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Float32, UInt8, UInt16
from torch import Tensor
from trtkit import TensorRuntime, run_chunked

IMAGE_INPUT_NAME: str = "image"
"""Runtime binding for normalized RGB in BCHW layout."""
PROMPT_INPUT_NAME: str = "prompt_depth"
"""Runtime binding for the metric prompt in B1HW layout."""
DEPTH_OUTPUT_NAME: str = "depth"
"""Runtime binding for completed metric depth in B1HW layout."""

RuntimeT = TypeVar("RuntimeT", bound=TensorRuntime)


@dataclass
class CompletionDepthPrediction:
    """Legacy single-frame numpy prediction retained for downstream compatibility."""

    depth_mm: UInt16[np.ndarray, "h w"]
    """Metric depth in millimetres."""
    confidence: Float[np.ndarray, "h w"]
    """Per-pixel confidence values."""


def preprocess_completion_batch(
    rgb_bhw3: UInt8[Tensor, "b h w 3"],
    prompt_depth_bhw: Float32[Tensor, "b 192 256"],
    image_hw: tuple[int, int],
) -> tuple[Float32[Tensor, "b 3 nh nw"], Float32[Tensor, "b 1 192 256"]]:
    """Prepare one completion batch for any tensor runtime.

    Args:
        rgb_bhw3: uint8 CUDA RGB frames with shape ``(batch, height, width, 3)``.
        prompt_depth_bhw: float32 CUDA metric prompts in metres with shape
            ``(batch, 192, 256)``.
        image_hw: Static runtime image height and width.

    Returns:
        Float32 image ``(batch, 3, image_height, image_width)`` in ``[0, 1]``
        and metric prompt ``(batch, 1, 192, 256)`` on the input device.
    """
    image_b3hw: Float32[Tensor, "b 3 h w"] = rearrange(rgb_bhw3, "b h w c -> b c h w").to(dtype=torch.float32) / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    if tuple(int(dim) for dim in image_b3hw.shape[-2:]) != image_hw:
        image_b3hw = F.interpolate(image_b3hw, size=image_hw, mode="bilinear", antialias=True)
    prompt_b1hw: Float32[Tensor, "b 1 192 256"] = rearrange(prompt_depth_bhw, "b h w -> b 1 h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    return image_b3hw.contiguous(), prompt_b1hw.contiguous()


def postprocess_completion_depth(
    depth_b1hw: Float32[Tensor, "b 1 nh nw"],
    output_hw: tuple[int, int],
) -> Float32[Tensor, "b h w"]:
    """Resize runtime depth to the caller's image resolution.

    Args:
        depth_b1hw: float32 CUDA metric depth with shape
            ``(batch, 1, network_height, network_width)``.
        output_hw: Caller image height and width.

    Returns:
        Owning float32 CUDA metric depth with shape
        ``(batch, output_height, output_width)``.
    """
    if tuple(int(dim) for dim in depth_b1hw.shape[-2:]) != output_hw:
        depth_b1hw = F.interpolate(depth_b1hw, size=output_hw, mode="bilinear", align_corners=False)
    else:
        depth_b1hw = depth_b1hw.clone()
    return rearrange(depth_b1hw, "b 1 h w -> b h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive


class BaseCompletionPredictor(Generic[RuntimeT]):
    """One batched GPU tensor contract over torch, ONNX, and TensorRT."""

    def __init__(self, runtime: RuntimeT) -> None:
        """Store the runtime and derive its static image resolution.

        Args:
            runtime: Tensor runtime with the shared image, prompt, and depth
                binding names.
        """
        self.runtime: RuntimeT = runtime
        image_spec = next(spec for spec in runtime.spec.inputs if spec.name == IMAGE_INPUT_NAME)
        self.image_hw: tuple[int, int] = (image_spec.shape[1], image_spec.shape[2])

    @torch.inference_mode()
    def __call__(
        self,
        rgb_bhw3: UInt8[Tensor, "b h w 3"],
        prompt_depth_bhw: Float32[Tensor, "b 192 256"],
    ) -> Float32[Tensor, "b h w"]:
        """Complete metric depth for a batch of RGB frames and raw prompts.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames with shape
                ``(batch, height, width, 3)``.
            prompt_depth_bhw: float32 CUDA raw prompt depth in metres with
                shape ``(batch, 192, 256)``.

        Returns:
            Owning float32 CUDA metric depth in metres with shape
            ``(batch, height, width)``.
        """
        output_hw: tuple[int, int] = (int(rgb_bhw3.shape[1]), int(rgb_bhw3.shape[2]))
        image_b3hw: Float32[Tensor, "b 3 nh nw"]
        prompt_b1hw: Float32[Tensor, "b 1 192 256"]
        image_b3hw, prompt_b1hw = preprocess_completion_batch(rgb_bhw3, prompt_depth_bhw, self.image_hw)
        outputs: dict[str, Tensor] = run_chunked(
            self.runtime,
            {IMAGE_INPUT_NAME: image_b3hw, PROMPT_INPUT_NAME: prompt_b1hw},
        )
        depth_b1hw: Float32[Tensor, "b 1 nh nw"] = outputs[DEPTH_OUTPUT_NAME]
        return postprocess_completion_depth(depth_b1hw, output_hw)


__all__ = (
    "BaseCompletionPredictor",
    "CompletionDepthPrediction",
    "DEPTH_OUTPUT_NAME",
    "IMAGE_INPUT_NAME",
    "PROMPT_INPUT_NAME",
    "postprocess_completion_depth",
    "preprocess_completion_batch",
)
