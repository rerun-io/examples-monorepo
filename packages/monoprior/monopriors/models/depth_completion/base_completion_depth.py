"""Backend-neutral family contract for metric depth-completion predictors."""

from dataclasses import dataclass

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
PROMPT_DEPTH_HW: tuple[int, int] = (192, 256)
"""ARKit LiDAR prompt-depth resolution used by completion models."""
MIN_METRIC_DEPTH_M: float = 0.1
"""Lower edge of the shared metric-depth validity window, in metres."""
MAX_METRIC_DEPTH_M: float = 4.0
"""Upper edge of the shared metric-depth validity window, in metres."""


@dataclass
class CompletionDepthPrediction:
    """Legacy single-frame numpy prediction retained for downstream compatibility."""

    depth_mm: UInt16[np.ndarray, "h w"]
    """Metric depth in millimetres."""
    confidence: Float[np.ndarray, "h w"]
    """Per-pixel confidence values."""


def preprocess_completion_batch(
    rgb_bhwc: UInt8[Tensor, "b h w 3"],
    prompt_depth_bhw: Float32[Tensor, "b 192 256"],
    image_hw: tuple[int, int],
) -> tuple[Float32[Tensor, "b 3 nh nw"], Float32[Tensor, "b 1 192 256"]]:
    """Prepare one completion batch for any tensor runtime.

    Args:
        rgb_bhwc: uint8 CUDA RGB frames with shape ``(batch, height, width, 3)``.
        prompt_depth_bhw: float32 CUDA metric prompts in metres with shape
            ``(batch, 192, 256)``.
        image_hw: Static runtime image height and width.

    Returns:
        Float32 image ``(batch, 3, image_height, image_width)`` in ``[0, 1]``
        and metric prompt ``(batch, 1, 192, 256)`` on the input device.
    """
    image_bchw: Float32[Tensor, "b 3 h w"] = rearrange(rgb_bhwc, "b h w c -> b c h w").to(
        dtype=torch.float32,
        memory_format=torch.contiguous_format,
    ).div_(255.0)
    if tuple(int(dim) for dim in image_bchw.shape[-2:]) != image_hw:
        image_bchw = F.interpolate(image_bchw, size=image_hw, mode="bilinear", antialias=True)
    prompt_bchw: Float32[Tensor, "b 1 192 256"] = rearrange(prompt_depth_bhw, "b h w -> b 1 h w")
    return image_bchw.contiguous(), prompt_bchw.contiguous()


def upsample_depth_to_image(
    depth_bchw: Float32[Tensor, "b 1 dh dw"],
    image_hw: tuple[int, int],
) -> Float32[Tensor, "b h w"]:
    """Return owned BHW depth, using bilinear upsampling with ``align_corners=False`` and no antialiasing when the image size differs."""
    if tuple(int(dim) for dim in depth_bchw.shape[-2:]) != image_hw:
        depth_bchw = F.interpolate(depth_bchw, size=image_hw, mode="bilinear", align_corners=False, antialias=False)
    else:
        depth_bchw = depth_bchw.clone()
    return rearrange(depth_bchw, "b 1 h w -> b h w")


class BaseCompletionPredictor:
    """One batched GPU tensor contract over torch, ONNX, and TensorRT."""

    def __init__(self, runtime: TensorRuntime) -> None:
        """Store the runtime and derive its static image resolution.

        Args:
            runtime: Tensor runtime with the shared image, prompt, and depth
                binding names.
        """
        self.runtime: TensorRuntime = runtime
        image_spec = next(spec for spec in runtime.spec.inputs if spec.name == IMAGE_INPUT_NAME)
        self.image_hw: tuple[int, int] = (image_spec.shape[1], image_spec.shape[2])

    @torch.inference_mode()
    def __call__(
        self,
        rgb_bhwc: UInt8[Tensor, "b h w 3"],
        prompt_depth_bhw: Float32[Tensor, "b 192 256"],
    ) -> Float32[Tensor, "b h w"]:
        """Complete metric depth for a batch of RGB frames and raw prompts.

        Args:
            rgb_bhwc: uint8 CUDA RGB frames with shape
                ``(batch, height, width, 3)``.
            prompt_depth_bhw: float32 CUDA raw prompt depth in metres with
                shape ``(batch, 192, 256)``.

        Returns:
            Owning float32 CUDA metric depth in metres with shape
            ``(batch, height, width)``.
        """
        output_hw: tuple[int, int] = (int(rgb_bhwc.shape[1]), int(rgb_bhwc.shape[2]))
        image_bchw: Float32[Tensor, "b 3 nh nw"]
        prompt_bchw: Float32[Tensor, "b 1 192 256"]
        image_bchw, prompt_bchw = preprocess_completion_batch(rgb_bhwc, prompt_depth_bhw, self.image_hw)
        outputs: dict[str, Tensor] = run_chunked(
            self.runtime,
            {IMAGE_INPUT_NAME: image_bchw, PROMPT_INPUT_NAME: prompt_bchw},
        )
        depth_bchw: Float32[Tensor, "b 1 nh nw"] = outputs[DEPTH_OUTPUT_NAME]
        return upsample_depth_to_image(depth_bchw, output_hw)


__all__ = (
    "BaseCompletionPredictor",
    "CompletionDepthPrediction",
    "DEPTH_OUTPUT_NAME",
    "IMAGE_INPUT_NAME",
    "MAX_METRIC_DEPTH_M",
    "MIN_METRIC_DEPTH_M",
    "PROMPT_DEPTH_HW",
    "PROMPT_INPUT_NAME",
    "preprocess_completion_batch",
    "upsample_depth_to_image",
)
