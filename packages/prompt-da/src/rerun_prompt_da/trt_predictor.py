"""Batched GPU predictors for Prompt Depth Anything (TensorRT and torch).

Both predictors share one tensor contract so they are interchangeable in
pipelines and parity tests: uint8 RGB ``[B,H,W,3]`` frames plus float32 metric
prompt depth ``[B,192,256]`` in, float32 metric depth ``[B,H,W]`` out, all
torch CUDA tensors. Preprocessing (resize to the 14-aligned network
resolution, [0,1] scaling) and postprocessing (resize back to the input
resolution) run on the GPU, so a video decoder can feed CUDA tensors straight
through without host round-trips.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float32, UInt8
from torch import Tensor

from rerun_prompt_da.trt_engine import (
    DEFAULT_CACHE_DIR,
    ModelType,
    TrtBuildConfig,
    ensure_engine,
    export_promptda_onnx,
)


def preprocess_batch(
    rgb_bhw3: UInt8[Tensor, "b h w 3"],
    prompt_depth_bhw: Float32[Tensor, "b 192 256"],
    image_hw: tuple[int, int],
) -> tuple[Float32[Tensor, "b 3 nh nw"], Float32[Tensor, "b 1 192 256"]]:
    """Prepare a uint8 RGB batch and metric prompt depth for the network.

    Args:
        rgb_bhw3: uint8 RGB frames, any resolution (moved to CUDA if needed).
        prompt_depth_bhw: float32 prompt depth in meters at the ARKit LiDAR resolution.
        image_hw: Static network (height, width), multiples of 14.

    Returns:
        float32 CUDA tensors: RGB ``[B,3,nh,nw]`` in [0,1] resized to
        ``image_hw``, and prompt depth ``[B,1,192,256]`` in meters.
    """
    rgb_b3hw: Float32[Tensor, "b 3 h w"] = rearrange(rgb_bhw3.to("cuda", non_blocking=True), "b h w c -> b c h w").float() / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    if tuple(rgb_b3hw.shape[-2:]) != image_hw:
        # antialias=True approximates the cv2 INTER_AREA downscale the torch demo uses.
        rgb_b3hw = F.interpolate(rgb_b3hw, size=image_hw, mode="bilinear", antialias=True)
    prompt_b1hw: Float32[Tensor, "b 1 192 256"] = rearrange(prompt_depth_bhw.to("cuda", non_blocking=True), "b h w -> b 1 h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    return rgb_b3hw, prompt_b1hw


def postprocess_depth(
    depth_b1hw: Float32[Tensor, "b 1 nh nw"],
    out_hw: tuple[int, int],
) -> Float32[Tensor, "b h w"]:
    """Resize network depth back to the input resolution.

    Args:
        depth_b1hw: float32 CUDA metric depth at the network resolution (may be
            a view into a reused runtime buffer).
        out_hw: Target (height, width) — the caller's input resolution.

    Returns:
        float32 CUDA metric depth at ``out_hw``, owning its memory — safe to
        hold across predictor calls.
    """
    if tuple(depth_b1hw.shape[-2:]) != out_hw:
        depth_b1hw = F.interpolate(depth_b1hw, size=out_hw, mode="bilinear", align_corners=False)
    else:
        depth_b1hw = depth_b1hw.clone()
    return rearrange(depth_b1hw, "b 1 h w -> b h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive


class PromptDATrtPredictor:
    """Batched PromptDA depth completion on a cached dynamic-batch TensorRT engine."""

    def __init__(
        self,
        model_type: ModelType = "large",
        image_hw: tuple[int, int] = (756, 1008),
        batch_size: int = 8,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        """Export ONNX and build/load the machine-local engine on first use.

        Args:
            model_type: PromptDA checkpoint variant.
            image_hw: Static network (height, width), multiples of 14.
            batch_size: Engine profile max and optimum batch.
            cache_dir: Cache root for ONNX and engine artifacts.
        """
        from trtkit.tensorrt_runtime import TensorRtRuntime

        onnx_path: Path = export_promptda_onnx(model_type=model_type, image_hw=image_hw, cache_dir=cache_dir)
        config = TrtBuildConfig(max_batch_size=batch_size, opt_batch_size=batch_size)
        engine_path: Path = ensure_engine(onnx_path, config, cache_dir=cache_dir / "trt")
        self.runtime = TensorRtRuntime(engine_path)
        self.image_hw: tuple[int, int] = image_hw

    def __call__(
        self,
        rgb_bhw3: UInt8[Tensor, "b h w 3"],
        prompt_depth_bhw: Float32[Tensor, "b 192 256"],
    ) -> Float32[Tensor, "b h w"]:
        """Complete metric depth for a batch of frames.

        Args:
            rgb_bhw3: uint8 RGB frames at any resolution.
            prompt_depth_bhw: float32 prompt depth in meters.

        Returns:
            float32 CUDA metric depth in meters at the input resolution
            (owning its memory — safe to hold across calls).
        """
        in_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        image_b3hw: Float32[Tensor, "b 3 nh nw"]
        prompt_b1hw: Float32[Tensor, "b 1 192 256"]
        image_b3hw, prompt_b1hw = preprocess_batch(rgb_bhw3, prompt_depth_bhw, self.image_hw)
        depth_b1hw: Float32[Tensor, "b 1 nh nw"] = self.runtime({"image": image_b3hw, "prompt_depth": prompt_b1hw})["depth"]
        return postprocess_depth(depth_b1hw, in_hw)


class PromptDATorchPredictor:
    """Plain-torch twin of :class:`PromptDATrtPredictor` for parity and baselines."""

    def __init__(
        self,
        model_type: ModelType = "large",
        image_hw: tuple[int, int] = (756, 1008),
    ) -> None:
        """Load the fp32 torch PromptDA network onto the GPU.

        Args:
            model_type: PromptDA checkpoint variant.
            image_hw: Static network (height, width), multiples of 14.
        """
        from monopriors.models.depth_completion.prompt_da import NAME_TO_HFNAME
        from monopriors.third_party.promptda.promptda import PromptDA

        self.model = PromptDA.from_pretrained(NAME_TO_HFNAME[model_type]).to("cuda").eval()
        self.image_hw: tuple[int, int] = image_hw

    def __call__(
        self,
        rgb_bhw3: UInt8[Tensor, "b h w 3"],
        prompt_depth_bhw: Float32[Tensor, "b 192 256"],
    ) -> Float32[Tensor, "b h w"]:
        """Complete metric depth for a batch of frames (same contract as TRT).

        Args:
            rgb_bhw3: uint8 RGB frames at any resolution.
            prompt_depth_bhw: float32 prompt depth in meters.

        Returns:
            float32 CUDA metric depth in meters at the input resolution.
        """
        in_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        image_b3hw: Float32[Tensor, "b 3 nh nw"]
        prompt_b1hw: Float32[Tensor, "b 1 192 256"]
        image_b3hw, prompt_b1hw = preprocess_batch(rgb_bhw3, prompt_depth_bhw, self.image_hw)
        with torch.inference_mode():
            depth_b1hw: Float32[Tensor, "b 1 nh nw"] = self.model(image_b3hw, prompt_b1hw)
        return postprocess_depth(depth_b1hw, in_hw)
