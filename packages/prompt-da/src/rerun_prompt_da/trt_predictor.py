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
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float32, UInt8
from torch import Tensor

from rerun_prompt_da.trt_engine import (
    DEFAULT_CACHE_DIR,
    PROMPT_DEPTH_HW,
    ModelType,
    TrtBuildConfig,
    TrtPrecision,
    _import_tensorrt,
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
    rgb_b3hw: Float32[Tensor, "b 3 h w"] = rearrange(rgb_bhw3.to("cuda", non_blocking=True), "b h w c -> b c h w").float() / 255.0
    if tuple(rgb_b3hw.shape[-2:]) != image_hw:
        # antialias=True approximates the cv2 INTER_AREA downscale the torch demo uses.
        rgb_b3hw = F.interpolate(rgb_b3hw, size=image_hw, mode="bilinear", antialias=True)
    prompt_b1hw: Float32[Tensor, "b 1 192 256"] = rearrange(prompt_depth_bhw.to("cuda", non_blocking=True), "b h w -> b 1 h w")
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
    return rearrange(depth_b1hw, "b 1 h w -> b h w")


class PromptDATrtRuntime:
    """A deserialized PromptDA engine with persistent torch-tensor I/O.

    Buffers are allocated once at the profile's max batch and bound via
    ``set_tensor_address``; each call copies inputs in, runs the true batch
    size through ``execute_async_v3`` on a dedicated stream (ordered against
    torch's current stream on both sides), and returns a view into the output
    buffer sliced to the submitted batch. The view is overwritten by the next
    call — clone it if it must survive.
    """

    def __init__(self, engine_path: Path) -> None:
        """Deserialize a machine-local engine and bind persistent I/O buffers.

        Args:
            engine_path: Engine built by :func:`rerun_prompt_da.trt_engine.ensure_engine`.

        Raises:
            RuntimeError: If CUDA is unavailable or the engine fails to load.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT execution requires CUDA.")
        trt: Any = _import_tensorrt()
        engine: Any = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_path.expanduser().read_bytes())
        context: Any = None if engine is None else engine.create_execution_context()
        if engine is None or context is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        self._engine: Any = engine
        self._context: Any = context
        self._device: torch.device = torch.device("cuda")
        image_shape: tuple[int, ...] = tuple(int(dim) for dim in engine.get_tensor_shape("image"))
        prompt_shape: tuple[int, ...] = tuple(int(dim) for dim in engine.get_tensor_shape("prompt_depth"))
        depth_shape: tuple[int, ...] = tuple(int(dim) for dim in engine.get_tensor_shape("depth"))
        profile_max: tuple[int, ...] = tuple(int(dim) for dim in engine.get_tensor_profile_shape("image", 0)[2])
        self.max_batch_size: int = profile_max[0]
        self.image_hw: tuple[int, int] = (image_shape[2], image_shape[3])
        # Persistent buffers: stable addresses are required for the one-time
        # set_tensor_address binding.
        self._buffers: dict[str, Tensor] = {
            "image": torch.zeros((self.max_batch_size, *image_shape[1:]), dtype=torch.float32, device=self._device),
            "prompt_depth": torch.zeros((self.max_batch_size, *prompt_shape[1:]), dtype=torch.float32, device=self._device),
            "depth": torch.empty((self.max_batch_size, *depth_shape[1:]), dtype=torch.float32, device=self._device),
        }
        for name, tensor in self._buffers.items():
            self._context.set_tensor_address(name, int(tensor.data_ptr()))
        self._active_batch: int = -1
        self._stream: torch.cuda.Stream = torch.cuda.Stream(device=self._device)

    def __call__(
        self,
        image_b3hw: Float32[Tensor, "b 3 h w"],
        prompt_depth_b1hw: Float32[Tensor, "b 1 192 256"],
    ) -> Float32[Tensor, "b 1 h w"]:
        """Run one batch at its true size through the dynamic engine.

        Args:
            image_b3hw: float32 CUDA RGB batch in [0,1] at the engine resolution.
            prompt_depth_b1hw: float32 CUDA prompt depth in meters.

        Returns:
            float32 CUDA metric depth, a view into the reused output buffer
            sliced to the submitted batch size.

        Raises:
            ValueError: If the batch exceeds the engine's profile max.
            RuntimeError: If TensorRT reports a failed launch.
        """
        batch_size: int = image_b3hw.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch {batch_size} exceeds the engine's max batch {self.max_batch_size}; chunk the input.")
        self._buffers["image"][:batch_size].copy_(image_b3hw)
        self._buffers["prompt_depth"][:batch_size].copy_(prompt_depth_b1hw)
        if batch_size != self._active_batch:
            self._context.set_input_shape("image", (batch_size, 3, *self.image_hw))
            self._context.set_input_shape("prompt_depth", (batch_size, 1, *PROMPT_DEPTH_HW))
            self._active_batch = batch_size
        current: torch.cuda.Stream = torch.cuda.current_stream(self._device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            ok: bool = bool(self._context.execute_async_v3(stream_handle=int(self._stream.cuda_stream)))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        current.wait_stream(self._stream)
        return self._buffers["depth"][:batch_size]


class PromptDATrtPredictor:
    """Batched PromptDA depth completion on a cached dynamic-batch TensorRT engine."""

    def __init__(
        self,
        model_type: ModelType = "large",
        image_hw: tuple[int, int] = (756, 1008),
        batch_size: int = 8,
        precision: TrtPrecision = "fp16",
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        """Export ONNX and build/load the machine-local engine on first use.

        Args:
            model_type: PromptDA checkpoint variant.
            image_hw: Static network (height, width), multiples of 14.
            batch_size: Engine profile max and optimum batch.
            precision: TensorRT builder precision.
            cache_dir: Cache root for ONNX and engine artifacts.
        """
        onnx_path: Path = export_promptda_onnx(model_type=model_type, image_hw=image_hw, cache_dir=cache_dir)
        config = TrtBuildConfig(max_batch_size=batch_size, opt_batch_size=batch_size, precision=precision)
        engine_path: Path = ensure_engine(onnx_path, config, cache_dir=cache_dir)
        self.runtime = PromptDATrtRuntime(engine_path)
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
        depth_b1hw: Float32[Tensor, "b 1 nh nw"] = self.runtime(image_b3hw, prompt_b1hw)
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
