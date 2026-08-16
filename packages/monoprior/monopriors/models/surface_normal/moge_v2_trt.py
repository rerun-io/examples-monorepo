"""Batched GPU predictors for MoGe v2 camera-space surface normals.

The torch and TensorRT predictors share one tensor contract: uint8 CUDA RGB
frames shaped ``[B,H,W,3]`` in, then float32 camera-space RDF normals shaped
``[B,H,W,3]`` and float32 validity probabilities shaped ``[B,H,W]`` out.
Normals point toward the camera, matching :class:`MoGeV2NormalPredictor`.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Float32, UInt8
from torch import Tensor

from monopriors.models.surface_normal.moge_v2 import MOGE_V2_NORMAL_CHECKPOINTS
from monopriors.third_party.moge.model.v2 import ForwardOutput, MoGeModel

Encoder: TypeAlias = Literal["vits", "vitb", "vitl"]

DEFAULT_CACHE_DIR: Path = Path(os.environ.get("MONOPRIOR_TRT_CACHE", "~/.cache/monoprior")).expanduser()
"""Cache root holding portable ``onnx/`` and machine-local ``trt/`` artifacts."""

DEFAULT_IMAGE_HW: tuple[int, int] = (756, 1008)
"""ARKitScenes-aligned MoGe network height and width."""

MOGE_V2_NUM_TOKENS_RANGE: tuple[int, int] = (1200, 3600)
"""Training-time token range shared by all pinned MoGe v2 normal checkpoints."""

ONNX_EXPORT_VERSION: int = 1
"""Cache-breaking version of the MoGe v2 normal ONNX export recipe."""

_EXPORT_WORKER_ENV: str = "MONOPRIOR_MOGE_V2_ONNX_EXPORT_WORKER"


class MoGeV2NormalOutput(NamedTuple):
    """Owning batched normal and mask tensors returned by both predictors."""

    normals_bhw3: Float32[Tensor, "b h w 3"]
    mask_bhw: Float32[Tensor, "b h w"]


class _MoGeV2NormalHeads(torch.nn.Module):
    """Fixed-token adapter exposing the normal and mask heads as tuple outputs."""

    def __init__(self, model: MoGeModel, num_tokens: int) -> None:
        super().__init__()
        self.model: MoGeModel = model
        self.num_tokens: int = num_tokens

    def forward(
        self,
        image_b3hw: Float[Tensor, "b 3 h w"],
    ) -> tuple[Float[Tensor, "b h w 3"], Float[Tensor, "b h w"]]:
        """Run only the two heads required by the surface-normal pipeline.

        Args:
            image_b3hw: Float RGB batch shaped ``b 3 h w`` in ``[0, 1]``.

        Returns:
            Camera-space unit normals and mask probabilities.
        """
        output: ForwardOutput = self.model(
            image_b3hw,
            num_tokens=self.num_tokens,
            output_heads=("normal", "mask"),
        )
        normals_bhw3: Float[Tensor, "b h w 3"] = cast(Tensor, output["normal"])
        mask_bhw: Float[Tensor, "b h w"] = cast(Tensor, output["mask"])
        return normals_bhw3, mask_bhw


def _resolve_num_tokens(resolution_level: int) -> int:
    """Map a MoGe resolution level to the pinned checkpoint token range.

    Args:
        resolution_level: Detail level from 0 through 9.

    Returns:
        Fixed token count for export and inference.

    Raises:
        ValueError: If the resolution level is outside 0 through 9.
    """
    if not 0 <= resolution_level <= 9:
        raise ValueError(f"resolution_level must be within [0, 9], got {resolution_level}.")
    min_tokens: int = MOGE_V2_NUM_TOKENS_RANGE[0]
    max_tokens: int = MOGE_V2_NUM_TOKENS_RANGE[1]
    return int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))


def export_moge_v2_normal_onnx(
    encoder: Encoder = "vitl",
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    resolution_level: int = 9,
    max_batch_size: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Export a pinned MoGe v2 normal checkpoint with dynamic batch only.

    The graph accepts float32 RGB ``[B,3,H,W]`` in ``[0,1]`` and returns
    float32 normals ``[B,H,W,3]`` plus masks ``[B,H,W]``. Compute is fp16,
    image height, image width, and token count are static, and only batch is
    dynamic. The cache identity includes all three static model dimensions,
    this export recipe version, and the immutable checkpoint revision.

    Args:
        encoder: DINOv2 encoder size.
        image_hw: Static network height and width.
        resolution_level: Detail level from 0 through 9.
        max_batch_size: Largest batch encoded in the dynamic ONNX constraint.
        cache_dir: Cache root; the graph lands in ``cache_dir / "onnx"``.

    Returns:
        Cached or newly exported ONNX path.

    Raises:
        RuntimeError: If CUDA is unavailable.
        ValueError: If dimensions, batch size, or checkpoint token metadata are invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("MoGe v2 ONNX export requires CUDA.")
    height: int = image_hw[0]
    width: int = image_hw[1]
    if height < 1 or width < 1:
        raise ValueError(f"MoGe v2 network dimensions must be positive, got {image_hw}.")
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")

    num_tokens: int = _resolve_num_tokens(resolution_level)
    checkpoint: tuple[str, str] = MOGE_V2_NORMAL_CHECKPOINTS[encoder]
    checkpoint_revision: str = checkpoint[1][:8]
    onnx_dir: Path = cache_dir / "onnx"
    onnx_path: Path = onnx_dir / (
        f"moge-v2-{encoder}-normal_{height}x{width}_t{num_tokens}_v{ONNX_EXPORT_VERSION}_{checkpoint_revision}.onnx"
    )
    if onnx_path.exists():
        return onnx_path

    # Beartype's dev import hook checks annotated locals inside the vendored
    # forward. Dynamic ONNX export represents the batch as torch.SymInt, which
    # correctly violates those eager-only ``int`` checks. Re-enter this exact
    # export in a clean interpreter without dev instrumentation; the parent
    # process keeps beartype active for every predictor call and test.
    if os.environ.get("PIXI_DEV_MODE") == "1" and os.environ.get(_EXPORT_WORKER_ENV) != "1":
        worker_env: dict[str, str] = dict(os.environ)
        worker_env["PIXI_DEV_MODE"] = "0"
        worker_env[_EXPORT_WORKER_ENV] = "1"
        worker_command: list[str] = [
            sys.executable,
            "-m",
            "monopriors.models.surface_normal._moge_v2_onnx_worker",
            encoder,
            str(height),
            str(width),
            str(resolution_level),
            str(max_batch_size),
            str(cache_dir),
        ]
        subprocess.run(worker_command, check=True, env=worker_env)
        if not onnx_path.exists():
            raise RuntimeError(f"MoGe v2 ONNX export worker did not produce {onnx_path}.")
        return onnx_path

    print(f"[monoprior] exporting MoGe v2 normals to ONNX (one-time, may take minutes): {onnx_path.name}")
    model: MoGeModel = MoGeModel.from_pretrained(checkpoint[0], revision=checkpoint[1]).to("cuda").eval()
    if model.num_tokens_range != MOGE_V2_NUM_TOKENS_RANGE:
        raise ValueError(
            f"Pinned MoGe v2 checkpoint token range changed from {MOGE_V2_NUM_TOKENS_RANGE} to {model.num_tokens_range}; bump the export recipe."
        )
    model.onnx_compatible_mode = True
    wrapper: _MoGeV2NormalHeads = _MoGeV2NormalHeads(model, num_tokens).eval()
    example_batch_size: int = min(2, max_batch_size)
    dummy_image_b3hw: Float32[Tensor, "b 3 h w"] = torch.zeros(
        (example_batch_size, 3, height, width),
        dtype=torch.float32,
        device="cuda",
    )

    from trtkit import export_onnx, sweep_stale_onnx_exports

    export_onnx(
        wrapper,
        (dummy_image_b3hw,),
        onnx_path,
        input_names=["image"],
        output_names=["normal", "mask"],
        compute_dtype=torch.float16,
        dynamic_batch_max=max_batch_size,
    )
    del dummy_image_b3hw, wrapper, model
    torch.cuda.empty_cache()

    # Preserve the current external-data sidecar. trtkit publishes the ONNX
    # protobuf atomically, while dynamo keeps its large weights in the
    # pid-suffixed sidecar referenced by that protobuf.
    current_sidecar: Path = onnx_path.with_name(f"{onnx_path.name}.part{os.getpid()}.data")
    stale_prefix: str = f"moge-v2-{encoder}-normal_{height}x{width}_t{num_tokens}_"
    sweep_stale_onnx_exports(onnx_dir, stale_prefix, keep_paths={onnx_path, current_sidecar})
    return onnx_path


def _preprocess_rgb(
    rgb_bhw3: UInt8[Tensor, "b h w 3"],
    image_hw: tuple[int, int],
) -> Float32[Tensor, "b 3 nh nw"]:
    """Convert a CUDA uint8 RGB batch to fixed-resolution float network input.

    Args:
        rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.
        image_hw: Static network height and width.

    Returns:
        float32 CUDA RGB tensor shaped ``b 3 nh nw`` in ``[0, 1]``.

    Raises:
        ValueError: If the batch is empty or the network size is invalid.
        RuntimeError: If the input is not already CUDA-resident.
    """
    if not rgb_bhw3.is_cuda:
        raise RuntimeError("MoGe v2 batched normal predictors require CUDA input tensors.")
    if rgb_bhw3.shape[0] < 1:
        raise ValueError("MoGe v2 batched normal predictors require a non-empty batch.")
    if image_hw[0] < 1 or image_hw[1] < 1:
        raise ValueError(f"MoGe v2 network dimensions must be positive, got {image_hw}.")

    image_b3hw: Float32[Tensor, "b 3 h w"] = rearrange(rgb_bhw3, "b h w c -> b c h w").to(dtype=torch.float32) / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    if tuple(image_b3hw.shape[-2:]) != image_hw:
        image_b3hw = F.interpolate(image_b3hw, size=image_hw, mode="bilinear", antialias=True)
    return image_b3hw


def _postprocess_normal_output(
    normals_bnhw3: Float32[Tensor, "b nh nw 3"],
    mask_bnhw: Float32[Tensor, "b nh nw"],
    output_hw: tuple[int, int],
) -> MoGeV2NormalOutput:
    """Restore input resolution, unit-normalize, and take ownership of outputs.

    Args:
        normals_bnhw3: float32 camera-space normals at network resolution.
        mask_bnhw: float32 mask probabilities at network resolution.
        output_hw: Caller input height and width.

    Returns:
        Owning float32 normal and mask tensors at ``output_hw``.
    """
    normals_b3nhw: Float32[Tensor, "b 3 nh nw"] = rearrange(normals_bnhw3.float(), "b h w c -> b c h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    mask_b1nhw: Float32[Tensor, "b 1 nh nw"] = rearrange(mask_bnhw.float(), "b h w -> b 1 h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    if tuple(normals_b3nhw.shape[-2:]) != output_hw:
        normals_b3hw: Float32[Tensor, "b 3 h w"] = F.interpolate(normals_b3nhw, size=output_hw, mode="bilinear", align_corners=False)
        mask_b1hw: Float32[Tensor, "b 1 h w"] = F.interpolate(mask_b1nhw, size=output_hw, mode="bilinear", align_corners=False)
    else:
        normals_b3hw = normals_b3nhw.clone()
        mask_b1hw = mask_b1nhw.clone()
    normals_bhw3: Float32[Tensor, "b h w 3"] = F.normalize(
        rearrange(normals_b3hw, "b c h w -> b h w c"),  # pyrefly: ignore  # bad-argument-type — einops stub false positive
        dim=-1,
    )
    mask_bhw: Float32[Tensor, "b h w"] = rearrange(mask_b1hw, "b 1 h w -> b h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    return MoGeV2NormalOutput(normals_bhw3=normals_bhw3, mask_bhw=mask_bhw)


class MoGeV2TorchNormalPredictor:
    """Plain-torch fp16-compute twin of the batched TensorRT predictor."""

    def __init__(
        self,
        encoder: Encoder = "vitl",
        image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
        resolution_level: int = 9,
    ) -> None:
        """Load a pinned MoGe v2 normal checkpoint onto CUDA.

        Args:
            encoder: DINOv2 encoder size.
            image_hw: Static network height and width.
            resolution_level: Detail level from 0 through 9; level 9 uses
                3600 tokens for the pinned checkpoints.

        Raises:
            RuntimeError: If CUDA is unavailable.
            ValueError: If ``resolution_level`` is outside 0 through 9.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("MoGe v2 batched normal prediction requires CUDA.")
        num_tokens: int = _resolve_num_tokens(resolution_level)
        checkpoint: tuple[str, str] = MOGE_V2_NORMAL_CHECKPOINTS[encoder]
        self.model: MoGeModel = MoGeModel.from_pretrained(checkpoint[0], revision=checkpoint[1]).to("cuda").eval()
        if self.model.num_tokens_range != MOGE_V2_NUM_TOKENS_RANGE:
            raise ValueError(
                f"Pinned MoGe v2 checkpoint token range changed from {MOGE_V2_NUM_TOKENS_RANGE} to {self.model.num_tokens_range}; update the predictor."
            )
        self.model.onnx_compatible_mode = True
        self.num_tokens: int = num_tokens
        self.image_hw: tuple[int, int] = image_hw

    def __call__(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2NormalOutput:
        """Predict camera-space RDF normals and mask probabilities for a batch.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.

        Returns:
            Owning float32 unit normals shaped ``b h w 3`` and float32 masks
            shaped ``b h w``, both on CUDA. Normals point toward the camera.
        """
        output_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = _preprocess_rgb(rgb_bhw3, self.image_hw)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output: ForwardOutput = self.model(
                image_b3hw,
                num_tokens=self.num_tokens,
                output_heads=("normal", "mask"),
            )
        normals_bnhw3: Float32[Tensor, "b nh nw 3"] = cast(Tensor, output["normal"]).float()
        mask_bnhw: Float32[Tensor, "b nh nw"] = cast(Tensor, output["mask"]).float()
        return _postprocess_normal_output(normals_bnhw3, mask_bnhw, output_hw)


class MoGeV2TrtNormalPredictor:
    """Batched MoGe v2 normals on a cached dynamic-batch TensorRT engine."""

    def __init__(
        self,
        encoder: Encoder = "vitl",
        image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
        resolution_level: int = 9,
        batch_size: int = 8,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        """Export ONNX and build or load the machine-local engine on first use.

        Args:
            encoder: DINOv2 encoder size.
            image_hw: Static network height and width.
            resolution_level: Detail level from 0 through 9; level 9 uses
                3600 tokens for the pinned checkpoints.
            batch_size: Engine profile maximum and optimization batch.
            cache_dir: Cache root for ONNX and TensorRT artifacts.
        """
        from trtkit import TrtBuildConfig, ensure_engine
        from trtkit.tensorrt_runtime import TensorRtRuntime

        onnx_path: Path = export_moge_v2_normal_onnx(
            encoder=encoder,
            image_hw=image_hw,
            resolution_level=resolution_level,
            max_batch_size=batch_size,
            cache_dir=cache_dir,
        )
        config: TrtBuildConfig = TrtBuildConfig(max_batch_size=batch_size, opt_batch_size=batch_size)
        engine_path: Path = ensure_engine(onnx_path, config, cache_dir=cache_dir / "trt")
        self.runtime: TensorRtRuntime = TensorRtRuntime(engine_path)
        self.image_hw: tuple[int, int] = image_hw

    def __call__(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2NormalOutput:
        """Predict camera-space RDF normals and mask probabilities for a batch.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.

        Returns:
            Owning float32 unit normals shaped ``b h w 3`` and float32 masks
            shaped ``b h w``, both on CUDA. Normals point toward the camera.
        """
        output_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = _preprocess_rgb(rgb_bhw3, self.image_hw)
        runtime_output: dict[str, Tensor] = self.runtime({"image": image_b3hw})
        normals_bnhw3: Float32[Tensor, "b nh nw 3"] = runtime_output["normal"].float()
        mask_bnhw: Float32[Tensor, "b nh nw"] = runtime_output["mask"].float()
        return _postprocess_normal_output(normals_bnhw3, mask_bnhw, output_hw)
