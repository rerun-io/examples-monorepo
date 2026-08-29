"""Batched GPU predictors for MoGe v2 camera-space surface normals.

The torch and TensorRT predictors share one tensor contract: uint8 CUDA RGB
frames shaped ``[B,H,W,3]`` in, then float32 camera-space RDF normals shaped
``[B,H,W,3]`` and float32 validity probabilities shaped ``[B,H,W]`` out.
Normals point toward the camera, matching :class:`MoGeV2NormalPredictor`.
"""

from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float32, UInt8
from torch import Tensor

from monopriors.models.moge_v2_trt_shared import (
    DEFAULT_CACHE_DIR,
    Encoder,
    HardwareCompatibility,
    export_moge_v2_onnx,
    load_pinned_moge_v2,
    preprocess_rgb,
)
from monopriors.third_party.moge.model.v2 import ForwardOutput, MoGeModel

DEFAULT_IMAGE_HW: tuple[int, int] = (756, 1008)
"""ARKitScenes-aligned MoGe network height and width."""



class MoGeV2NormalOutput(NamedTuple):
    """Owning batched normal and mask tensors returned by both predictors."""

    normals_bhw3: Float32[Tensor, "b h w 3"]
    """Unit camera-space RDF normals at caller resolution."""
    mask_bhw: Float32[Tensor, "b h w"]
    """Validity probabilities at caller resolution."""


def export_moge_v2_normal_onnx(
    encoder: Encoder = "vitl",
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    resolution_level: int = 9,
    max_batch_size: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Export the normal-only graph while preserving the public helper."""
    return export_moge_v2_onnx("normal", encoder, image_hw, resolution_level, max_batch_size, cache_dir)


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
            resolution_level: Detail level from 0 through 9.

        Raises:
            RuntimeError: If CUDA is unavailable.
            ValueError: If ``resolution_level`` or checkpoint metadata is invalid.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("MoGe v2 batched normal prediction requires CUDA.")
        loaded: tuple[MoGeModel, int] = load_pinned_moge_v2(encoder, resolution_level)
        self.model: MoGeModel = loaded[0]
        self.num_tokens: int = loaded[1]
        self.image_hw: tuple[int, int] = image_hw

    def __call__(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2NormalOutput:
        """Predict camera-space RDF normals and mask probabilities for a batch.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.

        Returns:
            Owning float32 unit normals and masks at caller resolution on CUDA.
        """
        output_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = preprocess_rgb(rgb_bhw3, self.image_hw)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output: ForwardOutput = self.model(
                image_b3hw,
                num_tokens=self.num_tokens,
                output_heads=("normal", "mask"),
            )
        normals_bnhw3: Float32[Tensor, "b nh nw 3"] = output["normal"].float()
        mask_bnhw: Float32[Tensor, "b nh nw"] = output["mask"].float()
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
        hardware_compatibility: HardwareCompatibility = "none",
        engine_path: Path | None = None,
    ) -> None:
        """Export ONNX and build or load the machine-local engine on first use.

        Args:
            encoder: DINOv2 encoder size.
            image_hw: Static network height and width.
            resolution_level: Detail level from 0 through 9.
            batch_size: Engine profile maximum and optimization batch.
            cache_dir: Cache root for ONNX and TensorRT artifacts.
            hardware_compatibility: Plan portability requested when building an engine.
            engine_path: Optional prebuilt engine that skips export and build.
        """
        from trtkit.tensorrt_runtime import TensorRtRuntime

        resolved_engine_path: Path
        if engine_path is None:
            from trtkit import TrtBuildConfig, ensure_engine

            onnx_path: Path = export_moge_v2_normal_onnx(
                encoder=encoder,
                image_hw=image_hw,
                resolution_level=resolution_level,
                max_batch_size=batch_size,
                cache_dir=cache_dir,
            )
            config: TrtBuildConfig = TrtBuildConfig(
                max_batch_size=batch_size,
                opt_batch_size=batch_size,
                hardware_compatibility=hardware_compatibility,
            )
            resolved_engine_path = ensure_engine(onnx_path, config, cache_dir=cache_dir / "trt")
        else:
            resolved_engine_path = engine_path
        self.engine_path: Path = resolved_engine_path
        self.runtime: TensorRtRuntime = TensorRtRuntime(resolved_engine_path)
        self.image_hw: tuple[int, int] = image_hw

    def __call__(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2NormalOutput:
        """Predict camera-space RDF normals and mask probabilities for a batch.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.

        Returns:
            Owning float32 unit normals and masks at caller resolution on CUDA.
        """
        output_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = preprocess_rgb(rgb_bhw3, self.image_hw)
        runtime_output: dict[str, Tensor] = self.runtime({"image": image_b3hw})
        normals_bnhw3: Float32[Tensor, "b nh nw 3"] = runtime_output["normal"].float()
        mask_bnhw: Float32[Tensor, "b nh nw"] = runtime_output["mask"].float()
        return _postprocess_normal_output(normals_bnhw3, mask_bnhw, output_hw)
