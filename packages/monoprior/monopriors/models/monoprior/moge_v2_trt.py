"""Batched GPU predictors for full MoGe v2 metric geometry.

The torch and TensorRT predictors share one contract: uint8 CUDA RGB frames
shaped ``[B,H,W,3]`` in, then metric depth, raw affine camera-space points,
RDF normals, validity probabilities, normalized intrinsics, and recovered
scale and shift values out. The network runs on CUDA; camera recovery uses
MoGe's per-image CPU SciPy solve.
"""

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Protocol, cast, runtime_checkable

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Float32, UInt8
from torch import Tensor

from monopriors.models.surface_normal.moge_v2 import MOGE_V2_NORMAL_CHECKPOINTS
from monopriors.models.surface_normal.moge_v2_trt import (
    DEFAULT_CACHE_DIR,
    DEFAULT_IMAGE_HW,
    MOGE_V2_NUM_TOKENS_RANGE,
    Encoder,
    HardwareCompatibility,
    preprocess_rgb,
    resolve_num_tokens,
)
from monopriors.third_party.moge.model._inference import CameraRecovery, recover_shift_and_intrinsics
from monopriors.third_party.moge.model.v2 import ForwardOutput, MoGeModel

ONNX_EXPORT_VERSION: int = 1
"""Cache-breaking version of the full MoGe v2 geometry ONNX recipe."""

_EXPORT_WORKER_ENV: str = "MONOPRIOR_MOGE_V2_ONNX_EXPORT_WORKER"


def token_grid_hw(aspect_ratio: float, num_tokens: int) -> tuple[int, int]:
    """Compute MoGe's patch-aligned network size for one image aspect.

    Args:
        aspect_ratio: Image width divided by height.
        num_tokens: Approximate DINOv2 patch-token budget.

    Returns:
        Network height and width in pixels, each aligned to the 14-pixel patch size.

    Raises:
        ValueError: If the aspect ratio or token budget is not positive and finite.
    """
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        raise ValueError(f"aspect_ratio must be positive and finite, got {aspect_ratio}.")
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
    token_rows: int = round(math.sqrt(num_tokens / aspect_ratio))
    token_cols: int = round(math.sqrt(num_tokens * aspect_ratio))
    if token_rows < 1 or token_cols < 1:
        raise ValueError(f"aspect_ratio {aspect_ratio} produces an empty token grid.")
    return token_rows * 14, token_cols * 14


ASPECT_BUCKETS: tuple[tuple[int, int], ...] = (
    token_grid_hw(4.0 / 3.0, 3600),
    token_grid_hw(3.0 / 4.0, 3600),
    token_grid_hw(16.0 / 9.0, 3600),
    token_grid_hw(9.0 / 16.0, 3600),
    token_grid_hw(1.0, 3600),
)
"""MoGe network sizes for 4:3, 3:4, 16:9, 9:16, and 1:1 inputs."""


def select_aspect_bucket(
    input_hw: tuple[int, int],
    buckets: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    """Select the network bucket nearest to an input aspect in log-space.

    Args:
        input_hw: Caller input height and width.
        buckets: Candidate network height and width pairs.

    Returns:
        The nearest candidate bucket. Ties retain candidate order.

    Raises:
        ValueError: If the input, any bucket, or the candidate set is invalid.
    """
    input_height: int = input_hw[0]
    input_width: int = input_hw[1]
    if input_height < 1 or input_width < 1:
        raise ValueError(f"Input dimensions must be positive, got {input_hw}.")
    if not buckets:
        raise ValueError("At least one aspect bucket is required.")

    input_log_aspect: float = math.log(input_width / input_height)
    selected_bucket: tuple[int, int] = buckets[0]
    selected_distance: float = math.inf
    for bucket_hw in buckets:
        bucket_height: int = bucket_hw[0]
        bucket_width: int = bucket_hw[1]
        if bucket_height < 1 or bucket_width < 1:
            raise ValueError(f"Bucket dimensions must be positive, got {bucket_hw}.")
        distance: float = abs(math.log(bucket_width / bucket_height) - input_log_aspect)
        if distance < selected_distance:
            selected_bucket = bucket_hw
            selected_distance = distance
    return selected_bucket


class MoGeV2GeometryOutput(NamedTuple):
    """Owning metric geometry tensors returned by both predictors."""

    depth_bhw: Float32[Tensor, "b h w"]
    """Metric depth at caller resolution; exactly ``points_bhw3[..., 2]``."""
    points_bhw3: Float32[Tensor, "b h w 3"]
    """Raw affine point map after recovered Z shift and metric scaling."""
    normals_bhw3: Float32[Tensor, "b h w 3"]
    """Unit camera-space RDF normals at caller resolution."""
    mask_bhw: Float32[Tensor, "b h w"]
    """Validity probabilities at caller resolution, zero where depth is non-positive."""
    intrinsics_b33: Float32[Tensor, "b 3 3"]
    """Normalized MoGe intrinsics; pixels are ``diag([W, H, 1]) @ K``."""
    metric_scale_b: Float32[Tensor, "b"]
    """Positive per-image scale applied to the shifted affine geometry."""
    shift_b: Float32[Tensor, "b"]
    """Per-image Z translation in affine point-map units, before metric scaling."""


class MoGeV2GeometryGraphOutput(NamedTuple):
    """Full-head tensors at the fixed network resolution before camera recovery."""

    points_bhw3: Float32[Tensor, "b nh nw 3"]
    """Raw affine point-head output at network resolution."""
    normals_bhw3: Float32[Tensor, "b nh nw 3"]
    """Unit camera-space RDF normal-head output at network resolution."""
    mask_bhw: Float32[Tensor, "b nh nw"]
    """Validity probabilities at network resolution."""
    metric_scale_b: Float32[Tensor, "b"]
    """Positive metric-scale head output for each image."""


@runtime_checkable
class _GeometryRuntime(Protocol):
    """Delayed-import TensorRT runtime interface used by the predictor."""

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Execute one batch and return named runtime buffers."""
        ...


class _MoGeV2GeometryHeads(torch.nn.Module):
    """Fixed-token adapter exposing all four MoGe geometry heads as a tuple."""

    def __init__(self, model: MoGeModel, num_tokens: int) -> None:
        super().__init__()
        self.model: MoGeModel = model
        self.num_tokens: int = num_tokens

    def forward(
        self,
        image_b3hw: Float[Tensor, "b 3 h w"],
    ) -> tuple[
        Float[Tensor, "b h w 3"],
        Float[Tensor, "b h w 3"],
        Float[Tensor, "b h w"],
        Float[Tensor, "b"],
    ]:
        """Run the points, normal, mask, and metric-scale heads.

        Args:
            image_b3hw: Float RGB batch shaped ``b 3 h w`` in ``[0, 1]``.

        Returns:
            Affine points, unit normals, mask probabilities, and metric scales.
        """
        output: ForwardOutput = self.model(
            image_b3hw,
            num_tokens=self.num_tokens,
            output_heads=("points", "normal", "mask", "scale"),
        )
        points_bhw3: Float[Tensor, "b h w 3"] = cast(Tensor, output["points"])
        normals_bhw3: Float[Tensor, "b h w 3"] = cast(Tensor, output["normal"])
        mask_bhw: Float[Tensor, "b h w"] = cast(Tensor, output["mask"])
        metric_scale_b: Float[Tensor, "b"] = cast(Tensor, output["metric_scale"])
        return points_bhw3, normals_bhw3, mask_bhw, metric_scale_b


def export_moge_v2_geometry_onnx(
    encoder: Encoder = "vitl",
    image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
    resolution_level: int = 9,
    max_batch_size: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Export the full MoGe v2 geometry graph with dynamic batch only.

    Args:
        encoder: DINOv2 encoder size.
        image_hw: Static network height and width.
        resolution_level: Detail level from 0 through 9.
        max_batch_size: Largest batch encoded in the dynamic ONNX constraint.
        cache_dir: Cache root for the ONNX artifact.

    Returns:
        Cached or newly exported ONNX path.

    Raises:
        RuntimeError: If CUDA is unavailable or the export worker fails.
        ValueError: If dimensions, batch size, or checkpoint token metadata are invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Full MoGe v2 ONNX export requires CUDA.")
    height: int = image_hw[0]
    width: int = image_hw[1]
    if height < 1 or width < 1:
        raise ValueError(f"MoGe v2 network dimensions must be positive, got {image_hw}.")
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")

    num_tokens: int = resolve_num_tokens(resolution_level)
    checkpoint: tuple[str, str] = MOGE_V2_NORMAL_CHECKPOINTS[encoder]
    checkpoint_revision: str = checkpoint[1][:8]
    onnx_dir: Path = cache_dir / "onnx"
    onnx_path: Path = onnx_dir / (
        f"moge-v2-{encoder}-geometry_{height}x{width}_t{num_tokens}_v{ONNX_EXPORT_VERSION}_{checkpoint_revision}.onnx"
    )
    if onnx_path.exists():
        return onnx_path

    if os.environ.get("PIXI_DEV_MODE") == "1" and os.environ.get(_EXPORT_WORKER_ENV) != "1":
        worker_env: dict[str, str] = dict(os.environ)
        worker_env["PIXI_DEV_MODE"] = "0"
        worker_env[_EXPORT_WORKER_ENV] = "1"
        worker_command: list[str] = [
            sys.executable,
            "-m",
            "monopriors.models.surface_normal._moge_v2_onnx_worker",
            "--heads",
            "geometry",
            "--encoder",
            encoder,
            "--height",
            str(height),
            "--width",
            str(width),
            "--resolution-level",
            str(resolution_level),
            "--max-batch-size",
            str(max_batch_size),
            "--cache-dir",
            str(cache_dir),
        ]
        subprocess.run(worker_command, check=True, env=worker_env)
        if not onnx_path.exists():
            raise RuntimeError(f"Full MoGe v2 ONNX export worker did not produce {onnx_path}.")
        return onnx_path

    print(f"[monoprior] exporting full MoGe v2 geometry to ONNX (one-time, may take minutes): {onnx_path.name}")
    model: MoGeModel = MoGeModel.from_pretrained(checkpoint[0], revision=checkpoint[1]).to("cuda").eval()
    if model.num_tokens_range != MOGE_V2_NUM_TOKENS_RANGE:
        raise ValueError(
            f"Pinned MoGe v2 checkpoint token range changed from {MOGE_V2_NUM_TOKENS_RANGE} to {model.num_tokens_range}; bump the export recipe."
        )
    model.onnx_compatible_mode = True
    wrapper: _MoGeV2GeometryHeads = _MoGeV2GeometryHeads(model, num_tokens).eval()
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
        output_names=["points", "normal", "mask", "metric_scale"],
        compute_dtype=torch.float16,
        dynamic_batch_max=max_batch_size,
    )
    del dummy_image_b3hw, wrapper, model
    torch.cuda.empty_cache()

    current_sidecar: Path = onnx_path.with_name(f"{onnx_path.name}.data")
    stale_prefix: str = f"moge-v2-{encoder}-geometry_{height}x{width}_t{num_tokens}_"
    sweep_stale_onnx_exports(onnx_dir, stale_prefix, keep_paths={onnx_path, current_sidecar})
    return onnx_path


def postprocess_moge_v2_geometry(
    graph_output: MoGeV2GeometryGraphOutput,
    output_hw: tuple[int, int],
) -> MoGeV2GeometryOutput:
    """Recover metric pinhole geometry and resize it to the caller resolution.

    Args:
        graph_output: Float32 full-head outputs at network resolution.
        output_hw: Caller input height and width.

    Returns:
        Owning float32 CUDA depth, raw points, normals, mask, normalized
        intrinsics, metric scale, and recovered shift.
    """
    points_bnhw3: Float32[Tensor, "b nh nw 3"] = graph_output.points_bhw3.float().clone()
    normals_bnhw3: Float32[Tensor, "b nh nw 3"] = graph_output.normals_bhw3.float().clone()
    mask_bnhw: Float32[Tensor, "b nh nw"] = graph_output.mask_bhw.float().clone()
    metric_scale_b: Float32[Tensor, "b"] = graph_output.metric_scale_b.float().clone()
    network_height: int = points_bnhw3.shape[1]
    network_width: int = points_bnhw3.shape[2]
    aspect_ratio: float = network_width / network_height
    recovery_mask_bnhw: Bool[Tensor, "b nh nw"] = mask_bnhw > 0.5
    camera: CameraRecovery = recover_shift_and_intrinsics(
        points_bnhw3,
        recovery_mask_bnhw,
        fov_x=None,
        aspect_ratio=aspect_ratio,
    )

    shift_b: Float32[Tensor, "b"] = camera.shift_b.float().clone()
    intrinsics_b33: Float32[Tensor, "b 3 3"] = camera.intrinsics_b33.float().clone()
    points_bnhw3[..., 2] += shift_b[:, None, None]
    points_bnhw3 *= metric_scale_b[:, None, None, None]
    positive_z_bnhw: Bool[Tensor, "b nh nw"] = points_bnhw3[..., 2] > 0.0
    mask_bnhw = torch.where(positive_z_bnhw, mask_bnhw, torch.zeros_like(mask_bnhw))

    points_b3nhw: Float32[Tensor, "b 3 nh nw"] = rearrange(points_bnhw3, "b h w c -> b c h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    normals_b3nhw: Float32[Tensor, "b 3 nh nw"] = rearrange(normals_bnhw3, "b h w c -> b c h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    mask_b1nhw: Float32[Tensor, "b 1 nh nw"] = mask_bnhw[:, None]
    if (network_height, network_width) != output_hw:
        points_b3hw: Float32[Tensor, "b 3 h w"] = F.interpolate(points_b3nhw, size=output_hw, mode="bilinear", align_corners=False)
        normals_b3hw: Float32[Tensor, "b 3 h w"] = F.interpolate(normals_b3nhw, size=output_hw, mode="bilinear", align_corners=False)
        mask_b1hw: Float32[Tensor, "b 1 h w"] = F.interpolate(mask_b1nhw, size=output_hw, mode="bilinear", align_corners=False)
    else:
        points_b3hw = points_b3nhw.clone()
        normals_b3hw = normals_b3nhw.clone()
        mask_b1hw = mask_b1nhw.clone()

    points_bhw3: Float32[Tensor, "b h w 3"] = rearrange(points_b3hw, "b c h w -> b h w c").contiguous()  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    depth_bhw: Float32[Tensor, "b h w"] = points_bhw3[..., 2].clone()
    normals_bhw3: Float32[Tensor, "b h w 3"] = F.normalize(
        rearrange(normals_b3hw, "b c h w -> b h w c"),  # pyrefly: ignore  # bad-argument-type — einops stub false positive
        dim=-1,
    )
    mask_bhw: Float32[Tensor, "b h w"] = torch.where(depth_bhw > 0.0, mask_b1hw[:, 0], torch.zeros_like(mask_b1hw[:, 0]))
    return MoGeV2GeometryOutput(
        depth_bhw=depth_bhw,
        points_bhw3=points_bhw3,
        normals_bhw3=normals_bhw3,
        mask_bhw=mask_bhw,
        intrinsics_b33=intrinsics_b33,
        metric_scale_b=metric_scale_b,
        shift_b=shift_b,
    )


class MoGeV2TorchGeometryPredictor:
    """Plain-torch fp16-compute twin of the full TensorRT predictor."""

    def __init__(
        self,
        encoder: Encoder = "vitl",
        image_hw: tuple[int, int] | None = None,
        resolution_level: int = 9,
    ) -> None:
        """Load a pinned normal-capable MoGe v2 checkpoint onto CUDA.

        Args:
            encoder: DINOv2 encoder size.
            image_hw: Static network height and width, or ``None`` to select
                an aspect bucket for each call.
            resolution_level: Detail level from 0 through 9.

        Raises:
            RuntimeError: If CUDA is unavailable.
            ValueError: If checkpoint token metadata changed.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("Full MoGe v2 batched prediction requires CUDA.")
        num_tokens: int = resolve_num_tokens(resolution_level)
        checkpoint: tuple[str, str] = MOGE_V2_NORMAL_CHECKPOINTS[encoder]
        self.model: MoGeModel = MoGeModel.from_pretrained(checkpoint[0], revision=checkpoint[1]).to("cuda").eval()
        if self.model.num_tokens_range != MOGE_V2_NUM_TOKENS_RANGE:
            raise ValueError(
                f"Pinned MoGe v2 checkpoint token range changed from {MOGE_V2_NUM_TOKENS_RANGE} to {self.model.num_tokens_range}; update the predictor."
            )
        self.model.onnx_compatible_mode = True
        self.num_tokens: int = num_tokens
        self.image_hw: tuple[int, int] | None = image_hw

    def forward(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2GeometryGraphOutput:
        """Run preprocessing and all four network heads.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.

        Returns:
            Float32 graph outputs at the fixed network resolution.
        """
        input_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        network_hw: tuple[int, int] = self.image_hw if self.image_hw is not None else select_aspect_bucket(input_hw, ASPECT_BUCKETS)
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = preprocess_rgb(rgb_bhw3, network_hw)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output: ForwardOutput = self.model(
                image_b3hw,
                num_tokens=self.num_tokens,
                output_heads=("points", "normal", "mask", "scale"),
            )
        return MoGeV2GeometryGraphOutput(
            points_bhw3=cast(Tensor, output["points"]).float(),
            normals_bhw3=cast(Tensor, output["normal"]).float(),
            mask_bhw=cast(Tensor, output["mask"]).float(),
            metric_scale_b=cast(Tensor, output["metric_scale"]).float(),
        )

    def __call__(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2GeometryOutput:
        """Predict metric depth, points, normals, validity, and intrinsics.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.

        Returns:
            Owning float32 CUDA geometry tensors at the caller resolution.
        """
        output_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        graph_output: MoGeV2GeometryGraphOutput = self.forward(rgb_bhw3)
        return postprocess_moge_v2_geometry(graph_output, output_hw)


class MoGeV2TrtGeometryPredictor:
    """Batched full MoGe v2 geometry on a cached dynamic-batch TensorRT engine."""

    def __init__(
        self,
        encoder: Encoder = "vitl",
        image_hw: tuple[int, int] | None = None,
        resolution_level: int = 9,
        batch_size: int = 8,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        hardware_compatibility: HardwareCompatibility = "none",
        engine_path: Path | None = None,
        use_cuda_graph: bool = False,
        builder_optimization_level: int = 5,
    ) -> None:
        """Build or load the TensorRT engine.

        Args:
            encoder: DINOv2 encoder size.
            image_hw: Static network height and width, or ``None`` to lazily
                build one engine per selected aspect bucket.
            resolution_level: Detail level from 0 through 9.
            batch_size: Engine profile maximum and optimization batch.
            cache_dir: Cache root for ONNX and TensorRT artifacts.
            hardware_compatibility: TensorRT plan portability level.
            engine_path: Prebuilt engine that bypasses export and build. This
                requires an explicit ``image_hw``.
            use_cuda_graph: Whether each runtime captures engine launches in a CUDA graph.
            builder_optimization_level: TensorRT builder optimization level from 0 through 5.

        Raises:
            ValueError: If a prebuilt engine has no static image size or a configuration value is invalid.
        """
        if engine_path is not None and image_hw is None:
            raise ValueError("image_hw is required when engine_path supplies a single static engine.")
        if image_hw is not None and (image_hw[0] < 1 or image_hw[1] < 1):
            raise ValueError(f"MoGe v2 network dimensions must be positive, got {image_hw}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not 0 <= builder_optimization_level <= 5:
            raise ValueError(f"builder_optimization_level must be within [0, 5], got {builder_optimization_level}.")

        self.encoder: Encoder = encoder
        self.image_hw: tuple[int, int] | None = image_hw
        self.resolution_level: int = resolution_level
        self.batch_size: int = batch_size
        self.cache_dir: Path = cache_dir
        self.hardware_compatibility: HardwareCompatibility = hardware_compatibility
        self.prebuilt_engine_path: Path | None = engine_path
        self.use_cuda_graph: bool = use_cuda_graph
        self.builder_optimization_level: int = builder_optimization_level
        self.engine_paths: dict[tuple[int, int], Path] = {}
        self.runtimes: dict[tuple[int, int], _GeometryRuntime] = {}
        if image_hw is not None:
            self._ensure_runtime(image_hw)

    def _ensure_runtime(self, image_hw: tuple[int, int]) -> _GeometryRuntime:
        """Build or load the runtime for one static network size."""
        existing_runtime: _GeometryRuntime | None = self.runtimes.get(image_hw)
        if existing_runtime is not None:
            return existing_runtime

        resolved_engine_path: Path
        if self.prebuilt_engine_path is None:
            from trtkit import TrtBuildConfig, ensure_engine

            onnx_path: Path = export_moge_v2_geometry_onnx(
                encoder=self.encoder,
                image_hw=image_hw,
                resolution_level=self.resolution_level,
                max_batch_size=self.batch_size,
                cache_dir=self.cache_dir,
            )
            config: TrtBuildConfig = TrtBuildConfig(
                max_batch_size=self.batch_size,
                opt_batch_size=self.batch_size,
                builder_optimization_level=self.builder_optimization_level,
                hardware_compatibility=self.hardware_compatibility,
            )
            resolved_engine_path = ensure_engine(onnx_path, config, cache_dir=self.cache_dir / "trt")
        else:
            resolved_engine_path = self.prebuilt_engine_path

        from trtkit.tensorrt_runtime import TensorRtRuntime

        runtime: _GeometryRuntime = TensorRtRuntime(resolved_engine_path, use_cuda_graph=self.use_cuda_graph)
        self.engine_paths[image_hw] = resolved_engine_path
        self.runtimes[image_hw] = runtime
        return runtime

    def forward(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2GeometryGraphOutput:
        """Run preprocessing and the TensorRT geometry engine.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.

        Returns:
            Float32 graph outputs backed by runtime-reused buffers. Consume the
            result before the next call to this method.
        """
        input_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        network_hw: tuple[int, int] = self.image_hw if self.image_hw is not None else select_aspect_bucket(input_hw, ASPECT_BUCKETS)
        runtime: _GeometryRuntime = self._ensure_runtime(network_hw)
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = preprocess_rgb(rgb_bhw3, network_hw)
        runtime_output: dict[str, Tensor] = runtime({"image": image_b3hw})
        return MoGeV2GeometryGraphOutput(
            points_bhw3=runtime_output["points"].float(),
            normals_bhw3=runtime_output["normal"].float(),
            mask_bhw=runtime_output["mask"].float(),
            metric_scale_b=runtime_output["metric_scale"].float(),
        )

    def __call__(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2GeometryOutput:
        """Predict metric depth, points, normals, validity, and intrinsics.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.

        Returns:
            Owning float32 CUDA geometry tensors at the caller resolution.
        """
        output_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        graph_output: MoGeV2GeometryGraphOutput = self.forward(rgb_bhw3)
        return postprocess_moge_v2_geometry(graph_output, output_hw)
