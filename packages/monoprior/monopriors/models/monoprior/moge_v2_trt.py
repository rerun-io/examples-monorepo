"""Batched GPU predictors for full MoGe v2 metric geometry.

The torch and TensorRT predictors share one contract: uint8 CUDA RGB frames
shaped ``[B,H,W,3]`` in, then metric depth, raw affine camera-space points,
RDF normals, validity probabilities, normalized intrinsics, and recovered
scale and shift values out. The network runs on CUDA; camera recovery uses
MoGe's per-image CPU SciPy solve.
"""

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8
from torch import Tensor

from monopriors.models.moge_v2_trt_shared import (
    DEFAULT_CACHE_DIR,
    ONNX_OUTPUT_NAMES,
    Encoder,
    HardwareCompatibility,
    aspect_buckets,
    export_moge_v2_onnx,
    load_pinned_moge_v2,
    preprocess_rgb,
    resolve_num_tokens,
    select_aspect_bucket,
)
from monopriors.third_party.moge.model._inference import CameraRecovery, recover_shift_and_intrinsics
from monopriors.third_party.moge.model.v2 import ForwardOutput, MoGeModel

# trtkit imports TensorRT at package import, so it stays a call-site import; beartype
# resolves local annotations at runtime, hence the plain-object stand-in outside type checking.
if TYPE_CHECKING:
    from trtkit.base import TensorRuntime
else:
    TensorRuntime = object

GEOMETRY_OUTPUT_NAMES: tuple[str, ...] = ONNX_OUTPUT_NAMES["geometry"]
"""Stable runtime output names for the full geometry graph."""


class MoGeV2GeometryOutput(NamedTuple):
    """Owning metric geometry tensors returned by both predictors."""

    depth_bhw: Float32[Tensor, "b h w"]
    """Metric depth at caller resolution; exactly ``points_bhw3[..., 2]``."""
    points_bhw3: Float32[Tensor, "b h w 3"]
    """Metric points for the bucket-aspect camera, sampled at caller resolution."""
    normals_bhw3: Float32[Tensor, "b h w 3"]
    """Unit camera-space RDF normals at caller resolution."""
    mask_bhw: Float32[Tensor, "b h w"]
    """Validity probabilities at caller resolution, zero where depth is non-positive."""
    intrinsics_b33: Float32[Tensor, "b 3 3"]
    """Normalized bucket-aspect camera intrinsics; input is stretched to the nearest bucket."""
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


def postprocess_moge_v2_geometry(
    graph_output: MoGeV2GeometryGraphOutput,
    output_hw: tuple[int, int],
) -> MoGeV2GeometryOutput:
    """Recover metric pinhole geometry and resize it to the caller resolution.

    Args:
        graph_output: Float32 full-head outputs at network resolution.
        output_hw: Caller input height and width.

    Returns:
        Owning float32 depth, points, normals, mask, intrinsics, metric scale,
        and recovered shift tensors. Intrinsics and points describe the bucket
        aspect camera even when the caller aspect differs.
    """
    metric_scale_b: Float32[Tensor, "b"] = graph_output.metric_scale_b.float().clone()
    points_bnhw3: Float32[Tensor, "b nh nw 3"] = graph_output.points_bhw3.float() * metric_scale_b[:, None, None, None]
    normals_bnhw3: Float32[Tensor, "b nh nw 3"] = graph_output.normals_bhw3.float()
    mask_bnhw: Float32[Tensor, "b nh nw"] = graph_output.mask_bhw.float()
    network_height: int = points_bnhw3.shape[1]
    network_width: int = points_bnhw3.shape[2]

    normals_b3nhw: Float32[Tensor, "b 3 nh nw"] = rearrange(normals_bnhw3, "b h w c -> b c h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    if (network_height, network_width) != output_hw:
        normals_b3hw: Float32[Tensor, "b 3 h w"] = F.interpolate(normals_b3nhw, size=output_hw, mode="bilinear", align_corners=False)
    else:
        normals_b3hw = normals_b3nhw
    normals_bhw3: Float32[Tensor, "b h w 3"] = F.normalize(
        rearrange(normals_b3hw, "b c h w -> b h w c"),  # pyrefly: ignore  # bad-argument-type — einops stub false positive
        dim=-1,
    )

    recovery_mask_bnhw: Bool[Tensor, "b nh nw"] = mask_bnhw > 0.5
    aspect_ratio: float = network_width / network_height
    camera: CameraRecovery = recover_shift_and_intrinsics(
        graph_output.points_bhw3.float(),
        recovery_mask_bnhw,
        fov_x=None,
        aspect_ratio=aspect_ratio,
    )
    shift_b: Float32[Tensor, "b"] = camera.shift_b.float()
    intrinsics_b33: Float32[Tensor, "b 3 3"] = camera.intrinsics_b33.float()
    points_bnhw3[..., 2] += shift_b[:, None, None] * metric_scale_b[:, None, None]
    positive_z_bnhw: Bool[Tensor, "b nh nw"] = points_bnhw3[..., 2] > 0.0
    mask_bnhw = torch.where(positive_z_bnhw, mask_bnhw, 0.0)

    points_b3nhw: Float32[Tensor, "b 3 nh nw"] = rearrange(points_bnhw3, "b h w c -> b c h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    mask_b1nhw: Float32[Tensor, "b 1 nh nw"] = rearrange(mask_bnhw, "b h w -> b 1 h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    if (network_height, network_width) != output_hw:
        points_b3hw: Float32[Tensor, "b 3 h w"] = F.interpolate(points_b3nhw, size=output_hw, mode="bilinear", align_corners=False)
        mask_b1hw: Float32[Tensor, "b 1 h w"] = F.interpolate(mask_b1nhw, size=output_hw, mode="bilinear", align_corners=False)
    else:
        points_b3hw = points_b3nhw
        mask_b1hw = mask_b1nhw

    points_bhw3: Float32[Tensor, "b h w 3"] = rearrange(points_b3hw, "b c h w -> b h w c").contiguous()  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    depth_bhw: Float32[Tensor, "b h w"] = points_bhw3[..., 2]
    resized_mask_bhw: Float32[Tensor, "b h w"] = rearrange(mask_b1hw, "b 1 h w -> b h w")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    mask_bhw = torch.where(depth_bhw > 0.0, resized_mask_bhw, 0.0)
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
    """Plain-torch full geometry with a bucket-aspect camera model.

    Caller images are stretched to the nearest network bucket. Intrinsics and
    points describe that bucket-aspect camera. For example, snapping 3:2 input
    to 4:3 causes about 12.5% anisotropic stretch; caller-aspect correction is
    a follow-up.
    """

    def __init__(
        self,
        encoder: Encoder = "vitl",
        network_hw_options: tuple[tuple[int, int], ...] | None = None,
        resolution_level: int = 9,
    ) -> None:
        """Load a pinned normal-capable MoGe v2 checkpoint onto CUDA.

        Args:
            encoder: DINOv2 encoder size.
            network_hw_options: Candidate network sizes. ``None`` derives five
                aspect buckets from the selected resolution level; one pair
                fixes the network size.
            resolution_level: Detail level from 0 through 9.

        Raises:
            RuntimeError: If CUDA is unavailable.
            ValueError: If checkpoint token metadata changed.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("Full MoGe v2 batched prediction requires CUDA.")
        loaded: tuple[MoGeModel, int] = load_pinned_moge_v2(encoder, resolution_level)
        self.model: MoGeModel = loaded[0]
        self.num_tokens: int = loaded[1]
        self.network_hw_options: tuple[tuple[int, int], ...] = aspect_buckets(self.num_tokens) if network_hw_options is None else network_hw_options

    def forward(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2GeometryGraphOutput:
        """Run preprocessing and all four network heads.

        Args:
            rgb_bhw3: uint8 CUDA RGB frames shaped ``b h w 3``.

        Returns:
            Float32 graph outputs at the selected network resolution.
        """
        input_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        network_hw: tuple[int, int] = select_aspect_bucket(input_hw, self.network_hw_options)
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = preprocess_rgb(rgb_bhw3, network_hw)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output: ForwardOutput = self.model(
                image_b3hw,
                num_tokens=self.num_tokens,
                output_heads=("points", "normal", "mask", "scale"),
            )
        points_bhw3: Float32[Tensor, "b nh nw 3"] = output["points"].float()
        normals_bhw3: Float32[Tensor, "b nh nw 3"] = output["normal"].float()
        mask_bhw: Float32[Tensor, "b nh nw"] = output["mask"].float()
        metric_scale_b: Float32[Tensor, "b"] = output["metric_scale"].float()
        return MoGeV2GeometryGraphOutput(points_bhw3, normals_bhw3, mask_bhw, metric_scale_b)

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
    """TensorRT full geometry with a bucket-aspect camera model.

    Caller images are stretched to the nearest network bucket. Intrinsics and
    points describe that bucket-aspect camera. For example, snapping 3:2 input
    to 4:3 causes about 12.5% anisotropic stretch; caller-aspect correction is
    a follow-up. With multiple buckets, the first frame of a new aspect causes
    ONNX export and engine build when uncached. Every bucket seen stays resident.
    """

    def __init__(
        self,
        encoder: Encoder = "vitl",
        network_hw_options: tuple[tuple[int, int], ...] | None = None,
        resolution_level: int = 9,
        batch_size: int = 8,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        hardware_compatibility: HardwareCompatibility = "none",
        use_cuda_graph: bool = False,
        builder_optimization_level: int = 5,
    ) -> None:
        """Configure eager fixed-size or lazy bucketed TensorRT runtimes.

        Args:
            encoder: DINOv2 encoder size.
            network_hw_options: Candidate network sizes. ``None`` derives five
                aspect buckets from the selected resolution level; one pair
                builds eagerly.
            resolution_level: Detail level from 0 through 9.
            batch_size: Engine profile maximum and optimization batch.
            cache_dir: Cache root for ONNX and TensorRT artifacts.
            hardware_compatibility: TensorRT plan portability level.
            use_cuda_graph: Whether each runtime captures engine launches in a CUDA graph.
            builder_optimization_level: TensorRT builder optimization level.

        Raises:
            ValueError: If the batch size is invalid.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        num_tokens: int = resolve_num_tokens(resolution_level)
        self.encoder: Encoder = encoder
        self.network_hw_options: tuple[tuple[int, int], ...] = aspect_buckets(num_tokens) if network_hw_options is None else network_hw_options
        self.resolution_level: int = resolution_level
        self.batch_size: int = batch_size
        self.cache_dir: Path = cache_dir
        self.hardware_compatibility: HardwareCompatibility = hardware_compatibility
        self.use_cuda_graph: bool = use_cuda_graph
        self.builder_optimization_level: int = builder_optimization_level
        self.runtimes: dict[tuple[int, int], TensorRuntime] = {}
        if len(self.network_hw_options) == 1:
            self._ensure_runtime(self.network_hw_options[0])

    def _ensure_runtime(self, image_hw: tuple[int, int]) -> TensorRuntime:
        """Build or load the runtime for one static network size."""
        existing_runtime: TensorRuntime | None = self.runtimes.get(image_hw)
        if existing_runtime is not None:
            return existing_runtime

        from trtkit import TrtBuildConfig, ensure_engine

        onnx_path: Path = export_moge_v2_onnx(
            heads="geometry",
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
        engine_path: Path = ensure_engine(onnx_path, config, cache_dir=self.cache_dir / "trt")

        from trtkit.tensorrt_runtime import TensorRtRuntime

        runtime: TensorRuntime = TensorRtRuntime(engine_path, use_cuda_graph=self.use_cuda_graph)
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
        network_hw: tuple[int, int] = select_aspect_bucket(input_hw, self.network_hw_options)
        runtime: TensorRuntime = self._ensure_runtime(network_hw)
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = preprocess_rgb(rgb_bhw3, network_hw)
        runtime_output: dict[str, Tensor] = runtime({"image": image_b3hw})
        points_bhw3: Float32[Tensor, "b nh nw 3"] = runtime_output[GEOMETRY_OUTPUT_NAMES[0]].float()
        normals_bhw3: Float32[Tensor, "b nh nw 3"] = runtime_output[GEOMETRY_OUTPUT_NAMES[1]].float()
        mask_bhw: Float32[Tensor, "b nh nw"] = runtime_output[GEOMETRY_OUTPUT_NAMES[2]].float()
        metric_scale_b: Float32[Tensor, "b"] = runtime_output[GEOMETRY_OUTPUT_NAMES[3]].float()
        return MoGeV2GeometryGraphOutput(points_bhw3, normals_bhw3, mask_bhw, metric_scale_b)

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
