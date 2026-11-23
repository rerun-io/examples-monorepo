"""Unified batched Torch and TensorRT predictors for MoGe v2."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8
from torch import Tensor

from monopriors.models.moge_v2.export import (
    DEFAULT_CACHE_DIR,
    DYNAMIC_ONNX_MAX_BATCH,
    FORWARD_HEADS,
    ONNX_OUTPUT_NAMES,
    Encoder,
    HardwareCompatibility,
    HeadSet,
    aspect_buckets,
    dynamic_patch_range,
    export_moge_v2_onnx,
    load_pinned_moge_v2,
    moge_v2_onnx_path,
    preprocess_rgb,
    resolve_num_tokens,
    select_aspect_bucket,
    token_grid_hw,
)
from monopriors.third_party.moge.model._inference import CameraRecovery, recover_shift_and_intrinsics
from monopriors.third_party.moge.model.v2 import ForwardOutput, MoGeModel

if TYPE_CHECKING:
    from trtkit import TensorRtDynamicRuntime, TrtBuildConfig
    from trtkit.base import TensorRuntime
else:
    TensorRuntime = object
    TensorRtDynamicRuntime = object
    TrtBuildConfig = object


EngineProfile: TypeAlias = Literal["dynamic", "static"]
"""``dynamic`` (default): one engine over a range of token grids; ``static``: one engine per network size."""


class MoGeV2NormalOutput(NamedTuple):
    """Owning normal and mask tensors at caller resolution."""

    normals_bhw3: Float32[Tensor, "b h w 3"]
    """Unit camera-space RDF normals."""
    mask_bhw: Float32[Tensor, "b h w"]
    """Validity probabilities."""


class MoGeV2GeometryOutput(NamedTuple):
    """Owning metric geometry tensors at caller resolution."""

    depth_bhw: Float32[Tensor, "b h w"]
    """Metric depth; exactly ``points_bhw3[..., 2]``."""
    points_bhw3: Float32[Tensor, "b h w 3"]
    """Metric camera-space points."""
    normals_bhw3: Float32[Tensor, "b h w 3"]
    """Unit camera-space RDF normals."""
    mask_bhw: Float32[Tensor, "b h w"]
    """Validity probabilities, zero where depth is non-positive."""
    intrinsics_b33: Float32[Tensor, "b 3 3"]
    """Normalized camera intrinsics."""
    metric_scale_b: Float32[Tensor, "b"]
    """Positive per-image metric scale."""
    shift_b: Float32[Tensor, "b"]
    """Recovered Z translation before metric scaling."""


class MoGeV2NormalGraphOutput(NamedTuple):
    """Normal-head tensors at network resolution."""

    normals_bhw3: Float32[Tensor, "b nh nw 3"]
    """Unit camera-space RDF normal-head output."""
    mask_bhw: Float32[Tensor, "b nh nw"]
    """Validity probabilities."""


class MoGeV2GeometryGraphOutput(NamedTuple):
    """Full geometry-head tensors at network resolution."""

    points_bhw3: Float32[Tensor, "b nh nw 3"]
    """Raw affine point-head output."""
    normals_bhw3: Float32[Tensor, "b nh nw 3"]
    """Unit camera-space RDF normal-head output."""
    mask_bhw: Float32[Tensor, "b nh nw"]
    """Validity probabilities."""
    metric_scale_b: Float32[Tensor, "b"]
    """Positive metric-scale head output."""



def _resize_and_normalize_normals(
    normals_bnhw3: Float32[Tensor, "b nh nw 3"],
    output_hw: tuple[int, int],
) -> Float32[Tensor, "b h w 3"]:
    """Resize camera-space normals and restore unit length.

    Args:
        normals_bnhw3: Float32 camera-space normals shaped ``b nh nw 3``.
        output_hw: Caller output height and width.

    Returns:
        Owning float32 unit normals shaped ``b h w 3``.
    """
    normals_b3nhw: Float32[Tensor, "b 3 nh nw"] = rearrange(normals_bnhw3.float(), "b h w c -> b c h w")
    if tuple(normals_b3nhw.shape[-2:]) != output_hw:
        normals_b3hw: Float32[Tensor, "b 3 h w"] = F.interpolate(normals_b3nhw, size=output_hw, mode="bilinear", align_corners=False)
    else:
        normals_b3hw = normals_b3nhw
    normals_bhw3: Float32[Tensor, "b h w 3"] = rearrange(normals_b3hw, "b c h w -> b h w c")
    return F.normalize(normals_bhw3, dim=-1)


def postprocess_moge_v2_normal(
    graph_output: MoGeV2NormalGraphOutput,
    output_hw: tuple[int, int],
) -> MoGeV2NormalOutput:
    """Resize normal-only graph tensors and take ownership.

    Args:
        graph_output: Float32 normal and mask tensors at network resolution.
        output_hw: Caller output height and width.

    Returns:
        Owning float32 unit normals and validity probabilities at caller resolution.
    """
    normals_bhw3: Float32[Tensor, "b h w 3"] = _resize_and_normalize_normals(graph_output.normals_bhw3, output_hw)
    mask_b1nhw: Float32[Tensor, "b 1 nh nw"] = rearrange(graph_output.mask_bhw.float(), "b h w -> b 1 h w")
    if tuple(mask_b1nhw.shape[-2:]) != output_hw:
        mask_b1hw: Float32[Tensor, "b 1 h w"] = F.interpolate(mask_b1nhw, size=output_hw, mode="bilinear", align_corners=False)
    else:
        mask_b1hw = mask_b1nhw.clone()
    mask_bhw: Float32[Tensor, "b h w"] = rearrange(mask_b1hw, "b 1 h w -> b h w")
    return MoGeV2NormalOutput(normals_bhw3=normals_bhw3, mask_bhw=mask_bhw)


def postprocess_moge_v2_geometry(
    graph_output: MoGeV2GeometryGraphOutput,
    output_hw: tuple[int, int],
    *,
    fov_x: float | None = None,
) -> MoGeV2GeometryOutput:
    """Recover metric pinhole geometry and resize it to caller resolution.

    Args:
        graph_output: Float32 full-head outputs at network resolution.
        output_hw: Caller output height and width.
        fov_x: Known horizontal field of view in degrees of the caller image (a calibrated camera, ``2 atan(w / 2 fx)``;
            MoGe assumes a centred principal point), solved with the caller aspect ``output_hw``. None: MoGe recovers
            the focal together with the shift on the network grid's aspect.

    Returns:
        Owning float32 depth, points, normals, mask, normalized intrinsics,
        metric scale, and recovered shift tensors.
    """
    metric_scale_b: Float32[Tensor, "b"] = graph_output.metric_scale_b.float().clone()
    points_bnhw3: Float32[Tensor, "b nh nw 3"] = graph_output.points_bhw3.float() * metric_scale_b[:, None, None, None]
    normals_bhw3: Float32[Tensor, "b h w 3"] = _resize_and_normalize_normals(graph_output.normals_bhw3, output_hw)
    mask_bnhw: Float32[Tensor, "b nh nw"] = graph_output.mask_bhw.float()
    network_height: int = points_bnhw3.shape[1]
    network_width: int = points_bnhw3.shape[2]

    recovery_mask_bnhw: Bool[Tensor, "b nh nw"] = mask_bnhw > 0.5
    # A known focal belongs to the caller image, so its fov needs the caller aspect (upstream ``infer``); the recovered
    # focal path keeps the network grid's aspect.
    aspect_ratio: float = network_width / network_height if fov_x is None else output_hw[1] / output_hw[0]
    camera: CameraRecovery = recover_shift_and_intrinsics(
        graph_output.points_bhw3.float(),
        recovery_mask_bnhw,
        fov_x=fov_x,  # pyrefly: ignore[bad-argument-type]  # a float is a numbers.Number at runtime
        aspect_ratio=aspect_ratio,
    )
    shift_b: Float32[Tensor, "b"] = camera.shift_b.float()
    intrinsics_b33: Float32[Tensor, "b 3 3"] = camera.intrinsics_b33.float()
    points_bnhw3[..., 2] += shift_b[:, None, None] * metric_scale_b[:, None, None]
    positive_z_bnhw: Bool[Tensor, "b nh nw"] = points_bnhw3[..., 2] > 0.0
    mask_bnhw = torch.where(positive_z_bnhw, mask_bnhw, 0.0)

    points_b3nhw: Float32[Tensor, "b 3 nh nw"] = rearrange(points_bnhw3, "b h w c -> b c h w")
    mask_b1nhw: Float32[Tensor, "b 1 nh nw"] = rearrange(mask_bnhw, "b h w -> b 1 h w")
    if (network_height, network_width) != output_hw:
        points_b3hw: Float32[Tensor, "b 3 h w"] = F.interpolate(points_b3nhw, size=output_hw, mode="bilinear", align_corners=False)
        mask_b1hw: Float32[Tensor, "b 1 h w"] = F.interpolate(mask_b1nhw, size=output_hw, mode="bilinear", align_corners=False)
    else:
        points_b3hw = points_b3nhw
        mask_b1hw = mask_b1nhw

    points_bhw3: Float32[Tensor, "b h w 3"] = rearrange(points_b3hw, "b c h w -> b h w c").contiguous()
    depth_bhw: Float32[Tensor, "b h w"] = points_bhw3[..., 2]
    resized_mask_bhw: Float32[Tensor, "b h w"] = rearrange(mask_b1hw, "b 1 h w -> b h w")
    mask_bhw: Float32[Tensor, "b h w"] = torch.where(depth_bhw > 0.0, resized_mask_bhw, 0.0)
    return MoGeV2GeometryOutput(
        depth_bhw=depth_bhw,
        points_bhw3=points_bhw3,
        normals_bhw3=normals_bhw3,
        mask_bhw=mask_bhw,
        intrinsics_b33=intrinsics_b33,
        metric_scale_b=metric_scale_b,
        shift_b=shift_b,
    )


class _MoGeV2Predictor(ABC):
    """Typed entry points shared by the Torch and TensorRT cores."""

    heads: HeadSet

    @abstractmethod
    def _run_graph(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> dict[str, Tensor]:
        """Preprocess and run the configured heads; float32 tensors keyed by ONNX output name."""

    def _require_heads(self, heads: HeadSet) -> None:
        if self.heads != heads:
            raise ValueError(f"This predictor was built with heads={self.heads!r}; it cannot produce {heads!r} outputs.")

    def forward_normals(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2NormalGraphOutput:
        """Run the normal-only heads and return network-resolution tensors (TensorRT: may alias runtime buffers)."""
        self._require_heads("normal")
        graph: dict[str, Tensor] = self._run_graph(rgb_bhw3)
        return MoGeV2NormalGraphOutput(normals_bhw3=graph["normal"], mask_bhw=graph["mask"])

    def forward_geometry(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2GeometryGraphOutput:
        """Run the full geometry heads and return network-resolution tensors (TensorRT: may alias runtime buffers)."""
        self._require_heads("geometry")
        graph: dict[str, Tensor] = self._run_graph(rgb_bhw3)
        return MoGeV2GeometryGraphOutput(
            points_bhw3=graph["points"],
            normals_bhw3=graph["normal"],
            mask_bhw=graph["mask"],
            metric_scale_b=graph["metric_scale"],
        )

    def predict_normals(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> MoGeV2NormalOutput:
        """Predict owning unit normals and validity at caller resolution."""
        return postprocess_moge_v2_normal(self.forward_normals(rgb_bhw3), (rgb_bhw3.shape[1], rgb_bhw3.shape[2]))

    def predict_geometry(self, rgb_bhw3: UInt8[Tensor, "b h w 3"], *, fov_x: float | None = None) -> MoGeV2GeometryOutput:
        """Predict owning metric depth, points, normals, validity, and intrinsics at caller resolution.

        ``fov_x``: known horizontal field of view in degrees (``postprocess_moge_v2_geometry``); None recovers it.
        """
        return postprocess_moge_v2_geometry(self.forward_geometry(rgb_bhw3), (rgb_bhw3.shape[1], rgb_bhw3.shape[2]), fov_x=fov_x)


class MoGeV2TorchPredictor(_MoGeV2Predictor):
    """One plain-Torch core for normal-only or full-geometry prediction."""

    def __init__(
        self,
        encoder: Encoder = "vitl",
        heads: HeadSet = "geometry",
        network_hw_options: tuple[tuple[int, int], ...] | None = None,
        resolution_level: int = 9,
    ) -> None:
        """Load one pinned normal-capable MoGe v2 checkpoint on CUDA.

        Args:
            encoder: DINOv2 encoder size.
            heads: Normal-only or full-geometry head set.
            network_hw_options: Candidate static network sizes. ``None`` uses
                the exact token grid for each caller aspect.
            resolution_level: Detail level from 0 through 9.

        Raises:
            RuntimeError: If CUDA is unavailable.
            ValueError: If a supplied network-size collection is invalid.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("MoGe v2 batched prediction requires CUDA.")
        if network_hw_options is not None:
            if not network_hw_options:
                raise ValueError("network_hw_options must contain at least one size when provided.")
            if any(height < 1 or width < 1 for height, width in network_hw_options):
                raise ValueError(f"MoGe v2 network dimensions must be positive, got {network_hw_options}.")
        loaded: tuple[MoGeModel, int] = load_pinned_moge_v2(encoder, resolution_level)
        self.model: MoGeModel = loaded[0]
        self.num_tokens: int = loaded[1]
        self.encoder: Encoder = encoder
        self.heads: HeadSet = heads
        self.network_hw_options: tuple[tuple[int, int], ...] | None = network_hw_options
        self.resolution_level: int = resolution_level

    def _run_graph(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> dict[str, Tensor]:
        input_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
        network_hw: tuple[int, int] = (
            token_grid_hw(input_hw[1] / input_hw[0], self.num_tokens)
            if self.network_hw_options is None
            else select_aspect_bucket(input_hw, self.network_hw_options)
        )
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = preprocess_rgb(rgb_bhw3, network_hw)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output: ForwardOutput = self.model(image_b3hw, num_tokens=self.num_tokens, output_heads=FORWARD_HEADS[self.heads])
        return {name: output[name].float() for name in ONNX_OUTPUT_NAMES[self.heads]}


class MoGeV2TrtPredictor(_MoGeV2Predictor):
    """One cached TensorRT sibling for normal-only or full geometry heads.

    ``profile="dynamic"`` (default): one engine per GPU, encoder, head set and detail level serves every token grid of
    aspect 1:2 through 2:1 (``dynamic_patch_range``); each image runs at its own token grid, as the torch twin does.
    ``profile="static"``: one engine per network size, the size baked into the graph.
    """

    def __init__(
        self,
        encoder: Encoder = "vitl",
        heads: HeadSet = "geometry",
        network_hw_options: tuple[tuple[int, int], ...] | None = None,
        resolution_level: int = 9,
        batch_size: int = 8,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        hardware_compatibility: HardwareCompatibility = "none",
        use_cuda_graph: bool = False,
        builder_optimization_level: int = 5,
        profile: EngineProfile = "dynamic",
    ) -> None:
        """Configure the TensorRT runtime(s); one supplied network size builds (or loads) the engine eagerly.

        Args:
            encoder: DINOv2 encoder size.
            heads: Normal-only or full-geometry head set.
            network_hw_options: Candidate network sizes; each image takes the one nearest its aspect. ``None``: the
                image's own token grid (dynamic) or five aspect buckets (static). The dynamic profile runs the token
                grid of the chosen size's aspect, which is the grid MoGe itself derives from that size.
            resolution_level: Detail level from 0 through 9.
            batch_size: Engine profile maximum and optimization batch.
            cache_dir: Cache root for ONNX and TensorRT artifacts.
            hardware_compatibility: TensorRT plan portability level.
            use_cuda_graph: Whether runtimes capture launches in CUDA graphs (dynamic: one graph per input shape).
            builder_optimization_level: TensorRT builder optimization level.
            profile: ``dynamic`` (one engine over a range of token grids) or ``static`` (one engine per network size).

        Raises:
            ValueError: If batch size or supplied network sizes are invalid, or a supplied size is outside the
                dynamic profile.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if network_hw_options is not None:
            if not network_hw_options:
                raise ValueError("network_hw_options must contain at least one size when provided.")
            if any(height < 1 or width < 1 for height, width in network_hw_options):
                raise ValueError(f"MoGe v2 network dimensions must be positive, got {network_hw_options}.")
        self.num_tokens: int = resolve_num_tokens(resolution_level)
        self.encoder: Encoder = encoder
        self.heads: HeadSet = heads
        self.profile: EngineProfile = profile
        self.network_hw_options: tuple[tuple[int, int], ...] | None = (
            aspect_buckets(self.num_tokens) if network_hw_options is None and profile == "static" else network_hw_options
        )
        self.resolution_level: int = resolution_level
        self.batch_size: int = batch_size
        self.cache_dir: Path = cache_dir
        self.hardware_compatibility: HardwareCompatibility = hardware_compatibility
        self.use_cuda_graph: bool = use_cuda_graph
        self.builder_optimization_level: int = builder_optimization_level
        self.patch_range: tuple[int, int] = dynamic_patch_range(self.num_tokens)
        """Per-axis token-grid range of the dynamic engine, in patches."""
        self.runtimes: dict[tuple[int, int] | None, TensorRuntime | TensorRtDynamicRuntime] = {}
        """Built runtimes by network size; the dynamic engine's key is None."""
        if network_hw_options is not None:
            for option_hw in network_hw_options:
                self.network_hw(option_hw)  # fail at construction on a size the dynamic engine cannot serve
            if len(network_hw_options) == 1:
                self._ensure_runtime(self.network_hw(network_hw_options[0]))

    def network_hw(self, input_hw: tuple[int, int]) -> tuple[int, int]:
        """The network size an ``input_hw`` image runs at.

        Raises:
            ValueError: If the dynamic engine's range does not hold that token grid (an aspect beyond 1:2 or 2:1).
        """
        selected_hw: tuple[int, int] = input_hw if self.network_hw_options is None else select_aspect_bucket(input_hw, self.network_hw_options)
        if self.profile == "static":
            return selected_hw
        grid_hw: tuple[int, int] = token_grid_hw(selected_hw[1] / selected_hw[0], self.num_tokens)
        low, high = self.patch_range
        if not (low <= grid_hw[0] // 14 <= high and low <= grid_hw[1] // 14 <= high):
            raise ValueError(
                f"MoGe v2 token grid {grid_hw[0]}x{grid_hw[1]} of a {input_hw[0]}x{input_hw[1]} image is outside the dynamic engine "
                f"({14 * low}-{14 * high} px per side, aspects 1:2 to 2:1); use profile='static' for it."
            )
        return grid_hw

    def _engine_paths(self, network_hw: tuple[int, int]) -> tuple[Path, TrtBuildConfig]:
        """ONNX path and build settings (the engine cache key) of the engine that serves ``network_hw``."""
        from trtkit import InputShapeProfile, TrtBuildConfig

        if self.profile == "static":
            onnx_path: Path = moge_v2_onnx_path(self.heads, self.encoder, network_hw, self.resolution_level, self.cache_dir)
            shape_profiles: tuple[InputShapeProfile, ...] = ()
        else:
            onnx_path = moge_v2_onnx_path(self.heads, self.encoder, None, self.resolution_level, self.cache_dir)
            low, high = self.patch_range
            opt_hw: tuple[int, int] = token_grid_hw(4.0 / 3.0, self.num_tokens)
            shape_profiles = (
                InputShapeProfile(
                    name="image",
                    min_shape=(1, 3, 14 * low, 14 * low),
                    opt_shape=(self.batch_size, 3, *opt_hw),
                    max_shape=(self.batch_size, 3, 14 * high, 14 * high),
                ),
            )
        config: TrtBuildConfig = TrtBuildConfig(
            max_batch_size=self.batch_size,
            opt_batch_size=self.batch_size,
            builder_optimization_level=self.builder_optimization_level,
            hardware_compatibility=self.hardware_compatibility,
            runtime_activation_resize=self.profile == "dynamic",
            shape_profiles=shape_profiles,
        )
        return onnx_path, config

    def has_engine(self, input_hw: tuple[int, int]) -> bool:
        """Whether the engine for an ``input_hw`` image is already built on this machine; checks the caches only (no
        export, no build), so a caller can fall back to the torch predictor instead of waiting for a build. Dynamic:
        true for every in-range size once the one engine exists; an out-of-range size is false. False in an
        environment without trtkit."""
        try:
            from trtkit.trt_builder import cached_engine_path, onnx_content_hash
        except ImportError:
            return False

        try:
            network_hw: tuple[int, int] = self.network_hw(input_hw)
        except ValueError:
            return False
        onnx_path, config = self._engine_paths(network_hw)
        if not onnx_path.exists():
            return False
        trt_dir: Path = self.cache_dir / "trt"
        # The digest record lives in this predictor's cache, as ensure_engine keeps it (else the whole ONNX is hashed again).
        digest: str = onnx_content_hash(onnx_path, cache_dir=trt_dir)
        return cached_engine_path(onnx_path, config, cache_dir=trt_dir, onnx_sha256=digest).exists()

    def _ensure_runtime(self, network_hw: tuple[int, int]) -> TensorRuntime | TensorRtDynamicRuntime:
        """Build or load the runtime that serves one network size."""
        key: tuple[int, int] | None = network_hw if self.profile == "static" else None
        existing_runtime: TensorRuntime | TensorRtDynamicRuntime | None = self.runtimes.get(key)
        if existing_runtime is not None:
            return existing_runtime

        from trtkit import ensure_engine

        onnx_path, config = self._engine_paths(network_hw)
        export_moge_v2_onnx(
            heads=self.heads,
            encoder=self.encoder,
            image_hw=key,
            resolution_level=self.resolution_level,
            max_batch_size=self.batch_size if self.profile == "static" else DYNAMIC_ONNX_MAX_BATCH,
            cache_dir=self.cache_dir,
        )
        engine_path: Path = ensure_engine(onnx_path, config, cache_dir=self.cache_dir / "trt")

        runtime: TensorRuntime | TensorRtDynamicRuntime
        if self.profile == "static":
            from trtkit.tensorrt_runtime import TensorRtRuntime

            runtime = TensorRtRuntime(engine_path, use_cuda_graph=self.use_cuda_graph)
        else:
            from trtkit import TensorRtDynamicRuntime as DynamicRuntime

            # Direct binding: no copy into max-shape buffers; outputs are views valid until the next call, the
            # contract forward_* already states. Activation memory sized per shape: the profile's max corner
            # (1190x1190) needs twice what a real token grid does (10.5 vs 5.2 GiB at batch 8). CUDA graphs need
            # fixed addresses, so they take the copying path with max-shape memory.
            runtime = DynamicRuntime(
                engine_path,
                use_cuda_graph=self.use_cuda_graph,
                direct_binding=not self.use_cuda_graph,
                per_shape_activation_memory=not self.use_cuda_graph,
            )
        self.runtimes[key] = runtime
        return runtime

    def _run_graph(self, rgb_bhw3: UInt8[Tensor, "b h w 3"]) -> dict[str, Tensor]:
        network_hw: tuple[int, int] = self.network_hw((rgb_bhw3.shape[1], rgb_bhw3.shape[2]))
        runtime: TensorRuntime | TensorRtDynamicRuntime = self._ensure_runtime(network_hw)
        image_b3hw: Float32[Tensor, "b 3 nh nw"] = preprocess_rgb(rgb_bhw3, network_hw).contiguous()
        runtime_output: dict[str, Tensor] = runtime({"image": image_b3hw})
        return {name: runtime_output[name].float() for name in ONNX_OUTPUT_NAMES[self.heads]}
