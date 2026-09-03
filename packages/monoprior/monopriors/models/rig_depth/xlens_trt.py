"""TensorRT X-Lens predictor: a static-rig fp16 engine fed with images plus frozen geometry.

The engine bakes the view count, resolution, camera types, and pose presence of
one rig. Everything geometric that is large (ray-map features, RoPE positions,
the combined per-head attention bias) stays outside the graph as persistent
device tensors handed to the engine each call, so a ~1 GB bias never lives in
an ONNX file or an engine plan.
"""

import copy
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor, nn

from monopriors.models.rig_depth.base_rig_depth import BaseRigDepthPredictor, BaseRigDepthPredictorConfig, RigDepthPrediction
from monopriors.models.rig_depth.xlens import RigKey, RigKeyMemo, RigTensors, load_xlens_model, normalize_images, rig_tensors, validate_rig_inputs
from monopriors.third_party.xlens.models.dinov2.vision_transformer import AttentionBias, FrozenRigGeometry
from monopriors.third_party.xlens.models.net import ScaleHead, XLensNet, XLensNetOutput

if TYPE_CHECKING:
    from trtkit.base import TensorRuntime
else:
    TensorRuntime = object

ModuleT = TypeVar("ModuleT", bound=nn.Module)

XLENS_CACHE_DIR: Path = Path(os.environ.get("MONOPRIOR_TRT_CACHE", "~/.cache/monoprior")).expanduser()
"""Cache root holding portable ``onnx/`` and machine-local ``trt/`` artifacts."""

ONNX_EXPORT_VERSION: int = 1
"""Bump when the export graph or its input contract changes."""

ENGINE_OUTPUT_NAMES: tuple[str, ...] = ("depth_metric", "depth_conf", "mask", "metric_scaling_factor")
"""Engine outputs in wrapper order."""

MASK_FLOOR: float = -1.0e4
"""Additive-mask floor replacing float32 ``finfo.min`` before the fp16 cast.

``exp(-1e4)`` is exactly zero in fp16 and fp32, so allowed/blocked attention is
unchanged; the float32 minimum would become ``-inf`` in fp16 and turn fully
blocked placeholder rows into NaN instead of the uniform rows eager produces
(those rows belong to tokens that are dropped after every layer).
"""


@dataclass(frozen=True, slots=True)
class EngineGeometry:
    """One rig's frozen geometry flattened into named engine inputs."""

    inputs: dict[str, Tensor]
    """Engine inputs by name: optional ``ray_feat``, ``pos_local``, ``pos_global``, and ``attn_bias_<i>`` biases."""
    layer_slots: tuple[str | None, ...]
    """Attention-bias input feeding each backbone layer; None where the layer has no bias (all-zero masks are dropped)."""
    local_slots: frozenset[str]
    """Bias inputs shaped ``(1, views, L, L)`` for within-view attention; the rest are ``(1, heads, L, L)`` cross-view biases."""

    @property
    def names(self) -> tuple[str, ...]:
        """Geometry input names in engine order."""
        return tuple(self.inputs)


def engine_geometry(frozen: FrozenRigGeometry, bias_dtype: torch.dtype = torch.float16) -> EngineGeometry:
    """Flatten frozen geometry into engine inputs with a leading batch dimension of one.

    Args:
        frozen: Rig geometry from ``XLensNet.freeze_geometry`` (float FishRoPE positions required).
        bias_dtype: Storage dtype of the attention biases; fp16 halves the largest input.

    Returns:
        Named inputs plus the per-layer slot map the export graph bakes in.

    Raises:
        ValueError: If the rig has no float RoPE positions (no per-pixel rays).
    """
    if frozen.pos_local is None or frozen.pos_global is None or not frozen.pos_local.is_floating_point():
        raise ValueError("X-Lens TensorRT export needs float FishRoPE positions; freeze the geometry with per-pixel rays")
    inputs: dict[str, Tensor] = {}
    if frozen.ray_feat is not None:
        inputs["ray_feat"] = frozen.ray_feat.float().contiguous()
    inputs["pos_local"] = frozen.pos_local.float().contiguous()
    inputs["pos_global"] = frozen.pos_global.float().contiguous()
    slots: list[str | None] = []
    local_slots: set[str] = set()
    named: dict[int, str | None] = {}
    for mask in frozen.attn_masks:
        if mask is None:
            slots.append(None)
            continue
        if id(mask) not in named:
            if not bool(mask.any()):
                # An all-zero calibration mask (every view injects) adds nothing; drop it.
                named[id(mask)] = None
            else:
                name: str = f"attn_bias_{sum(1 for value in named.values() if value is not None)}"
                clamped: AttentionBias = mask.clamp(min=MASK_FLOOR).to(bias_dtype).contiguous()
                if clamped.dim() == 3:
                    inputs[name] = clamped[None]
                    local_slots.add(name)
                else:
                    inputs[name] = clamped
                named[id(mask)] = name
        slots.append(named[id(mask)])
    return EngineGeometry(inputs=inputs, layer_slots=tuple(slots), local_slots=frozenset(local_slots))


class _ManualMultiheadAttention(nn.Module):
    """``nn.MultiheadAttention`` (batch-first, no masks) as explicit projections plus scaled-dot-product attention.

    The eval fast path emits ``aten::_native_multi_head_attention``, which the
    exporter cannot lower; this keeps the same weights and math.
    """

    def __init__(self, inner: nn.MultiheadAttention) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self,
        query: Float[Tensor, "batch queries features"],
        key: Float[Tensor, "batch tokens features"],
        value: Float[Tensor, "batch tokens features"],
        need_weights: bool = False,
    ) -> tuple[Float[Tensor, "batch queries features"], None]:
        """Cross-attend queries over key/value tokens."""
        features: int = query.shape[-1]
        weight: Float[Tensor, "features3 features"] = self.inner.in_proj_weight
        bias: Float[Tensor, "features3"] = self.inner.in_proj_bias
        heads: int = self.inner.num_heads
        q: Float[Tensor, "batch heads queries head_features"] = rearrange(
            F.linear(query, weight[:features], bias[:features]), "b l (h d) -> b h l d", h=heads
        )
        k: Float[Tensor, "batch heads tokens head_features"] = rearrange(
            F.linear(key, weight[features : 2 * features], bias[features : 2 * features]), "b l (h d) -> b h l d", h=heads
        )
        v: Float[Tensor, "batch heads tokens head_features"] = rearrange(
            F.linear(value, weight[2 * features :], bias[2 * features :]), "b l (h d) -> b h l d", h=heads
        )
        pooled: Float[Tensor, "batch heads queries head_features"] = F.scaled_dot_product_attention(q, k, v)
        return self.inner.out_proj(rearrange(pooled, "b h l d -> b l (h d)")), None


def _with_manual_scale_attention(model: XLensNet) -> XLensNet:
    """Structural copy of the network whose scale head uses explicit attention; weights are shared, the original is untouched."""
    scale_head: ScaleHead = model.scale_head
    if scale_head.mode != "attn_pool":
        return model
    head_copy: ScaleHead = _shallow_module_copy(scale_head)
    head_copy.attn_layers = nn.ModuleList([_ManualMultiheadAttention(cast(nn.MultiheadAttention, layer)) for layer in scale_head.attn_layers])
    model_copy: XLensNet = _shallow_module_copy(model)
    model_copy.scale_head = head_copy
    return model_copy


def _shallow_module_copy(module: ModuleT) -> ModuleT:
    """Copy a module object so child assignments do not touch the original tree."""
    clone: ModuleT = copy.copy(module)
    clone._modules = dict(module._modules)
    return clone


class _XLensRigGraph(nn.Module):
    """Static-rig X-Lens graph: normalised images plus frozen geometry in, dense outputs out."""

    cam_types: Int64[Tensor, "1 s"]
    """Baked camera type ids (a registered buffer)."""

    def __init__(self, model: XLensNet, frozen: FrozenRigGeometry, geometry: EngineGeometry) -> None:
        super().__init__()
        if frozen.cam_types is None:
            raise ValueError("X-Lens TensorRT export needs camera types")
        self.model: XLensNet = _with_manual_scale_attention(model)
        self.register_buffer("cam_types", frozen.cam_types.clone())
        self.calib_k: int = frozen.calib_k
        self.input_names: tuple[str, ...] = geometry.names
        self.layer_slots: tuple[str | None, ...] = geometry.layer_slots
        self.local_slots: frozenset[str] = geometry.local_slots

    def forward(self, images: Float32[Tensor, "1 s 3 h w"], *geometry: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the frozen-rig network; ``geometry`` follows ``input_names`` order."""
        by_name: dict[str, Tensor] = dict(zip(self.input_names, geometry, strict=True))
        masks: list[AttentionBias | None] = []
        for slot in self.layer_slots:
            if slot is None:
                masks.append(None)
                continue
            bias: Tensor = by_name[slot]
            if slot in self.local_slots:
                # (1, views, L, L) -> one per-view mask broadcast over heads.
                bias = rearrange(bias, "1 v l m -> v 1 l m")  # pyrefly: ignore  # bad-argument-type — einops stub false positive
            masks.append(bias)
        frozen = FrozenRigGeometry(
            cam_types=self.cam_types,
            ray_feat=by_name.get("ray_feat"),
            calib_k=self.calib_k,
            pos_local=by_name["pos_local"],
            pos_global=by_name["pos_global"],
            attn_masks=tuple(masks),
        )
        output: XLensNetOutput = self.model(images, frozen=frozen)
        return output["depth_metric"], output["depth_conf"], output["mask"], output["metric_scaling_factor"]


def rig_signature(cam_types: Int64[ndarray, "s"], image_hw: tuple[int, int], has_poses: bool) -> str:
    """Filename fragment naming everything the graph bakes: views, resolution, camera types, pose presence."""
    cam_key: str = "".join(str(int(value)) for value in cam_types)
    return f"{len(cam_types)}v_{image_hw[0]}x{image_hw[1]}_ct{cam_key}_{'rays' if has_poses else 'norays'}"


def export_xlens_onnx(model: XLensNet, frozen: FrozenRigGeometry, geometry: EngineGeometry, image_hw: tuple[int, int], onnx_path: Path) -> None:
    """Export the static-rig fp16 graph with trtkit's strongly-typed recipe.

    Args:
        model: Released X-Lens network on CUDA.
        frozen: Rig geometry the graph structure is derived from.
        geometry: Engine inputs for this rig (traced as example values).
        image_hw: Network image height and width.
        onnx_path: Destination ONNX path, published atomically by trtkit.
    """
    from trtkit import export_onnx

    graph: _XLensRigGraph = _XLensRigGraph(model, frozen, geometry).eval()
    n_views: int = int(geometry.inputs["pos_local"].shape[1])
    example_images: Float32[Tensor, "1 s 3 h w"] = torch.zeros((1, n_views, 3, image_hw[0], image_hw[1]), dtype=torch.float32, device="cuda")
    print(f"[monoprior] exporting X-Lens rig graph to ONNX (one-time, may take minutes): {onnx_path.name}")
    export_onnx(
        graph,
        (example_images, *(geometry.inputs[name] for name in geometry.names)),
        onnx_path,
        input_names=["images", *geometry.names],
        output_names=list(ENGINE_OUTPUT_NAMES),
        compute_dtype=torch.float16,
    )


@dataclass
class XLensTrtConfig(BaseRigDepthPredictorConfig):
    """Configuration for the fp16 TensorRT X-Lens ViT-S engine (one engine per rig layout)."""

    checkpoint: Path | None = None
    """Local safetensors state dict, or None to download the pinned gated release."""
    use_cuda_graph: bool = True
    """Capture the engine launch in a CUDA graph and replay it."""
    cache_dir: Path = field(default_factory=lambda: XLENS_CACHE_DIR)
    """Cache root for ONNX exports and machine-local engines."""
    workspace_gib: float = 8.0
    """TensorRT builder workspace cap in GiB."""

    def setup(self, device: Literal["cpu", "cuda"]) -> "XLensTrtPredictor":
        """Build the TensorRT predictor; only CUDA is supported."""
        if device != "cuda":
            raise ValueError("the X-Lens TensorRT predictor requires device='cuda'")
        return XLensTrtPredictor(
            checkpoint=self.checkpoint, use_cuda_graph=self.use_cuda_graph, cache_dir=self.cache_dir, workspace_gib=self.workspace_gib
        )


class XLensTrtPredictor(BaseRigDepthPredictor):
    """Released X-Lens weights as a cached TensorRT engine fed with persistent rig geometry."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        use_cuda_graph: bool = True,
        cache_dir: Path = XLENS_CACHE_DIR,
        workspace_gib: float = 8.0,
    ) -> None:
        """Load the eager model (it computes each rig's geometry and exports the graph); engines build lazily per rig layout.

        Args:
            checkpoint: Local released state dict; downloaded with the user's Hugging Face login when None.
            use_cuda_graph: Replay the engine launch through a CUDA graph.
            cache_dir: Cache root for ONNX exports and engines.
            workspace_gib: TensorRT builder workspace cap.

        Raises:
            RuntimeError: If CUDA is unavailable.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("the X-Lens TensorRT predictor requires CUDA")
        loaded: tuple[XLensNet, Path] = load_xlens_model(checkpoint, "cuda")
        self.model: XLensNet = loaded[0]
        self.checkpoint_digest: str = hashlib.sha256(loaded[1].read_bytes()).hexdigest()
        self.use_cuda_graph: bool = use_cuda_graph
        self.cache_dir: Path = cache_dir
        self.workspace_gib: float = workspace_gib
        self._memo: RigKeyMemo = RigKeyMemo()
        self._rig_key: RigKey | None = None
        self._geometry: EngineGeometry | None = None
        self._signature: str | None = None
        self._runtime: TensorRuntime | None = None
        self.engine_path: Path | None = None
        """Machine-local engine serving the current rig layout; None before the first call."""

    def onnx_path(self, signature: str) -> Path:
        """Portable ONNX path for one rig layout of this checkpoint."""
        return self.cache_dir / "onnx" / f"xlens-vits-rig_{signature}_v{ONNX_EXPORT_VERSION}_{self.checkpoint_digest[:8]}.onnx"

    def prepare(
        self,
        rays: Float32[ndarray, "s h w 3"],
        cam_types: Int64[ndarray, "s"],
        cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    ) -> tuple[EngineGeometry, TensorRuntime]:
        """Return the engine inputs and runtime for one rig, freezing geometry and building the engine when they change."""
        key: RigKey = self._memo.key(rays, cam_types, cam_T_ref)
        if self._geometry is not None and self._runtime is not None and key == self._rig_key:
            return self._geometry, self._runtime
        image_hw: tuple[int, int] = (int(rays.shape[1]), int(rays.shape[2]))
        signature: str = rig_signature(cam_types, image_hw, cam_T_ref is not None)
        tensors: RigTensors = rig_tensors(rays, cam_types, cam_T_ref, torch.device("cuda"))
        with torch.inference_mode():
            frozen: FrozenRigGeometry = self.model.freeze_geometry(tensors.d_cam, tensors.cam_types, tensors.ray_map)
        geometry: EngineGeometry = engine_geometry(frozen)
        if self._runtime is None or signature != self._signature:
            from trtkit import TensorRtRuntime, TrtBuildConfig, ensure_engine

            self._runtime = None
            onnx_path: Path = self.onnx_path(signature)
            if not onnx_path.exists():
                export_xlens_onnx(self.model, frozen, geometry, image_hw, onnx_path)
            del frozen
            torch.cuda.empty_cache()
            build_config: TrtBuildConfig = TrtBuildConfig(max_batch_size=1, opt_batch_size=1, workspace_gib=self.workspace_gib)
            engine_path: Path = ensure_engine(onnx_path, build_config, cache_dir=self.cache_dir / "trt")
            self._runtime = TensorRtRuntime(engine_path, use_cuda_graph=self.use_cuda_graph)
            self.engine_path = engine_path
            self._signature = signature
        self._geometry = geometry
        self._rig_key = key
        return geometry, self._runtime

    def __call__(
        self,
        images: UInt8[ndarray, "s h w 3"],
        rays: Float32[ndarray, "s h w 3"],
        cam_types: Int64[ndarray, "s"],
        cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    ) -> RigDepthPrediction:
        """Predict per-view camera-frame metric z-depth with the cached engine.

        Args:
            images: Shared-resolution RGB views, ``UInt8[ndarray, "s h w 3"]``.
            rays: Camera-frame unit rays, ``Float32[ndarray, "s h w 3"]``; must not be mutated in place between calls.
            cam_types: X-Lens camera ids, ``Int64[ndarray, "s"]``.
            cam_T_ref: Optional camera-to-reference poses, ``Float64[ndarray, "s 4 4"]``.

        Returns:
            Owning metric depth, confidence, mask, and scale on CUDA.
        """
        validate_rig_inputs(images, rays, cam_types)
        geometry, runtime = self.prepare(rays, cam_types, cam_T_ref)
        image_tensor: Float32[Tensor, "1 s 3 h w"] = normalize_images(images, torch.device("cuda"))
        outputs: dict[str, Tensor] = runtime({"images": image_tensor, **geometry.inputs})
        return RigDepthPrediction(
            depth_m=outputs["depth_metric"][0].float().clone(),
            confidence=outputs["depth_conf"][0].float().clone(),
            mask=outputs["mask"][0].float().clone(),
            scale=float(outputs["metric_scaling_factor"][0]),
        )
