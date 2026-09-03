"""TensorRT X-Lens predictor: an fp16 engine fed with images plus frozen rig geometry.

Two engine profiles, both with a dynamic frameset batch:

- ``dynamic`` (default): one engine spanning a range of view counts and
  resolutions, tuned at four 896x504 views. Every geometry input carries its
  own symbolic token dimension; the wrapper slices each to the token count the
  image shape implies, and the TensorRT profile pins the ranges.
- ``rig``: one engine per rig layout (view count, resolution, camera types,
  pose presence); the ranges collapse to that rig, which buys a few
  milliseconds per call at the price of a rebuild for every new rig.

Everything geometric that is large (ray-map features, RoPE positions, the
combined per-head attention bias) stays outside the graph as persistent device
tensors handed to the engine each call, so a ~1 GB bias never lives in an ONNX
file or an engine plan.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor, nn

from monopriors.models.rig_depth.base_rig_depth import BaseRigDepthPredictor, BaseRigDepthPredictorConfig, RigDepthPrediction
from monopriors.models.rig_depth.xlens import (
    RigKey,
    RigKeyMemo,
    RigTensors,
    load_xlens_model,
    normalize_framesets,
    rig_depth_prediction,
    rig_tensors,
    validate_rig_inputs,
)
from monopriors.third_party.xlens.models.dinov2.vision_transformer import AttentionBias, DinoVisionTransformer, FrozenRigGeometry
from monopriors.third_party.xlens.models.dpt_head import DPTHead
from monopriors.third_party.xlens.models.net import ScaleHead, XLensNet, XLensNetOutput
from monopriors.third_party.xlens.models.utils.head_utils import position_grid_to_embed

if TYPE_CHECKING:
    from trtkit import DynamicDim, DynamicDims, InputShapeProfile, TensorRtDynamicRuntime
else:
    DynamicDim = object
    DynamicDims = dict
    InputShapeProfile = object
    TensorRtDynamicRuntime = object

EngineProfile: TypeAlias = Literal["rig", "dynamic"]
"""``dynamic`` (default) spans view counts and resolutions; ``rig`` bakes one rig layout (its ranges collapse to the rig)."""

XLENS_CACHE_DIR: Path = Path(os.environ.get("MONOPRIOR_TRT_CACHE", "~/.cache/monoprior")).expanduser()
"""Cache root holding portable ``onnx/`` and machine-local ``trt/`` artifacts."""

ONNX_EXPORT_VERSION: int = 2
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

DYNAMIC_OPT_VIEWS: int = 4
"""View count the dynamic profile tunes kernels for (clamped into the configured range)."""
DYNAMIC_OPT_HW: tuple[int, int] = (504, 896)
"""Resolution the dynamic profile tunes kernels for (clamped into the configured range)."""


@dataclass(frozen=True, slots=True)
class EngineGeometry:
    """One rig's frozen geometry flattened into named engine inputs (leading dim 1, never batched)."""

    inputs: dict[str, Tensor]
    """Inputs by name: optional ``ray_feat``, then ``pos_embed``, ``pos_local``, ``pos_global``, ``cam_types``, and ``attn_bias_<i>``."""
    layer_slots: tuple[str | None, ...]
    """Attention-bias input feeding each backbone layer; None where the layer has no bias (all-zero masks are dropped)."""
    slot_kinds: dict[str, tuple[bool, int]]
    """Per bias input: whether it serves cross-view layers (shape ``(1, heads, L, L)``; within-view ones are ``(1, views, L, L)``)
    and how many non-patch tokens each view carries there (the scale token from ``alt_start`` plus the calibration tokens)."""
    trailing: int
    """Scale plus calibration tokens appended to each view's patch tokens."""

    @property
    def names(self) -> tuple[str, ...]:
        """Geometry input names in engine order."""
        return tuple(self.inputs)


def engine_geometry(frozen: FrozenRigGeometry, backbone: DinoVisionTransformer, bias_dtype: torch.dtype = torch.float16) -> EngineGeometry:
    """Flatten frozen geometry into engine inputs with a leading dimension of one.

    Args:
        frozen: Rig geometry from ``XLensNet.freeze_geometry`` (float FishRoPE positions required).
        backbone: The network's backbone, which says which layers attend across views and where the scale token starts.
        bias_dtype: Storage dtype of the attention biases; fp16 halves the largest input.

    Returns:
        Named inputs plus the per-layer slot map and per-slot token facts the export graph bakes in.

    Raises:
        ValueError: If the rig has no float RoPE positions (no per-pixel rays), camera types, or position embedding, or a
            bias mask's layout disagrees with the layer type it feeds.
    """
    if frozen.pos_local is None or frozen.pos_global is None or not frozen.pos_local.is_floating_point():
        raise ValueError("X-Lens TensorRT export needs float FishRoPE positions; freeze the geometry with per-pixel rays")
    if frozen.cam_types is None or frozen.pos_embed is None:
        raise ValueError("X-Lens TensorRT export needs camera types and a frozen position embedding")
    inputs: dict[str, Tensor] = {}
    if frozen.ray_feat is not None:
        inputs["ray_feat"] = frozen.ray_feat.float().contiguous()
    inputs["pos_embed"] = frozen.pos_embed.float().contiguous()
    inputs["pos_local"] = frozen.pos_local.float().contiguous()
    inputs["pos_global"] = frozen.pos_global.float().contiguous()
    inputs["cam_types"] = frozen.cam_types.to(torch.int64).contiguous()
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
    # Per slot, the layer type decides the mask layout the graph slices: cross-view layers take the per-head bias,
    # within-view layers the per-view one. The per-view token count of a layer is 1 (CLS) + patches + extra.
    kinds: dict[str, tuple[bool, int]] = {}
    for layer_index, slot in enumerate(slots):
        if slot is None or slot in kinds:
            continue
        is_global: bool = backbone._is_global_layer(layer_index)
        if is_global == (slot in local_slots):
            raise ValueError(f"{slot} feeds a {'cross-view' if is_global else 'within-view'} layer but carries a {'per-view' if is_global else 'per-head'} mask")
        scale_injected: bool = backbone.alt_start != -1 and layer_index >= backbone.alt_start
        kinds[slot] = (is_global, (1 if scale_injected else 0) + frozen.calib_k)
    trailing: int = (1 if backbone.alt_start != -1 else 0) + frozen.calib_k
    return EngineGeometry(inputs=inputs, layer_slots=tuple(slots), slot_kinds=kinds, trailing=trailing)


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


def _linspace_tensor(start: Float[Tensor, ""], end: Float[Tensor, ""], steps: int, dtype: torch.dtype, device: torch.device) -> Float[Tensor, "steps"]:
    """``torch.linspace`` with tensor endpoints and a traced (symbolic) ``steps``: the kernel's two-sided formula.

    ``torch.linspace`` pins ``steps`` to a constant under ``torch.export`` and the
    ONNX exporter cannot interpret symbolic float arithmetic on sizes, so every
    size-dependent value is a tensor here. Values match the kernel up to
    fused-multiply-add rounding.
    """
    index: Float[Tensor, "steps"] = torch.arange(steps, device=device, dtype=dtype)
    count: Float[Tensor, ""] = index[-1] + 1.0
    step: Float[Tensor, ""] = (end - start) / torch.clamp(count - 1.0, min=1.0)
    halfway: Float[Tensor, ""] = torch.clamp(torch.floor(count / 2.0), min=1.0)
    return torch.where(index < halfway, index * step + start, end - (count - 1.0 - index) * step)


class _ExportDPTHead(DPTHead):
    """DPT head whose positional UV grid is traced with symbolic sizes (export only; eager keeps ``create_uv_grid``)."""

    def _add_pos_embed(self, x: Float[Tensor, "batch channels height width"], W: int, H: int, ratio: float = 0.1) -> Float[Tensor, "batch channels height width"]:
        """Add the low-amplitude sinusoidal embedding without ``torch.linspace`` or symbolic float math."""
        width: int = x.shape[-1]
        height: int = x.shape[-2]
        # Sizes as tensors (never SymFloat): the last arange element plus one.
        width_t: Float[Tensor, ""] = torch.arange(W, device=x.device, dtype=x.dtype)[-1] + 1.0
        height_t: Float[Tensor, ""] = torch.arange(H, device=x.device, dtype=x.dtype)[-1] + 1.0
        grid_w: Float[Tensor, ""] = torch.arange(width, device=x.device, dtype=x.dtype)[-1] + 1.0
        grid_h: Float[Tensor, ""] = torch.arange(height, device=x.device, dtype=x.dtype)[-1] + 1.0
        aspect_ratio: Float[Tensor, ""] = width_t / height_t
        diag_factor: Float[Tensor, ""] = torch.sqrt(aspect_ratio * aspect_ratio + 1.0)
        span_x: Float[Tensor, ""] = aspect_ratio / diag_factor
        span_y: Float[Tensor, ""] = 1.0 / diag_factor
        right_x: Float[Tensor, ""] = span_x * (grid_w - 1.0) / grid_w
        bottom_y: Float[Tensor, ""] = span_y * (grid_h - 1.0) / grid_h
        x_coords: Float[Tensor, "width"] = _linspace_tensor(-right_x, right_x, width, x.dtype, x.device)
        y_coords: Float[Tensor, "height"] = _linspace_tensor(-bottom_y, bottom_y, height, x.dtype, x.device)
        uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
        grid: Float[Tensor, "height width 2"] = torch.stack((uu, vv), dim=-1)
        pe: Float[Tensor, "height width channels"] = position_grid_to_embed(grid, x.shape[1]) * ratio
        return x + pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)


def _export_friendly(model: XLensNet) -> XLensNet:
    """Structural copy of the network for export: explicit scale-head attention and a symbolic-size DPT grid.

    Weights are shared with the caller's model, whose module tree is never mutated.
    """
    from trtkit import shallow_module_copy

    model_copy: XLensNet = shallow_module_copy(model)
    scale_head: ScaleHead = model.scale_head
    if scale_head.mode == "attn_pool":
        head_copy: ScaleHead = shallow_module_copy(scale_head)
        head_copy.attn_layers = nn.ModuleList([_ManualMultiheadAttention(cast(nn.MultiheadAttention, layer)) for layer in scale_head.attn_layers])
        model_copy.scale_head = head_copy
    dpt_copy: DPTHead = shallow_module_copy(model.head)
    dpt_copy.__class__ = _ExportDPTHead
    model_copy.head = dpt_copy
    return model_copy


class _XLensRigGraph(nn.Module):
    """X-Lens graph over a batch of framesets: normalised images plus batch-1 rig geometry in, dense outputs out.

    Geometry inputs are sliced to the token counts the image shape implies (a
    no-op for a fixed rig, the shape contract for the dynamic profile) and
    broadcast over the frameset batch: per-view tensors are expanded, the
    cross-view bias stays batch-1 and broadcasts inside attention.
    """

    def __init__(self, model: XLensNet, frozen: FrozenRigGeometry, geometry: EngineGeometry) -> None:
        super().__init__()
        self.model: XLensNet = _export_friendly(model)
        self.patch_size: int = model.backbone.pretrained.patch_size
        self.calib_k: int = frozen.calib_k
        self.trailing: int = geometry.trailing
        self.input_names: tuple[str, ...] = geometry.names
        self.layer_slots: tuple[str | None, ...] = geometry.layer_slots
        self.slot_kinds: dict[str, tuple[bool, int]] = geometry.slot_kinds

    def forward(self, images: Float32[Tensor, "b s 3 h w"], *geometry: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run the network on ``b`` framesets of one rig; ``geometry`` follows ``input_names`` order."""
        by_name: dict[str, Tensor] = dict(zip(self.input_names, geometry, strict=True))
        batch: int = images.shape[0]
        n_views: int = images.shape[1]
        n_patch: int = (images.shape[3] // self.patch_size) * (images.shape[4] // self.patch_size)
        n_full: int = 1 + n_patch + self.trailing
        ray_feat: Tensor | None = by_name["ray_feat"].expand(batch, *by_name["ray_feat"].shape[1:]) if "ray_feat" in by_name else None
        pos_embed: Tensor = by_name["pos_embed"][:, : 1 + n_patch]
        pos_local: Tensor = by_name["pos_local"][:, :, :n_full].expand(batch, n_views, n_full, 2)
        pos_global: Tensor = by_name["pos_global"][:, :, :n_full].expand(batch, n_views, n_full, 2)
        cam_types: Tensor = by_name["cam_types"].expand(batch, n_views)
        sliced: dict[str, Tensor] = {}
        for name, (is_global, extra) in self.slot_kinds.items():
            length: int = 1 + n_patch + extra
            if is_global:
                length = n_views * length
                sliced[name] = by_name[name][:, :, :length, :length]
            else:
                per_view: Tensor = by_name[name][:, :, :length, :length]
                sliced[name] = per_view.expand(batch, n_views, length, length).reshape(batch * n_views, 1, length, length)
        masks: tuple[AttentionBias | None, ...] = tuple(None if slot is None else sliced[slot] for slot in self.layer_slots)
        frozen = FrozenRigGeometry(
            cam_types=cam_types,
            ray_feat=ray_feat,
            calib_k=self.calib_k,
            pos_local=pos_local,
            pos_global=pos_global,
            attn_masks=masks,
            pos_embed=pos_embed,
        )
        output: XLensNetOutput = self.model(images, frozen=frozen)
        return tuple(output[name] for name in ENGINE_OUTPUT_NAMES)  # pyrefly: ignore  # bad-return — TypedDict keys are the output names


@dataclass(frozen=True, slots=True)
class DynamicRanges:
    """Symbol bounds of the dynamic profile."""

    views: tuple[int, int]
    """Smallest and largest view count."""
    patch_rows: tuple[int, int]
    """Smallest and largest patch-grid height (image height / 14)."""
    patch_cols: tuple[int, int]
    """Smallest and largest patch-grid width (image width / 14)."""
    batch: tuple[int, int]
    """Smallest and largest frameset batch."""


def dynamic_dims_spec(geometry: EngineGeometry, ranges: DynamicRanges) -> "DynamicDims":
    """``torch.export`` symbols for the dynamic profile: shared ``batch``/``views``/``patch_rows``/``patch_cols``, one token symbol per geometry input.

    Token counts are products of the patch-grid symbols, which ``torch.export``
    cannot express as derived dims; each geometry input therefore gets an
    independent ``Dim.AUTO`` symbol (the wrapper slices it to the implied count,
    and the relation becomes a deferred runtime assert). The TensorRT profile
    pins the consistent ranges.
    """
    from trtkit import DynamicDim

    batch = DynamicDim("batch", ranges.batch[0], ranges.batch[1])
    views = DynamicDim("views", ranges.views[0], ranges.views[1])
    rows = DynamicDim("patch_rows", ranges.patch_rows[0], ranges.patch_rows[1])
    cols = DynamicDim("patch_cols", ranges.patch_cols[0], ranges.patch_cols[1])
    patches: tuple[int, int] = (ranges.patch_rows[0] * ranges.patch_cols[0], ranges.patch_rows[1] * ranges.patch_cols[1])
    trailing: int = geometry.trailing
    spec: DynamicDims = {
        "images": {0: batch, 1: views, 3: DynamicDim("patch_rows", rows.min, rows.max, multiple=14), 4: DynamicDim("patch_cols", cols.min, cols.max, multiple=14)},
        "pos_embed": {1: DynamicDim("pos_embed_tokens", 1 + patches[0], 1 + patches[1], auto=True)},
        "pos_local": {1: views, 2: DynamicDim("rope_tokens", 1 + patches[0] + trailing, 1 + patches[1] + trailing, auto=True)},
        "pos_global": {1: views, 2: DynamicDim("rope_tokens", 1 + patches[0] + trailing, 1 + patches[1] + trailing, auto=True)},
        "cam_types": {1: views},
    }
    if "ray_feat" in geometry.inputs:
        spec["ray_feat"] = {1: views, 3: rows, 4: cols}
    for name, (is_global, extra) in geometry.slot_kinds.items():
        lengths: tuple[int, int] = (1 + patches[0] + extra, 1 + patches[1] + extra)
        if is_global:
            lengths = (ranges.views[0] * lengths[0], ranges.views[1] * lengths[1])
            spec[name] = {2: DynamicDim(f"{name}_tokens", *lengths, auto=True), 3: DynamicDim(f"{name}_tokens", *lengths, auto=True)}
        else:
            spec[name] = {1: views, 2: DynamicDim(f"{name}_tokens", *lengths, auto=True), 3: DynamicDim(f"{name}_tokens", *lengths, auto=True)}
    return spec


def shape_profiles(
    geometry: EngineGeometry,
    ranges: DynamicRanges,
    opt: tuple[int, int, int, int],
    embed_dim: int,
    heads: int,
) -> tuple["InputShapeProfile", ...]:
    """TensorRT min/opt/max shapes for every engine input, consistent with :func:`dynamic_dims_spec`.

    Args:
        geometry: Input names and per-slot token facts of the rig the graph was exported from.
        ranges: Symbol bounds.
        opt: Batch, views, patch rows, and patch columns TensorRT tunes for.
        embed_dim: Backbone feature width.
        heads: Backbone attention heads.

    Returns:
        One profile entry per input, in engine input order.
    """
    from trtkit import InputShapeProfile

    def bounds(batch: int, views: int, rows: int, cols: int) -> dict[str, tuple[int, ...]]:
        patches: int = rows * cols
        shapes: dict[str, tuple[int, ...]] = {
            "images": (batch, views, 3, 14 * rows, 14 * cols),
            "ray_feat": (1, views, embed_dim, rows, cols),
            "pos_embed": (1, 1 + patches, embed_dim),
            "pos_local": (1, views, 1 + patches + geometry.trailing, 2),
            "pos_global": (1, views, 1 + patches + geometry.trailing, 2),
            "cam_types": (1, views),
        }
        for name, (is_global, extra) in geometry.slot_kinds.items():
            length: int = 1 + patches + extra
            shapes[name] = (1, heads, views * length, views * length) if is_global else (1, views, length, length)
        return shapes

    low: dict[str, tuple[int, ...]] = bounds(ranges.batch[0], ranges.views[0], ranges.patch_rows[0], ranges.patch_cols[0])
    tuned: dict[str, tuple[int, ...]] = bounds(*opt)
    high: dict[str, tuple[int, ...]] = bounds(ranges.batch[1], ranges.views[1], ranges.patch_rows[1], ranges.patch_cols[1])
    return tuple(InputShapeProfile(name=name, min_shape=low[name], opt_shape=tuned[name], max_shape=high[name]) for name in ("images", *geometry.names))


@dataclass(frozen=True, slots=True)
class ExportPlan:
    """Everything one graph export and its engine build need, derived from a rig and a profile."""

    frozen: FrozenRigGeometry
    """Rig geometry the graph structure is derived from."""
    geometry: EngineGeometry
    """Engine inputs for this rig (traced as example values)."""
    signature: str
    """Cache-name fragment of the graph."""
    ranges: DynamicRanges
    """Symbol bounds (degenerate for the ``rig`` profile)."""
    opt: tuple[int, int, int, int]
    """Batch, views, patch rows, and patch columns TensorRT tunes for."""
    dynamic_dims: "DynamicDims | None"
    """Symbolic export dimensions, or None for a fully static graph."""
    image_hw: tuple[int, int]
    """Rig image height and width (the export example)."""


def plan_export(
    model: XLensNet,
    rays: Float32[ndarray, "s h w 3"],
    cam_types: Int64[ndarray, "s"],
    cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    *,
    profile: EngineProfile,
    opt_batch_size: int,
    dynamic_ranges: DynamicRanges,
) -> ExportPlan:
    """Freeze one rig and derive the graph signature, symbols, and tuning shape for a profile.

    Raises:
        ValueError: If a ``dynamic`` rig falls outside the configured ranges.
    """
    n_views: int = int(rays.shape[0])
    image_hw: tuple[int, int] = (int(rays.shape[1]), int(rays.shape[2]))
    rows: int = image_hw[0] // 14
    cols: int = image_hw[1] // 14
    if profile == "dynamic" and not (
        dynamic_ranges.views[0] <= n_views <= dynamic_ranges.views[1]
        and dynamic_ranges.patch_rows[0] <= rows <= dynamic_ranges.patch_rows[1]
        and dynamic_ranges.patch_cols[0] <= cols <= dynamic_ranges.patch_cols[1]
    ):
        raise ValueError(
            f"rig with {n_views} views at {image_hw[0]}x{image_hw[1]} is outside the dynamic profile (views {dynamic_ranges.views}, "
            f"height {14 * dynamic_ranges.patch_rows[0]}-{14 * dynamic_ranges.patch_rows[1]}, width {14 * dynamic_ranges.patch_cols[0]}-{14 * dynamic_ranges.patch_cols[1]})"
        )
    tensors: RigTensors = rig_tensors(rays, cam_types, cam_T_ref, torch.device("cuda"))
    with torch.inference_mode():
        frozen: FrozenRigGeometry = model.freeze_geometry(tensors.d_cam, tensors.cam_types, tensors.ray_map)
    geometry: EngineGeometry = engine_geometry(frozen, model.backbone.pretrained)
    if profile == "rig":
        ranges = DynamicRanges(views=(n_views, n_views), patch_rows=(rows, rows), patch_cols=(cols, cols), batch=dynamic_ranges.batch)
        signature: str = f"{rig_signature(cam_types, image_hw, cam_T_ref is not None)}_b{ranges.batch[1]}"
        opt: tuple[int, int, int, int] = (opt_batch_size, n_views, rows, cols)
    else:
        ranges = dynamic_ranges
        signature = dynamic_signature(ranges, frozen.calib_k, geometry)
        opt = (
            opt_batch_size,
            min(max(DYNAMIC_OPT_VIEWS, ranges.views[0]), ranges.views[1]),
            min(max(DYNAMIC_OPT_HW[0] // 14, ranges.patch_rows[0]), ranges.patch_rows[1]),
            min(max(DYNAMIC_OPT_HW[1] // 14, ranges.patch_cols[0]), ranges.patch_cols[1]),
        )
    # Degenerate ranges become static dims in trtkit, so a rig graph with batch 1 is fully static.
    spec: DynamicDims = dynamic_dims_spec(geometry, ranges)
    symbolic: bool = any(dim.min != dim.max for dims in spec.values() for dim in dims.values())
    dynamic_dims: DynamicDims | None = spec if symbolic else None
    return ExportPlan(
        frozen=frozen, geometry=geometry, signature=signature, ranges=ranges, opt=opt, dynamic_dims=dynamic_dims, image_hw=image_hw
    )


_EXPORT_WORKER_ENV: str = "MONOPRIOR_XLENS_ONNX_EXPORT_WORKER"


def rig_signature(cam_types: Int64[ndarray, "s"], image_hw: tuple[int, int], has_poses: bool) -> str:
    """Filename fragment naming everything a rig-profile graph bakes: views, resolution, camera types, pose presence."""
    cam_key: str = "".join(str(int(value)) for value in cam_types)
    return f"{len(cam_types)}v_{image_hw[0]}x{image_hw[1]}_ct{cam_key}_{'rays' if has_poses else 'norays'}"


def dynamic_signature(ranges: DynamicRanges, calib_k: int, geometry: EngineGeometry) -> str:
    """Filename fragment naming a dynamic-profile graph: symbol ranges, calibration tokens, mask layout, pose presence."""
    masks: str = "local" if any(not is_global for is_global, _ in geometry.slot_kinds.values()) else "nolocal"
    rays: str = "rays" if "ray_feat" in geometry.inputs else "norays"
    return (
        f"dyn_v{ranges.views[0]}-{ranges.views[1]}_h{14 * ranges.patch_rows[0]}-{14 * ranges.patch_rows[1]}"
        f"_w{14 * ranges.patch_cols[0]}-{14 * ranges.patch_cols[1]}_b{ranges.batch[0]}-{ranges.batch[1]}_k{calib_k}_{masks}_{rays}"
    )


def export_plan_onnx(model: XLensNet, plan: ExportPlan, onnx_path: Path) -> None:
    """Export one planned graph with trtkit's strongly-typed recipe, then drop superseded exports of the same signature.

    The traced example carries two framesets when the batch is symbolic.
    """
    from trtkit import export_onnx, sweep_stale_onnx_exports

    graph: _XLensRigGraph = _XLensRigGraph(model, plan.frozen, plan.geometry).eval()
    n_views: int = int(plan.geometry.inputs["pos_local"].shape[1])
    example_images: Float32[Tensor, "b s 3 h w"] = torch.zeros(
        (min(2, plan.ranges.batch[1]), n_views, 3, plan.image_hw[0], plan.image_hw[1]), dtype=torch.float32, device="cuda"
    )
    print(f"[monoprior] exporting X-Lens rig graph to ONNX (one-time, may take minutes): {onnx_path.name}")
    export_onnx(
        graph,
        (example_images, *(plan.geometry.inputs[name] for name in plan.geometry.names)),
        onnx_path,
        input_names=["images", *plan.geometry.names],
        output_names=list(ENGINE_OUTPUT_NAMES),
        compute_dtype=torch.float16,
        dynamic_dims=plan.dynamic_dims,
    )
    # Earlier ONNX_EXPORT_VERSIONs or checkpoints of this signature are superseded; keep this export and its sidecar.
    sweep_stale_onnx_exports(onnx_path.parent, f"xlens-vits-rig_{plan.signature}_", keep_paths={onnx_path, onnx_path.with_name(onnx_path.name + ".data")})


@dataclass
class XLensTrtConfig(BaseRigDepthPredictorConfig):
    """Configuration for the fp16 TensorRT X-Lens ViT-S engine."""

    checkpoint: Path | None = None
    """Local safetensors state dict, or None to download the pinned gated release."""
    use_cuda_graph: bool = True
    """Capture the engine launch in a CUDA graph (one per input-shape signature) and replay it."""
    cache_dir: Path = field(default_factory=lambda: XLENS_CACHE_DIR)
    """Cache root for ONNX exports and machine-local engines."""
    workspace_gib: float = 8.0
    """TensorRT builder workspace cap in GiB."""
    max_batch_size: int = 1
    """Largest frameset batch per engine call (``predict_batch`` chunks beyond it); activation memory scales with batch x views x tokens^2 at the max shape."""
    opt_batch_size: int = 1
    """Frameset batch TensorRT tunes kernels for."""
    profile: EngineProfile = "dynamic"
    """``dynamic``: one engine over the view/resolution ranges below; ``rig``: one engine per rig layout (a few ms faster, rebuilt per rig)."""
    dynamic_views: tuple[int, int] = (2, 4)
    """View-count range of the dynamic profile."""
    dynamic_height: tuple[int, int] = (280, 630)
    """Image-height range of the dynamic profile (multiples of 14)."""
    dynamic_width: tuple[int, int] = (336, 1120)
    """Image-width range of the dynamic profile (multiples of 14); the attention-bias input at the max shape is heads x tokens^2 fp16, ~2.5 GB for 4 views at 630x1120."""

    def setup(self, device: Literal["cpu", "cuda"]) -> "XLensTrtPredictor":
        """Build the TensorRT predictor; only CUDA is supported."""
        if device != "cuda":
            raise ValueError("the X-Lens TensorRT predictor requires device='cuda'")
        return XLensTrtPredictor(
            checkpoint=self.checkpoint,
            use_cuda_graph=self.use_cuda_graph,
            cache_dir=self.cache_dir,
            workspace_gib=self.workspace_gib,
            max_batch_size=self.max_batch_size,
            opt_batch_size=self.opt_batch_size,
            profile=self.profile,
            dynamic_views=self.dynamic_views,
            dynamic_height=self.dynamic_height,
            dynamic_width=self.dynamic_width,
        )


class XLensTrtPredictor(BaseRigDepthPredictor):
    """Released X-Lens weights as a cached TensorRT engine fed with persistent rig geometry."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        use_cuda_graph: bool = True,
        cache_dir: Path = XLENS_CACHE_DIR,
        workspace_gib: float = 8.0,
        max_batch_size: int = 1,
        opt_batch_size: int = 1,
        profile: EngineProfile = "dynamic",
        dynamic_views: tuple[int, int] = (2, 4),
        dynamic_height: tuple[int, int] = (280, 630),
        dynamic_width: tuple[int, int] = (336, 1120),
    ) -> None:
        """Load the eager model (it computes each rig's geometry and exports the graph); engines build lazily.

        Raises:
            RuntimeError: If CUDA is unavailable.
            ValueError: If batch or dynamic ranges are invalid.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("the X-Lens TensorRT predictor requires CUDA")
        if not 1 <= opt_batch_size <= max_batch_size:
            raise ValueError("batch sizes must satisfy 1 <= opt_batch_size <= max_batch_size")
        if dynamic_views[0] < 2 or dynamic_views[0] > dynamic_views[1]:
            raise ValueError(f"dynamic_views must be an increasing range starting at 2 or more, got {dynamic_views}")
        for name, (low, high) in (("dynamic_height", dynamic_height), ("dynamic_width", dynamic_width)):
            if low < 28 or low > high or low % 14 != 0 or high % 14 != 0:
                raise ValueError(f"{name} must be an increasing range of multiples of 14 at least 28, got {(low, high)}")
        loaded: tuple[XLensNet, Path] = load_xlens_model(checkpoint, "cuda")
        self.model: XLensNet = loaded[0]
        self.checkpoint_path: Path = loaded[1]
        with loaded[1].open("rb") as stream:
            self.checkpoint_digest: str = hashlib.file_digest(stream, "sha256").hexdigest()
        self.use_cuda_graph: bool = use_cuda_graph
        self.cache_dir: Path = cache_dir
        self.workspace_gib: float = workspace_gib
        self.max_batch_size: int = max_batch_size
        self.opt_batch_size: int = opt_batch_size
        self.profile: EngineProfile = profile
        self.dynamic_ranges: DynamicRanges = DynamicRanges(
            views=dynamic_views,
            patch_rows=(dynamic_height[0] // 14, dynamic_height[1] // 14),
            patch_cols=(dynamic_width[0] // 14, dynamic_width[1] // 14),
            batch=(1, max_batch_size),
        )
        self._memo: RigKeyMemo = RigKeyMemo()
        self._rig_key: RigKey | None = None
        self._geometry: EngineGeometry | None = None
        self._signature: str | None = None
        self._runtime: TensorRtDynamicRuntime | None = None
        self.engine_path: Path | None = None
        """Machine-local engine serving the current rig; None before the first call."""

    def runtime_summary(self) -> str:
        """Max input shapes, the attention-bias buffer size at the profile maximum, and the engine's device memory."""
        if self._runtime is None:
            return "no engine loaded"
        shapes: dict[str, tuple[int, ...]] = self._runtime.max_input_shapes
        bias_bytes: int = sum(2 * int(np.prod(shape)) for name, shape in shapes.items() if name.startswith("attn_bias"))
        return (
            f"max input shapes {shapes}; attention-bias buffers at max shape {bias_bytes / 1e9:.2f} GB fp16; "
            f"engine device memory {self._runtime.device_memory_bytes / 1e9:.2f} GB"
        )

    def onnx_path(self, signature: str) -> Path:
        """Portable ONNX path for one graph signature of this checkpoint."""
        return self.cache_dir / "onnx" / f"xlens-vits-rig_{signature}_v{ONNX_EXPORT_VERSION}_{self.checkpoint_digest[:8]}.onnx"

    def _export(self, plan: ExportPlan, onnx_path: Path, rays: Float32[ndarray, "s h w 3"], cam_types: Int64[ndarray, "s"], cam_T_ref: Float64[ndarray, "s 4 4"] | None) -> None:
        """Export in-process, or in a clean interpreter when beartype's dev claw is active.

        Symbolic dims reach the fork as ``torch.SymInt`` values, which the dev-mode
        ``int`` checks reject (the same reason MoGe re-enters its export). The worker
        rebuilds the identical plan from the rig arrays and the settings file.
        """
        if os.environ.get("PIXI_DEV_MODE") != "1" or os.environ.get(_EXPORT_WORKER_ENV) == "1" or plan.dynamic_dims is None:
            export_plan_onnx(self.model, plan, onnx_path)
            return
        with tempfile.TemporaryDirectory(prefix="xlens-onnx-worker-") as directory:
            rig_path: Path = Path(directory) / "rig.npz"
            # An empty pose stack stands for "no poses" (keyword unpacking confuses the numpy stubs).
            np.savez(rig_path, rays=rays, cam_types=cam_types, cam_T_ref=np.zeros((0, 4, 4)) if cam_T_ref is None else cam_T_ref)
            settings_path: Path = Path(directory) / "plan.json"
            settings_path.write_text(json.dumps({"profile": self.profile, "opt_batch_size": self.opt_batch_size, "ranges": asdict(self.dynamic_ranges)}))
            worker_env: dict[str, str] = dict(os.environ)
            worker_env["PIXI_DEV_MODE"] = "0"
            worker_env[_EXPORT_WORKER_ENV] = "1"
            command: list[str] = [
                sys.executable,
                "-m",
                "monopriors.models.rig_depth._xlens_onnx_worker",
                "--rig-path",
                str(rig_path),
                "--settings-path",
                str(settings_path),
                "--onnx-path",
                str(onnx_path),
                "--checkpoint",
                str(self.checkpoint_path),
            ]
            subprocess.run(command, check=True, env=worker_env)
        if not onnx_path.exists():
            raise RuntimeError(f"X-Lens ONNX export worker did not produce {onnx_path}")

    def _prepare(
        self,
        rays: Float32[ndarray, "s h w 3"],
        cam_types: Int64[ndarray, "s"],
        cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    ) -> tuple[EngineGeometry, TensorRtDynamicRuntime]:
        """Return the engine inputs and runtime for one rig, freezing geometry and building the engine when they change."""
        key: RigKey = self._memo.key(rays, cam_types, cam_T_ref)
        if self._geometry is not None and self._runtime is not None and key == self._rig_key:
            return self._geometry, self._runtime
        plan: ExportPlan = plan_export(
            self.model,
            rays,
            cam_types,
            cam_T_ref,
            profile=self.profile,
            opt_batch_size=self.opt_batch_size,
            dynamic_ranges=self.dynamic_ranges,
        )
        geometry: EngineGeometry = plan.geometry
        if self._runtime is None or plan.signature != self._signature:
            from trtkit import TensorRtDynamicRuntime as Runtime
            from trtkit import TrtBuildConfig, ensure_engine

            self._runtime = None
            self._geometry = None  # the previous rig's inputs go before the new engine's buffers are allocated
            onnx_path: Path = self.onnx_path(plan.signature)
            if not onnx_path.exists():
                self._export(plan, onnx_path, rays, cam_types, cam_T_ref)
            torch.cuda.empty_cache()
            backbone: DinoVisionTransformer = self.model.backbone.pretrained
            profiles: tuple[InputShapeProfile, ...] = shape_profiles(geometry, plan.ranges, plan.opt, backbone.embed_dim, backbone.num_heads)
            build_config: TrtBuildConfig = TrtBuildConfig(
                max_batch_size=plan.ranges.batch[1], opt_batch_size=plan.opt[0], workspace_gib=self.workspace_gib, shape_profiles=profiles
            )
            engine_path: Path = ensure_engine(onnx_path, build_config, cache_dir=self.cache_dir / "trt")
            self._runtime = Runtime(engine_path, use_cuda_graph=self.use_cuda_graph)
            self.engine_path = engine_path
            self._signature = plan.signature
        self._geometry = geometry
        self._rig_key = key
        return geometry, self._runtime

    def predict_batch(
        self,
        images: UInt8[ndarray, "b s h w 3"],
        rays: Float32[ndarray, "s h w 3"],
        cam_types: Int64[ndarray, "s"],
        cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    ) -> list[RigDepthPrediction]:
        """Predict several framesets of one rig, chunked to the engine's batch profile.

        ``trtkit.base.run_chunked`` slices every input along dim 0, but the
        geometry inputs are batch-1 and shared, so only the images are chunked here.

        Args:
            images: RGB framesets, ``UInt8[ndarray, "b s h w 3"]``.
            rays: Camera-frame unit rays shared by all framesets, ``Float32[ndarray, "s h w 3"]``.
            cam_types: X-Lens camera ids, ``Int64[ndarray, "s"]``.
            cam_T_ref: Optional camera-to-reference poses, ``Float64[ndarray, "s 4 4"]``.

        Returns:
            One owning prediction per frameset, in order.
        """
        validate_rig_inputs(images[0], rays, cam_types)
        geometry, runtime = self._prepare(rays, cam_types, cam_T_ref)
        predictions: list[RigDepthPrediction] = []
        chunk: int = self.max_batch_size
        for start in range(0, images.shape[0], chunk):
            image_tensor: Float32[Tensor, "c s 3 h w"] = normalize_framesets(images[start : start + chunk], torch.device("cuda"))
            outputs: XLensNetOutput = cast(XLensNetOutput, runtime({"images": image_tensor, **geometry.inputs}))
            predictions.extend(rig_depth_prediction(outputs, index) for index in range(image_tensor.shape[0]))
        return predictions

    def __call__(
        self,
        images: UInt8[ndarray, "s h w 3"],
        rays: Float32[ndarray, "s h w 3"],
        cam_types: Int64[ndarray, "s"],
        cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    ) -> RigDepthPrediction:
        """Predict per-view camera-frame metric z-depth for one frameset with the cached engine.

        Args:
            images: Shared-resolution RGB views, ``UInt8[ndarray, "s h w 3"]``.
            rays: Camera-frame unit rays, ``Float32[ndarray, "s h w 3"]``; must not be mutated in place between calls.
            cam_types: X-Lens camera ids, ``Int64[ndarray, "s"]``.
            cam_T_ref: Optional camera-to-reference poses, ``Float64[ndarray, "s 4 4"]``.

        Returns:
            Owning metric depth, confidence, mask, and scale on CUDA.
        """
        return self.predict_batch(images[None], rays, cam_types, cam_T_ref)[0]
