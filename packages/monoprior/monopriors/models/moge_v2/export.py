"""Shared MoGe v2 checkpoint, preprocessing, export, and aspect plumbing."""

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Float32, UInt8
from torch import Tensor, nn

from monopriors.third_party.dinov2.vision_transformer import DinoVisionTransformer
from monopriors.third_party.moge.model.v2 import ForwardOutput, MoGeModel

Encoder: TypeAlias = Literal["vits", "vitb", "vitl"]
HardwareCompatibility: TypeAlias = Literal["none", "same_compute_capability"]
HeadSet: TypeAlias = Literal["normal", "geometry"]

DEFAULT_IMAGE_HW: tuple[int, int] = (756, 1008)
"""ARKitScenes-aligned MoGe network height and width."""

MOGE_V2_CHECKPOINTS: dict[Encoder, tuple[str, str]] = {
    "vitl": ("Ruicheng/moge-2-vitl-normal", "b135031bae30b5ac2ae141a0e68717795ce38340"),
    "vitb": ("Ruicheng/moge-2-vitb-normal", "54ad3a693e61907ea4633d13dec6ee682fa09419"),
    "vits": ("Ruicheng/moge-2-vits-normal", "679230677b4d282c6f304189a93e98e14f085902"),
}
"""Pinned normal-capable checkpoint repository and revision for each encoder."""

DEFAULT_CACHE_DIR: Path = Path(os.environ.get("MONOPRIOR_TRT_CACHE", "~/.cache/monoprior")).expanduser()
"""Cache root holding portable ``onnx/`` and machine-local ``trt/`` artifacts."""

MOGE_V2_NUM_TOKENS_RANGE: tuple[int, int] = (1200, 3600)
"""Training-time token range shared by all pinned MoGe v2 checkpoints."""

_EXPORT_WORKER_ENV: str = "MONOPRIOR_MOGE_V2_ONNX_EXPORT_WORKER"

FORWARD_HEADS: dict[HeadSet, tuple[str, ...]] = {
    "normal": ("normal", "mask"),
    "geometry": ("points", "normal", "mask", "scale"),
}
ONNX_OUTPUT_NAMES: dict[HeadSet, tuple[str, ...]] = {
    "normal": ("normal", "mask"),
    "geometry": ("points", "normal", "mask", "metric_scale"),
}
"""Stable ONNX output names in adapter tuple order for each head set."""

_ONNX_EXPORT_VERSIONS: dict[HeadSet, int] = {"normal": 1, "geometry": 1}


def resolve_num_tokens(resolution_level: int) -> int:
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


def token_grid_hw(aspect_ratio: float, num_tokens: int) -> tuple[int, int]:
    """Compute MoGe's patch-aligned network size for one image aspect.

    Args:
        aspect_ratio: Image width divided by height.
        num_tokens: Approximate DINOv2 patch-token budget.

    Returns:
        Network height and width, each aligned to the 14-pixel patch size.

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


def aspect_buckets(num_tokens: int) -> tuple[tuple[int, int], ...]:
    """Build the five pinned-aspect MoGe network buckets for a token budget.

    Args:
        num_tokens: Approximate DINOv2 patch-token budget.

    Returns:
        Network sizes for 4:3, 3:4, 16:9, 9:16, and 1:1 inputs.
    """
    return tuple(token_grid_hw(aspect_ratio, num_tokens) for aspect_ratio in (4.0 / 3.0, 3.0 / 4.0, 16.0 / 9.0, 9.0 / 16.0, 1.0))


ASPECT_BUCKETS: tuple[tuple[int, int], ...] = aspect_buckets(3600)
"""Five network buckets at 3,600 tokens, equal to ``resolve_num_tokens(9)``."""

DYNAMIC_MAX_ASPECT: float = 2.0
"""The dynamic engine serves token grids of aspects (width / height) from 1/2 through 2: 9:16 portrait to 16:9 and wider."""

DYNAMIC_OPT_ASPECT: float = 4.0 / 3.0
"""Aspect whose token grid the dynamic graph is traced at and its TensorRT profile tunes kernels for (inside the range on both axes)."""

DYNAMIC_ONNX_MAX_BATCH: int = 64
"""Batch bound of the dynamic ONNX graph; the TensorRT profile sets the batch a given engine accepts."""


def dynamic_patch_range(num_tokens: int) -> tuple[int, int]:
    """Per-axis token-grid range, in patches, of the dynamic engine at one token budget.

    Args:
        num_tokens: Approximate DINOv2 patch-token budget.

    Returns:
        Smallest and largest grid side over the aspects ``1 / DYNAMIC_MAX_ASPECT`` through ``DYNAMIC_MAX_ASPECT``
        (the grid sides are monotonic in the aspect, so the 2:1 grid holds both ends).
    """
    rows_cols: tuple[int, int] = token_grid_hw(DYNAMIC_MAX_ASPECT, num_tokens)
    return min(rows_cols) // 14, max(rows_cols) // 14


def select_aspect_bucket(
    input_hw: tuple[int, int],
    buckets: tuple[tuple[int, int], ...] = ASPECT_BUCKETS,
) -> tuple[int, int]:
    """Select the network bucket nearest to an input aspect in log-space.

    Args:
        input_hw: Caller input height and width.
        buckets: Candidate network height and width pairs.

    Returns:
        The nearest candidate bucket. Ties retain candidate order.

    Raises:
        ValueError: If the input dimensions are invalid.
    """
    input_height: int = input_hw[0]
    input_width: int = input_hw[1]
    if input_height < 1 or input_width < 1:
        raise ValueError(f"Input dimensions must be positive, got {input_hw}.")

    input_log_aspect: float = math.log(input_width / input_height)
    selected_bucket: tuple[int, int] = buckets[0]
    selected_distance: float = math.inf
    for bucket_hw in buckets:
        distance: float = abs(math.log(bucket_hw[1] / bucket_hw[0]) - input_log_aspect)
        if distance < selected_distance:
            selected_bucket = bucket_hw
            selected_distance = distance
    return selected_bucket


def preprocess_rgb(
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
        raise RuntimeError("MoGe v2 batched predictors require CUDA input tensors.")
    if rgb_bhw3.shape[0] < 1:
        raise ValueError("MoGe v2 batched predictors require a non-empty batch.")
    if image_hw[0] < 1 or image_hw[1] < 1:
        raise ValueError(f"MoGe v2 network dimensions must be positive, got {image_hw}.")

    image_b3hw: Float32[Tensor, "b 3 h w"] = rearrange(rgb_bhw3, "b h w c -> b c h w").to(dtype=torch.float32) / 255.0
    if tuple(image_b3hw.shape[-2:]) != image_hw:
        image_b3hw = F.interpolate(image_b3hw, size=image_hw, mode="bilinear", antialias=True)
    return image_b3hw


def load_pinned_moge_v2(encoder: Encoder, resolution_level: int) -> tuple[MoGeModel, int]:
    """Load one pinned checkpoint and enable its ONNX-compatible forward path.

    Args:
        encoder: DINOv2 encoder size.
        resolution_level: Detail level from 0 through 9.

    Returns:
        CUDA evaluation model and resolved token count.

    Raises:
        ValueError: If the resolution level or pinned token metadata is invalid.
    """
    num_tokens: int = resolve_num_tokens(resolution_level)
    checkpoint: tuple[str, str] = MOGE_V2_CHECKPOINTS[encoder]
    model: MoGeModel = MoGeModel.from_pretrained(checkpoint[0], revision=checkpoint[1]).to("cuda").eval()
    if model.num_tokens_range != MOGE_V2_NUM_TOKENS_RANGE:
        raise ValueError(
            f"Pinned MoGe v2 checkpoint token range changed from {MOGE_V2_NUM_TOKENS_RANGE} to {model.num_tokens_range}; update the export recipe and predictors."
        )
    model.onnx_compatible_mode = True
    return model, num_tokens


def _bicubic_weights(out_size: int, in_size: int, offset: float, device: torch.device) -> Float32[Tensor, "o i"]:
    """Row-stochastic matrix of PyTorch's bicubic resize along one axis at ``scale_factor = (out_size + offset) / in_size``.

    Same source coordinates (``align_corners=False``, no clamp before the taps), cubic coefficients (``A = -0.75``) and
    edge clamping as ``F.interpolate(mode="bicubic")``; built from ``arange`` so ``out_size`` may be symbolic.
    """
    a: float = -0.75
    out_t: Float32[Tensor, ""] = torch.ones((), dtype=torch.float32, device=device) * out_size
    source_o: Float32[Tensor, "o"] = (torch.arange(out_size, dtype=torch.float32, device=device) + 0.5) * (in_size / (out_t + offset)) - 0.5
    floor_o: Float32[Tensor, "o"] = torch.floor(source_o)
    t: Float32[Tensor, "o"] = source_o - floor_o
    w0: Float32[Tensor, "o"] = ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a
    w1: Float32[Tensor, "o"] = ((a + 2) * t - (a + 3)) * t * t + 1
    w2: Float32[Tensor, "o"] = ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1
    weights_o4: Float32[Tensor, "o 4"] = torch.stack([w0, w1, w2, 1 - w0 - w1 - w2], dim=-1)
    taps_o4: Float32[Tensor, "o 4"] = (floor_o[:, None] + torch.arange(-1, 3, dtype=torch.float32, device=device)).clamp(0, in_size - 1)
    columns_i: Float32[Tensor, "i"] = torch.arange(in_size, dtype=torch.float32, device=device)
    return (weights_o4[..., None] * (taps_o4[..., None] == columns_i).float()).sum(dim=1)


def dynamic_pos_encoding(pos_embed_1nc: Float[Tensor, "1 n c"], grid_hw: tuple[int, int], offset: float) -> Float32[Tensor, "1 m c"]:
    """DINOv2's positional table resized to a token grid, as ``interpolate_pos_encoding`` does, traceable at a symbolic grid.

    DINOv2 resizes with ``scale_factor = (grid + offset) / 37``; ``torch.export`` freezes a float scale factor at the
    traced grid, so the dynamic graph applies the same bicubic resize as two weight matrices instead.

    Args:
        pos_embed_1nc: Learned table: class token, then a square grid of patch tokens.
        grid_hw: Token rows and columns (may be symbolic).
        offset: DINOv2's ``interpolate_offset``.

    Returns:
        Float32 class token then ``rows * cols`` patch tokens.
    """
    side: int = math.isqrt(pos_embed_1nc.shape[1] - 1)
    table_ssc: Float32[Tensor, "s s c"] = pos_embed_1nc[0, 1:].float().unflatten(0, (side, side))
    rows_hs: Float32[Tensor, "h s"] = _bicubic_weights(grid_hw[0], side, offset, pos_embed_1nc.device)
    cols_ws: Float32[Tensor, "w s"] = _bicubic_weights(grid_hw[1], side, offset, pos_embed_1nc.device)
    resized_hwc: Float32[Tensor, "h w c"] = torch.einsum("hi,ijc,wj->hwc", rows_hs, table_ssc, cols_ws)
    return torch.cat([pos_embed_1nc[:, :1].float(), resized_hwc.flatten(0, 1)[None]], dim=1)


class _MoGeV2Heads(nn.Module):
    """Fixed-token adapter exposing one ordered ONNX head set.

    With ``token_grid_from_shape`` it also replaces the given model's DINOv2 positional resize with
    :func:`dynamic_pos_encoding`, so pass it a model loaded for the export only.
    """

    def __init__(self, model: MoGeModel, num_tokens: int, heads: HeadSet, *, token_grid_from_shape: bool = False) -> None:
        super().__init__()
        self.model: MoGeModel = model
        self.num_tokens: int = num_tokens
        self.heads: HeadSet = heads
        self.token_grid_from_shape: bool = token_grid_from_shape
        if token_grid_from_shape:
            backbone: DinoVisionTransformer = model.encoder.backbone
            pos_embed: Tensor = backbone.pos_embed
            offset: float = backbone.interpolate_offset
            patch: int = backbone.patch_size

            # Captures the table, not the backbone: a backbone -> function -> backbone cycle would keep the model on
            # the GPU through the engine build until the cyclic GC ran.
            def pos_encoding(x_bnc: Tensor, height: int, width: int) -> Tensor:
                return dynamic_pos_encoding(pos_embed, (height // patch, width // patch), offset).to(x_bnc.dtype)

            backbone.interpolate_pos_encoding = pos_encoding  # pyrefly: ignore[bad-assignment]

    def forward(self, image_b3hw: Float[Tensor, "b 3 h w"]) -> tuple[Float[Tensor, "*shape"], ...]:
        """Run the selected forward heads in stable ONNX output order.

        Args:
            image_b3hw: Float RGB batch shaped ``b 3 h w`` in ``[0, 1]``.

        Returns:
            Requested head tensors ordered by ``ONNX_OUTPUT_NAMES``.
        """
        output: ForwardOutput = self.model(
            image_b3hw,
            num_tokens=self.num_tokens,
            output_heads=FORWARD_HEADS[self.heads],
            token_grid_from_shape=self.token_grid_from_shape,
        )
        return tuple(output[name] for name in ONNX_OUTPUT_NAMES[self.heads])


def _onnx_signature(heads: HeadSet, encoder: Encoder, image_hw: tuple[int, int] | None, num_tokens: int) -> str:
    """File-name prefix shared by every export version and checkpoint of one graph (the stale-export sweep key)."""
    if image_hw is None:
        low, high = dynamic_patch_range(num_tokens)
        size: str = f"dyn{14 * low}-{14 * high}"
    else:
        size = f"{image_hw[0]}x{image_hw[1]}"
    return f"moge-v2-{encoder}-{heads}_{size}_t{num_tokens}_"


def moge_v2_onnx_path(
    heads: HeadSet, encoder: Encoder, image_hw: tuple[int, int] | None, resolution_level: int = 9, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Path:
    """Where :func:`export_moge_v2_onnx` caches the ONNX file of one head set, encoder, network size (None: the dynamic
    graph) and detail level."""
    signature: str = _onnx_signature(heads, encoder, image_hw, resolve_num_tokens(resolution_level))
    return cache_dir / "onnx" / f"{signature}v{_ONNX_EXPORT_VERSIONS[heads]}_{MOGE_V2_CHECKPOINTS[encoder][1][:8]}.onnx"


def export_moge_v2_onnx(
    heads: HeadSet,
    encoder: Encoder,
    image_hw: tuple[int, int] | None,
    resolution_level: int = 9,
    max_batch_size: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Export one pinned MoGe v2 head set with a dynamic batch, and with a dynamic token grid when ``image_hw`` is None.

    Args:
        heads: Normal-only or full geometry output set.
        encoder: DINOv2 encoder size.
        image_hw: Static network height and width, or None for the dynamic graph: height and width are any multiples of
            14 within :func:`dynamic_patch_range`, and the graph takes its token grid from them, so it expects images
            already resized to their token grid (``token_grid_hw``).
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
        raise RuntimeError("MoGe v2 ONNX export requires CUDA.")
    if image_hw is not None and (image_hw[0] < 1 or image_hw[1] < 1):
        raise ValueError(f"MoGe v2 network dimensions must be positive, got {image_hw}.")
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")

    num_tokens: int = resolve_num_tokens(resolution_level)
    onnx_path: Path = moge_v2_onnx_path(heads, encoder, image_hw, resolution_level, cache_dir)
    if onnx_path.exists():
        return onnx_path

    # Beartype's import hook (pytest only, PIXI_DEV_MODE=1) checks annotated locals inside the vendored
    # forward. Dynamic ONNX export represents the batch as torch.SymInt, which
    # correctly violates those eager-only ``int`` checks. Re-enter this exact
    # export in a clean interpreter without beartype; the parent test
    # process keeps beartype active for every predictor call.
    if os.environ.get("PIXI_DEV_MODE") == "1" and os.environ.get(_EXPORT_WORKER_ENV) != "1":
        worker_env: dict[str, str] = dict(os.environ)
        worker_env["PIXI_DEV_MODE"] = "0"
        worker_env[_EXPORT_WORKER_ENV] = "1"
        size_args: list[str] = [] if image_hw is None else ["--height", str(image_hw[0]), "--width", str(image_hw[1])]
        worker_command: list[str] = [
            sys.executable,
            "-m",
            "monopriors.models.moge_v2._onnx_worker",
            "--heads",
            heads,
            "--encoder",
            encoder,
            *size_args,
            "--resolution-level",
            str(resolution_level),
            "--max-batch-size",
            str(max_batch_size),
            "--cache-dir",
            str(cache_dir),
        ]
        subprocess.run(worker_command, check=True, env=worker_env)
        if not onnx_path.exists():
            raise RuntimeError(f"MoGe v2 ONNX export worker did not produce {onnx_path}.")
        return onnx_path

    print(f"[monoprior] exporting MoGe v2 {heads} heads to ONNX (one-time, may take minutes): {onnx_path.name}")
    loaded: tuple[MoGeModel, int] = load_pinned_moge_v2(encoder, resolution_level)
    model: MoGeModel = loaded[0]
    wrapper: _MoGeV2Heads = _MoGeV2Heads(model, loaded[1], heads, token_grid_from_shape=image_hw is None).eval()
    # The dynamic graph traces at the 4:3 grid, inside the range on both axes.
    example_hw: tuple[int, int] = token_grid_hw(DYNAMIC_OPT_ASPECT, num_tokens) if image_hw is None else image_hw
    example_batch_size: int = min(2, max_batch_size)
    dummy_image_b3hw: Float32[Tensor, "b 3 h w"] = torch.zeros(
        (example_batch_size, 3, example_hw[0], example_hw[1]),
        dtype=torch.float32,
        device="cuda",
    )

    from trtkit import DynamicDim, export_onnx, sweep_stale_onnx_exports

    low, high = dynamic_patch_range(num_tokens)
    # Static graphs keep the batch-only path (dynamic_batch_max), so their exports are unchanged.
    dynamic_dims: dict[str, dict[int, DynamicDim]] | None = (
        {
            "image": {
                0: DynamicDim("batch", 1, max_batch_size),
                2: DynamicDim("patch_rows", low, high, multiple=14),
                3: DynamicDim("patch_cols", low, high, multiple=14),
            }
        }
        if image_hw is None
        else None
    )
    export_onnx(
        wrapper,
        (dummy_image_b3hw,),
        onnx_path,
        input_names=["image"],
        output_names=list(ONNX_OUTPUT_NAMES[heads]),
        compute_dtype=torch.float16,
        dynamic_batch_max=None if image_hw is None else max_batch_size,
        dynamic_dims=dynamic_dims,
    )
    del dummy_image_b3hw, wrapper, model
    torch.cuda.empty_cache()

    current_sidecar: Path = onnx_path.with_name(f"{onnx_path.name}.data")
    sweep_stale_onnx_exports(onnx_path.parent, _onnx_signature(heads, encoder, image_hw, num_tokens), keep_paths={onnx_path, current_sidecar})
    return onnx_path
