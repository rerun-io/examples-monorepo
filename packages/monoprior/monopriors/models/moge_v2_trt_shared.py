"""Shared MoGe v2 TensorRT preprocessing, export, and aspect plumbing."""

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

from monopriors.models.surface_normal.moge_v2 import MOGE_V2_NORMAL_CHECKPOINTS
from monopriors.third_party.moge.model.v2 import ForwardOutput, MoGeModel

Encoder: TypeAlias = Literal["vits", "vitb", "vitl"]
HardwareCompatibility: TypeAlias = Literal["none", "same_compute_capability"]
HeadSet: TypeAlias = Literal["normal", "geometry"]

DEFAULT_CACHE_DIR: Path = Path(os.environ.get("MONOPRIOR_TRT_CACHE", "~/.cache/monoprior")).expanduser()
"""Cache root holding portable ``onnx/`` and machine-local ``trt/`` artifacts."""

MOGE_V2_NUM_TOKENS_RANGE: tuple[int, int] = (1200, 3600)
"""Training-time token range shared by all pinned MoGe v2 checkpoints."""

_EXPORT_WORKER_ENV: str = "MONOPRIOR_MOGE_V2_ONNX_EXPORT_WORKER"

_FORWARD_HEADS: dict[HeadSet, tuple[str, ...]] = {
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

    image_b3hw: Float32[Tensor, "b 3 h w"] = rearrange(rgb_bhw3, "b h w c -> b c h w").to(dtype=torch.float32) / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive
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
    checkpoint: tuple[str, str] = MOGE_V2_NORMAL_CHECKPOINTS[encoder]
    model: MoGeModel = MoGeModel.from_pretrained(checkpoint[0], revision=checkpoint[1]).to("cuda").eval()
    if model.num_tokens_range != MOGE_V2_NUM_TOKENS_RANGE:
        raise ValueError(
            f"Pinned MoGe v2 checkpoint token range changed from {MOGE_V2_NUM_TOKENS_RANGE} to {model.num_tokens_range}; update the export recipe and predictors."
        )
    model.onnx_compatible_mode = True
    return model, num_tokens


class _MoGeV2Heads(nn.Module):
    """Fixed-token adapter exposing one ordered ONNX head set."""

    def __init__(self, model: MoGeModel, num_tokens: int, heads: HeadSet) -> None:
        super().__init__()
        self.model: MoGeModel = model
        self.num_tokens: int = num_tokens
        self.heads: HeadSet = heads

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
            output_heads=_FORWARD_HEADS[self.heads],
        )
        return tuple(output[name] for name in ONNX_OUTPUT_NAMES[self.heads])


def export_moge_v2_onnx(
    heads: HeadSet,
    encoder: Encoder,
    image_hw: tuple[int, int],
    resolution_level: int = 9,
    max_batch_size: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Export one pinned MoGe v2 head set with dynamic batch only.

    Args:
        heads: Normal-only or full geometry output set.
        encoder: DINOv2 encoder size.
        image_hw: Required static network height and width.
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
    export_version: int = _ONNX_EXPORT_VERSIONS[heads]
    onnx_path: Path = onnx_dir / f"moge-v2-{encoder}-{heads}_{height}x{width}_t{num_tokens}_v{export_version}_{checkpoint_revision}.onnx"
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
            "monopriors.models._moge_v2_onnx_worker",
            "--heads",
            heads,
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
            raise RuntimeError(f"MoGe v2 ONNX export worker did not produce {onnx_path}.")
        return onnx_path

    print(f"[monoprior] exporting MoGe v2 {heads} heads to ONNX (one-time, may take minutes): {onnx_path.name}")
    loaded: tuple[MoGeModel, int] = load_pinned_moge_v2(encoder, resolution_level)
    model: MoGeModel = loaded[0]
    wrapper: _MoGeV2Heads = _MoGeV2Heads(model, loaded[1], heads).eval()
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
        output_names=list(ONNX_OUTPUT_NAMES[heads]),
        compute_dtype=torch.float16,
        dynamic_batch_max=max_batch_size,
    )
    del dummy_image_b3hw, wrapper, model
    torch.cuda.empty_cache()

    current_sidecar: Path = onnx_path.with_name(f"{onnx_path.name}.data")
    stale_prefix: str = f"moge-v2-{encoder}-{heads}_{height}x{width}_t{num_tokens}_"
    sweep_stale_onnx_exports(onnx_dir, stale_prefix, keep_paths={onnx_path, current_sidecar})
    return onnx_path
