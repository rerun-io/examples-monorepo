"""Static-batch ONNX export for ZipDepth-PromptDA."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from monopriors.models.depth_completion.base_completion_depth import DEPTH_OUTPUT_NAME, IMAGE_INPUT_NAME, PROMPT_INPUT_NAME
from monopriors.models.depth_completion.zipdepth_prompt import load_zipdepth_prompt
from monopriors.models.relative_depth.zipdepth import download_zipdepth_checkpoint

ONNX_EXPORT_VERSION: int = 1
DEFAULT_CACHE_DIR: Path = Path(os.environ.get("ZIPDEPTH_PROMPT_TRT_CACHE", "~/.cache/zipdepth-promptda")).expanduser()
"""Portable ZipDepth-PromptDA ONNX cache root."""

ModelLoader = Callable[[Path], Any]
ExportFn = Callable[..., None]


class _ExportOutputs(nn.Module):
    """Expose prompt min/max reductions as TensorRT fusion barriers."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: nn.Module = model

    def forward(self, image: Tensor, prompt_depth: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return depth and the two prompt-range fusion barriers."""
        prompt_model: Any = self.model
        return cast(tuple[Tensor, Tensor, Tensor], prompt_model.forward_with_range(image, prompt_depth))


def _checkpoint_tag(checkpoint: Path) -> str:
    """Return a stable short identity for released and local weights."""
    parts: tuple[str, ...] = checkpoint.parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1][:8]
    digest = hashlib.sha256()
    with checkpoint.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()[:8]


def prepare_zipdepth_prompt_for_export(
    model: Any,
    image_hw: tuple[int, int],
    device: torch.device,
) -> nn.Module:
    """Fuse and convert a prompted model for a static fp16 graph.

    Args:
        model: ZipDepth-PromptDA module exposing ``fuse_for_inference``.
        image_hw: Static export image height and width.
        device: Export device.

    Returns:
        Fused fp16 evaluation module on ``device``.
    """
    height, width = image_hw
    if height <= 0 or width <= 0 or height % 32 != 0 or width % 32 != 0:
        raise ValueError(f"ZipDepth image dimensions must be positive multiples of 32, got {image_hw}")
    fused_model = cast(nn.Module, model.fuse_for_inference())
    return fused_model.to(device).half()


def export_zipdepth_prompt_onnx(
    *,
    checkpoint: Path | None = None,
    image_hw: tuple[int, int] = (768, 1024),
    batch_size: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_loader: ModelLoader = load_zipdepth_prompt,
    export_fn: ExportFn | None = None,
    device: str = "cuda",
) -> Path:
    """Export a fixed-batch, two-input ONNX graph with fp32 I/O.

    Args:
        checkpoint: Prompted checkpoint; ``None`` downloads released base weights.
        image_hw: Static image height and width.
        batch_size: Batch baked into the graph.
        cache_dir: Portable ONNX cache root.
        model_loader: Model loader boundary used by export tests.
        export_fn: ONNX writer boundary; defaults to :func:`trtkit.export_onnx`.
        device: Torch export device.

    Returns:
        Existing or newly exported ONNX graph path.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    height, width = image_hw
    resolved_checkpoint: Path = checkpoint or download_zipdepth_checkpoint()
    tag: str = _checkpoint_tag(resolved_checkpoint)
    onnx_dir: Path = cache_dir / "onnx"
    onnx_path: Path = onnx_dir / f"zipdepth-prompt_{height}x{width}_b{batch_size}_fp16_v{ONNX_EXPORT_VERSION}_{tag}.onnx"
    if onnx_path.is_file():
        return onnx_path

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fp16 ZipDepth ONNX export")
    model: nn.Module = prepare_zipdepth_prompt_for_export(model_loader(resolved_checkpoint), image_hw, resolved_device)
    wrapper: nn.Module = _ExportOutputs(model).eval()
    dummy_image_b3hw: Tensor = torch.rand((batch_size, 3, height, width), dtype=torch.float32, device=resolved_device)
    dummy_prompt_b1hw: Tensor = torch.linspace(0.2, 3.8, 192 * 256, dtype=torch.float32, device=resolved_device).reshape(1, 1, 192, 256)
    dummy_prompt_b1hw = dummy_prompt_b1hw.expand(batch_size, -1, -1, -1).contiguous()
    dummy_prompt_b1hw[:, :, 24:160:3, 32:224:4] = 0.0

    if export_fn is None:
        from trtkit import export_onnx

        export_fn = export_onnx
    export_fn(
        wrapper,
        (dummy_image_b3hw, dummy_prompt_b1hw),
        onnx_path,
        input_names=[IMAGE_INPUT_NAME, PROMPT_INPUT_NAME],
        output_names=[DEPTH_OUTPUT_NAME, "min_depth", "max_depth"],
        compute_dtype=None,
        dynamic_batch_max=None,
    )
    del wrapper, model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return onnx_path


__all__ = (
    "DEFAULT_CACHE_DIR",
    "DEPTH_OUTPUT_NAME",
    "IMAGE_INPUT_NAME",
    "ONNX_EXPORT_VERSION",
    "PROMPT_INPUT_NAME",
    "_ExportOutputs",
    "export_zipdepth_prompt_onnx",
    "prepare_zipdepth_prompt_for_export",
)
