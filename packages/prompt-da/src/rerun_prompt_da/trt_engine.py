"""ONNX export and TensorRT engine caching for Prompt Depth Anything.

prompt-da owns the model-specific half of the acceleration path: it imports
the torch PromptDA network from monopriors and exports the ONNX interchange
graph itself (dynamic batch, checkpoint revision folded into the artifact
name). The generic half — engine building, cache keying, manifests — comes
from the shared ``trtkit`` package and is re-exported here for prompt-da's
historical import surface. ONNX files are portable; engines are machine-local
and rebuilt from ONNX on each target machine.
"""

import os
from pathlib import Path
from typing import Literal, TypeAlias

import torch
from trtkit import TrtBuildConfig, cached_engine_path, ensure_engine

__all__ = (
    "DEFAULT_CACHE_DIR",
    "ModelType",
    "ONNX_EXPORT_VERSION",
    "ONNX_OPSET",
    "PROMPT_DEPTH_HW",
    "TrtBuildConfig",
    "cached_engine_path",
    "ensure_engine",
    "export_promptda_onnx",
)

ModelType: TypeAlias = Literal["large", "small", "small-transparent"]

DEFAULT_CACHE_DIR: Path = Path(os.environ.get("PROMPTDA_TRT_CACHE", "~/.cache/prompt-da")).expanduser()
"""Cache root holding ``onnx/`` (portable) and ``trt/`` (machine-local) artifacts."""

PROMPT_DEPTH_HW: tuple[int, int] = (192, 256)
"""ARKit LiDAR prompt-depth resolution PromptDA was trained on."""

ONNX_OPSET: int = 18
"""Dynamo-exporter opset (fp16 Conv is valid here; TRT 11 parses up to 24)."""

ONNX_EXPORT_VERSION: int = 5
"""Bump whenever the export recipe or the vendored PromptDA implementation changes,
so cached ONNX graphs from older code are not silently reused.
v5: dynamo exporter, fp16 autocast compute baked into the graph (fp32 I/O) for
strongly-typed TensorRT 11 builds, plus ``min_val``/``max_val`` fusion-breaker
outputs — TensorRT 11.2's Myelin miscompiles the fusion that spans the prompt's
amin/amax reductions from the graph input to the final denormalize (garbage or
NaN depth in every precision); materializing the two scalars as engine outputs
splits that fusion and restores exact parity (0.7 mm median, ~52 FPS on sm_120,
equal to the TRT 10.13 weak-fp16 engine)."""


def export_promptda_onnx(
    model_type: ModelType = "large",
    image_hw: tuple[int, int] = (756, 1008),
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Export the monopriors PromptDA network to a dynamic-batch ONNX graph.

    The graph takes ``image`` (float32 ``[B,3,H,W]``, RGB in [0,1]) and
    ``prompt_depth`` (float32 ``[B,1,192,256]``, meters) and returns ``depth``
    (float32 ``[B,1,H,W]``, meters) plus tiny ``min_val``/``max_val`` outputs
    that exist only to break a miscompiling TensorRT fusion (see
    ``ONNX_EXPORT_VERSION``); consumers read ``depth`` and ignore the rest.
    Compute inside is fp16 (autocast traced into the graph — TensorRT 11 builds
    are strongly typed, so the graph's dtypes are the engine's). Only the batch
    axis is dynamic; H and W must be multiples of the DINOv2 patch size (14).

    Args:
        model_type: PromptDA checkpoint variant (monopriors ``NAME_TO_HFNAME`` key).
        image_hw: Static (height, width) the graph is exported at.
        cache_dir: Cache root; the file lands in ``cache_dir / "onnx"``.

    Returns:
        Path to the cached ONNX file (exported on first use).
    """
    height, width = image_hw
    if height % 14 != 0 or width % 14 != 0:
        raise ValueError(f"PromptDA image size must be a multiple of the 14px patch size, got {image_hw}.")

    from huggingface_hub import hf_hub_download
    from monopriors.models.depth_completion.prompt_da import NAME_TO_HFNAME
    from monopriors.third_party.promptda.promptda import PromptDA

    # Resolve the checkpoint first so its HF snapshot revision is part of the
    # cache identity — an updated checkpoint or export recipe (ONNX_EXPORT_VERSION)
    # must not silently reuse a stale graph.
    ckpt_path = Path(hf_hub_download(repo_id=NAME_TO_HFNAME[model_type], repo_type="model", filename="model.ckpt"))
    ckpt_rev: str = _checkpoint_revision(ckpt_path)
    onnx_dir: Path = cache_dir / "onnx"
    onnx_path: Path = onnx_dir / f"promptda-{model_type}_{height}x{width}_op{ONNX_OPSET}_v{ONNX_EXPORT_VERSION}_{ckpt_rev}.onnx"
    if onnx_path.exists():
        return onnx_path

    print(f"[prompt-da] exporting ONNX (one-time, may take a minute): {onnx_path.name}")
    model = PromptDA.from_pretrained(str(ckpt_path)).to("cuda").eval()

    class _ExportWrapper(torch.nn.Module):
        """fp32 I/O, fp16 autocast compute, and the fusion-breaker outputs.

        The re-computed ``amin``/``amax`` dedupe against the identical
        reductions inside ``PromptDA.normalize``, so marking them as outputs
        forces TensorRT to materialize the two scalars instead of fusing the
        input-side normalize with the output-side denormalize across the whole
        network — the fusion TRT 11.2 miscompiles (see ``ONNX_EXPORT_VERSION``).
        """

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, image: torch.Tensor, prompt_depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            with torch.autocast("cuda", dtype=torch.float16):
                depth = self.inner(image, prompt_depth)
            min_val = prompt_depth.amin(dim=(1, 2, 3), keepdim=True)
            max_val = prompt_depth.amax(dim=(1, 2, 3), keepdim=True)
            return depth.float(), min_val, max_val

    wrapper = _ExportWrapper(model).eval()

    # Trace at batch 2 so no op accidentally specializes on batch 1.
    dummy_image: torch.Tensor = torch.zeros((2, 3, height, width), dtype=torch.float32, device="cuda")
    dummy_prompt: torch.Tensor = torch.rand((2, 1, *PROMPT_DEPTH_HW), dtype=torch.float32, device="cuda") + 0.5
    onnx_dir.mkdir(parents=True, exist_ok=True)
    # pid-unique temp + atomic rename: concurrent exporters may duplicate work
    # but can never clobber each other's in-flight writes or publish a
    # truncated file.
    tmp_path: Path = onnx_path.with_name(f"{onnx_path.name}.part{os.getpid()}")
    batch_dim = torch.export.Dim("batch", min=1, max=64)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_prompt),
            str(tmp_path),
            input_names=["image", "prompt_depth"],
            output_names=["depth", "min_val", "max_val"],
            opset_version=ONNX_OPSET,
            # dynamo-native dynamic batch: dynamic_axes with dynamo=True goes
            # through a lossy conversion and produced miscompiled graphs.
            dynamic_shapes={"image": {0: batch_dim}, "prompt_depth": {0: batch_dim}},
            dynamo=True,
        )
    tmp_path.rename(onnx_path)
    del model
    torch.cuda.empty_cache()
    return onnx_path


def _checkpoint_revision(ckpt_path: Path) -> str:
    """Short revision identifying a resolved HF checkpoint file.

    Args:
        ckpt_path: Checkpoint path returned by ``hf_hub_download`` (its HF cache
            layout is ``…/snapshots/<commit>/model.ckpt``).

    Returns:
        First 8 chars of the snapshot commit, or a size-based tag for
        checkpoints outside the HF cache layout.
    """
    parts: tuple[str, ...] = ckpt_path.parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1][:8]
    return f"size{ckpt_path.stat().st_size}"
