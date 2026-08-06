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
from trtkit import TrtBuildConfig, TrtPrecision, cached_engine_path, ensure_engine

__all__ = (
    "DEFAULT_CACHE_DIR",
    "ModelType",
    "ONNX_EXPORT_VERSION",
    "ONNX_OPSET",
    "PROMPT_DEPTH_HW",
    "TrtBuildConfig",
    "TrtPrecision",
    "cached_engine_path",
    "ensure_engine",
    "export_promptda_onnx",
)

ModelType: TypeAlias = Literal["large", "small", "small-transparent"]

DEFAULT_CACHE_DIR: Path = Path(os.environ.get("PROMPTDA_TRT_CACHE", "~/.cache/prompt-da")).expanduser()
"""Cache root holding ``onnx/`` (portable) and ``trt/`` (machine-local) artifacts."""

PROMPT_DEPTH_HW: tuple[int, int] = (192, 256)
"""ARKit LiDAR prompt-depth resolution PromptDA was trained on."""

ONNX_OPSET: int = 17
"""Legacy-exporter opset; TRT 10.13's parser chokes on dynamo exports (see mamma)."""

ONNX_EXPORT_VERSION: int = 1
"""Bump whenever the export recipe or the vendored PromptDA implementation changes,
so cached ONNX graphs from older code are not silently reused."""


def export_promptda_onnx(
    model_type: ModelType = "large",
    image_hw: tuple[int, int] = (756, 1008),
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Export the monopriors PromptDA network to a dynamic-batch ONNX graph.

    The graph takes ``image`` (float32 ``[B,3,H,W]``, RGB in [0,1]) and
    ``prompt_depth`` (float32 ``[B,1,192,256]``, meters) and returns ``depth``
    (float32 ``[B,1,H,W]``, meters). Only the batch axis is dynamic; H and W
    must be multiples of the DINOv2 patch size (14).

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
    # Trace at batch 2 so no op accidentally specializes on batch 1.
    dummy_image: torch.Tensor = torch.zeros((2, 3, height, width), dtype=torch.float32, device="cuda")
    dummy_prompt: torch.Tensor = torch.rand((2, 1, *PROMPT_DEPTH_HW), dtype=torch.float32, device="cuda") + 0.5
    onnx_dir.mkdir(parents=True, exist_ok=True)
    # pid-unique temp + atomic rename: concurrent exporters may duplicate work
    # but can never clobber each other's in-flight writes or publish a
    # truncated file.
    tmp_path: Path = onnx_path.with_name(f"{onnx_path.name}.part{os.getpid()}")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (dummy_image, dummy_prompt),
            str(tmp_path),
            input_names=["image", "prompt_depth"],
            output_names=["depth"],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            dynamic_axes={
                "image": {0: "batch"},
                "prompt_depth": {0: "batch"},
                "depth": {0: "batch"},
            },
            dynamo=False,
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
