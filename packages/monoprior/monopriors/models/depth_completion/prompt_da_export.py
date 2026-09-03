"""PromptDA ONNX export beside the model-specific runtime adapter."""

from pathlib import Path

import torch
from jaxtyping import Float32
from torch import Tensor, nn
from trtkit import sweep_stale_onnx_exports

from monopriors.models.depth_completion.base_completion_depth import DEPTH_OUTPUT_NAME, IMAGE_INPUT_NAME, PROMPT_DEPTH_HW, PROMPT_INPUT_NAME
from monopriors.models.depth_completion.prompt_da import DEFAULT_PROMPTDA_CACHE_DIR, NAME_TO_HFNAME, ModelType

DEFAULT_CACHE_DIR: Path = DEFAULT_PROMPTDA_CACHE_DIR
"""Compatibility name for the PromptDA artifact cache root."""
ONNX_EXPORT_VERSION: int = 6
"""PromptDA export recipe version used in cache identities."""


class _FusionBreakerOutputs(nn.Module):
    """Materialize prompt extrema to prevent a miscompiled TensorRT fusion."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model: nn.Module = model

    def forward(
        self,
        image: Float32[Tensor, "b 3 h w"],
        prompt_depth: Float32[Tensor, "b 1 192 256"],
    ) -> tuple[Float32[Tensor, "b 1 h w"], Float32[Tensor, "b 1 1 1"], Float32[Tensor, "b 1 1 1"]]:
        """Return depth plus the prompt range fusion barriers."""
        depth_bchw: Float32[Tensor, "b 1 h w"] = self.model(image, prompt_depth)
        min_val_b: Float32[Tensor, "b 1 1 1"] = prompt_depth.amin(dim=(1, 2, 3), keepdim=True)
        max_val_b: Float32[Tensor, "b 1 1 1"] = prompt_depth.amax(dim=(1, 2, 3), keepdim=True)
        return depth_bchw, min_val_b, max_val_b


def export_promptda_onnx(
    model_type: ModelType = "large",
    image_hw: tuple[int, int] = (756, 1008),
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Export PromptDA to a cached dynamic-batch ONNX graph.

    Args:
        model_type: PromptDA checkpoint variant.
        image_hw: Static image height and width, both divisible by 14.
        cache_dir: Cache root; the graph is stored below ``onnx``.

    Returns:
        Existing or newly exported ONNX graph path.
    """
    height, width = image_hw
    if height % 14 != 0 or width % 14 != 0:
        raise ValueError(f"PromptDA image size must be a multiple of the 14px patch size, got {image_hw}.")

    from huggingface_hub import hf_hub_download

    from monopriors.third_party.promptda.promptda import PromptDA

    checkpoint_path: Path = Path(hf_hub_download(repo_id=NAME_TO_HFNAME[model_type], repo_type="model", filename="model.ckpt"))
    checkpoint_revision: str = _checkpoint_revision(checkpoint_path)
    onnx_dir: Path = cache_dir / "onnx"
    onnx_path: Path = onnx_dir / f"promptda-{model_type}_{height}x{width}_v{ONNX_EXPORT_VERSION}_{checkpoint_revision}.onnx"
    if onnx_path.exists():
        return onnx_path

    print(f"[prompt-da] exporting ONNX (one-time, may take a minute): {onnx_path.name}")
    model: nn.Module = PromptDA.from_pretrained(str(checkpoint_path)).to("cuda").eval()
    wrapper: nn.Module = _FusionBreakerOutputs(model).eval()

    from trtkit import export_onnx

    dummy_image_bchw: Float32[Tensor, "2 3 h w"] = torch.zeros((2, 3, height, width), dtype=torch.float32, device="cuda")
    dummy_prompt_bchw: Float32[Tensor, "2 1 192 256"] = torch.rand(
        (2, 1, *PROMPT_DEPTH_HW), dtype=torch.float32, device="cuda"
    ) + 0.5
    export_onnx(
        wrapper,
        (dummy_image_bchw, dummy_prompt_bchw),
        onnx_path,
        input_names=[IMAGE_INPUT_NAME, PROMPT_INPUT_NAME],
        output_names=[DEPTH_OUTPUT_NAME, "min_val", "max_val"],
        compute_dtype=torch.float16,
        dynamic_batch_max=64,
    )
    del wrapper, model
    torch.cuda.empty_cache()
    stale_prefix: str = f"promptda-{model_type}_{height}x{width}_"
    sweep_stale_onnx_exports(onnx_dir, stale_prefix, keep_paths={onnx_path})
    return onnx_path


def _checkpoint_revision(checkpoint_path: Path) -> str:
    """Return a short Hugging Face revision or size-based checkpoint tag."""
    parts: tuple[str, ...] = checkpoint_path.parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1][:8]
    return f"size{checkpoint_path.stat().st_size}"


__all__ = (
    "DEFAULT_CACHE_DIR",
    "ONNX_EXPORT_VERSION",
    "PROMPT_DEPTH_HW",
    "export_promptda_onnx",
)
