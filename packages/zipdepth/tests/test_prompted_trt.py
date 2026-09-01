"""Static-batch ONNX/TensorRT export tests for ZipDepth-PromptDA."""

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from monopriors.models.depth_completion.zipdepth_prompt import load_zipdepth_prompt
from monopriors.models.depth_completion.zipdepth_prompt_export import (
    IMAGE_INPUT_NAME,
    PROMPT_INPUT_NAME,
    export_zipdepth_prompt_onnx,
    prepare_zipdepth_prompt_for_export,
)
from monopriors.models.relative_depth.zipdepth import download_zipdepth_checkpoint
from torch import Tensor, nn


class _FakePromptModel(nn.Module):
    """Small model double exposing the prompted export contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.fused = False

    def fuse_for_inference(self) -> "_FakePromptModel":
        self.fused = True
        return self

    def forward_with_range(self, image: Tensor, prompt_depth: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        depth: Tensor = image[:, :1] * self.weight
        return depth, prompt_depth.amin(dim=(1, 2, 3), keepdim=True), prompt_depth.amax(dim=(1, 2, 3), keepdim=True)


def test_export_uses_two_fp32_inputs_and_static_batch_eight(tmp_path: Path) -> None:
    checkpoint: Path = tmp_path / "zipdepth_base.pth"
    checkpoint.write_bytes(b"released")
    fake_model = _FakePromptModel()
    calls: list[dict[str, Any]] = []

    def fake_export(model: nn.Module, example_inputs: tuple[Tensor, ...], out_path: Path, **kwargs: Any) -> None:
        calls.append({"model": model, "inputs": example_inputs, "out_path": out_path, **kwargs})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"onnx")

    onnx_path: Path = export_zipdepth_prompt_onnx(
        checkpoint=checkpoint,
        image_hw=(768, 1024),
        batch_size=8,
        cache_dir=tmp_path / "cache",
        model_loader=lambda _checkpoint: fake_model,
        export_fn=fake_export,
        device="cpu",
    )

    call: dict[str, Any] = calls[0]
    assert onnx_path == call["out_path"]
    assert "768x1024_b8_fp16" in onnx_path.name
    assert call["input_names"] == [IMAGE_INPUT_NAME, PROMPT_INPUT_NAME]
    assert call["output_names"] == ["depth", "min_depth", "max_depth"]
    assert "dynamic_batch_max" not in call or call["dynamic_batch_max"] is None
    assert [tuple(tensor.shape) for tensor in call["inputs"]] == [(8, 3, 768, 1024), (8, 1, 192, 256)]
    assert all(tensor.dtype == torch.float32 for tensor in call["inputs"])
    assert fake_model.fused
    assert fake_model.weight.dtype == torch.float16


trt_parity = pytest.mark.skipif(
    os.environ.get("ZIPDEPTH_TRT_PARITY") != "1" or not torch.cuda.is_available(),
    reason="set ZIPDEPTH_TRT_PARITY=1 on a CUDA/TensorRT host",
)


def _parity_prompts(batch_size: int, device: torch.device) -> Tensor:
    """Build alternating holey and narrow-range metric prompts."""
    wide: Tensor = torch.linspace(0.2, 3.8, 192 * 256, device=device).reshape(1, 1, 192, 256)
    wide[:, :, 24:160:3, 32:224:4] = 0.0
    narrow: Tensor = torch.linspace(1.500, 1.503, 192 * 256, device=device).reshape(1, 1, 192, 256)
    prompts: Tensor = torch.cat([wide, narrow] * (batch_size // 2), dim=0)
    return prompts.contiguous()


@trt_parity
def test_holey_and_narrow_prompt_torch_onnx_trt_parity() -> None:
    """Compare the real fp16 graph across Torch, ONNX Runtime, and TensorRT."""
    from trtkit import OnnxCudaRuntime, TensorRtRuntime, TrtBuildConfig, ensure_engine, onnx_static_batch_size

    batch_size: int = 8
    cache_dir: Path = Path(os.environ.get("ZIPDEPTH_PROMPT_TRT_CACHE", "/tmp/zd-pda-research/trt"))
    checkpoint: Path = download_zipdepth_checkpoint()
    onnx_path: Path = export_zipdepth_prompt_onnx(checkpoint=checkpoint, batch_size=batch_size, cache_dir=cache_dir)
    assert onnx_static_batch_size(onnx_path) == batch_size
    engine_path: Path = ensure_engine(
        onnx_path,
        TrtBuildConfig(max_batch_size=batch_size, opt_batch_size=batch_size),
        cache_dir=cache_dir / "engines",
    )

    model: nn.Module = prepare_zipdepth_prompt_for_export(load_zipdepth_prompt(checkpoint), (768, 1024), torch.device("cuda"))
    image: Tensor = torch.rand((batch_size, 3, 768, 1024), dtype=torch.float32, device="cuda")
    prompt: Tensor = _parity_prompts(batch_size, torch.device("cuda"))
    with torch.inference_mode():
        torch_depth: Tensor = model(image, prompt).clone()
    inputs: dict[str, Tensor] = {IMAGE_INPUT_NAME: image, PROMPT_INPUT_NAME: prompt}
    onnx_depth: Tensor = OnnxCudaRuntime(onnx_path)(inputs)["depth"].clone()
    trt_depth: Tensor = TensorRtRuntime(engine_path)(inputs)["depth"].clone()

    for case_name, rows in (("holey", slice(0, None, 2)), ("narrow", slice(1, None, 2))):
        for backend_name, actual in (("onnx", onnx_depth), ("trt", trt_depth)):
            error: Tensor = (actual[rows] - torch_depth[rows]).abs()
            assert float(error.median()) < 0.003, f"{backend_name}:{case_name} median"
            assert float(error.max()) < 0.02, f"{backend_name}:{case_name} max"
