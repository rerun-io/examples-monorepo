"""Static-batch ONNX/TensorRT export tests for ZipDepth-PromptDA."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pytest
import torch
from jaxtyping import Float32
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt
from monopriors.models.depth_completion.zipdepth_prompt_export import (
    IMAGE_INPUT_NAME,
    PROMPT_INPUT_NAME,
    ModelLoader,
    PromptedDepthModel,
    _ExportOutputs,
    export_zipdepth_prompt_onnx,
)
from monopriors.models.relative_depth.zipdepth import download_zipdepth_checkpoint
from monopriors.models.zipdepth_checkpoint import RANGE_MARGIN_M_KEY
from torch import Tensor, nn

from zipdepth.apis import prompted_trt

ExportInputs: TypeAlias = tuple[Float32[Tensor, "b 3 h w"], Float32[Tensor, "b 1 192 256"]]


@dataclass(frozen=True, slots=True)
class _ExportCall:
    """Typed ONNX export call captured by the test boundary."""

    model: nn.Module
    """Export wrapper module."""
    inputs: ExportInputs
    """Example tensors supplied to the exporter."""
    out_path: Path
    """Target ONNX path."""
    input_names: list[str]
    """Ordered ONNX input names."""
    output_names: list[str]
    """Ordered ONNX output names."""
    dynamic_batch_max: int | None
    """Dynamic batch limit, or None for a static graph."""


class _FakePromptModel(nn.Module):
    """Small model double exposing the prompted export contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.fused = False

    def fuse_for_inference(self) -> "_FakePromptModel":
        self.fused = True
        return self

    def forward_with_range(
        self,
        image: Float32[Tensor, "b 3 h w"],
        prompt_depth: Float32[Tensor, "b 1 192 256"],
    ) -> tuple[Float32[Tensor, "b 1 h w"], Float32[Tensor, "b 1 1 1"], Float32[Tensor, "b 1 1 1"]]:
        depth: Float32[Tensor, "b 1 h w"] = image[:, :1] * self.weight
        return depth, prompt_depth.amin(dim=(1, 2, 3), keepdim=True), prompt_depth.amax(dim=(1, 2, 3), keepdim=True)


def _fake_loader(fake_model: _FakePromptModel) -> ModelLoader:
    """Return a loader boundary that ignores the checkpoint and the margin."""

    def load(_checkpoint: Path, *, range_margin_m: float | None = None) -> PromptedDepthModel:
        return fake_model

    return load


def test_fake_model_satisfies_prompted_export_protocol() -> None:
    """Keep exporter tests structural rather than coupled to ZipDepthPrompt."""
    assert isinstance(_FakePromptModel(), PromptedDepthModel)


def test_export_uses_two_inputs_with_export_dtype_and_static_batch_eight(tmp_path: Path) -> None:
    checkpoint: Path = tmp_path / "zipdepth_base.pth"
    checkpoint.write_bytes(b"released")
    fake_model = _FakePromptModel()
    calls: list[_ExportCall] = []

    def fake_export(model: nn.Module, example_inputs: ExportInputs, out_path: Path, **kwargs: object) -> None:
        input_names: object = kwargs.get("input_names")
        output_names: object = kwargs.get("output_names")
        dynamic_batch_max: object = kwargs.get("dynamic_batch_max")
        if not isinstance(input_names, list) or not all(isinstance(name, str) for name in input_names):
            raise AssertionError("export input_names must be a list of strings")
        if not isinstance(output_names, list) or not all(isinstance(name, str) for name in output_names):
            raise AssertionError("export output_names must be a list of strings")
        if dynamic_batch_max is not None and not isinstance(dynamic_batch_max, int):
            raise AssertionError("dynamic_batch_max must be an integer or None")
        calls.append(
            _ExportCall(
                model=model,
                inputs=example_inputs,
                out_path=out_path,
                input_names=[name for name in input_names if isinstance(name, str)],
                output_names=[name for name in output_names if isinstance(name, str)],
                dynamic_batch_max=dynamic_batch_max,
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"onnx")

    onnx_path: Path = export_zipdepth_prompt_onnx(
        checkpoint=checkpoint,
        image_hw=(768, 1024),
        batch_size=8,
        cache_dir=tmp_path / "cache",
        range_margin_m=0.0,
        model_loader=_fake_loader(fake_model),
        export_fn=fake_export,
        device="cpu",
    )

    call: _ExportCall = calls[0]
    assert onnx_path == call.out_path
    assert "768x1024_b8_fp16" in onnx_path.name
    assert onnx_path.name.endswith("_m0.00.onnx")
    assert call.input_names == [IMAGE_INPUT_NAME, PROMPT_INPUT_NAME]
    assert call.output_names == ["depth", "min_depth", "max_depth"]
    assert call.dynamic_batch_max is None
    assert [tuple(tensor.shape) for tensor in call.inputs] == [(8, 3, 768, 1024), (8, 1, 192, 256)]
    assert all(tensor.dtype == torch.float32 for tensor in call.inputs)
    assert fake_model.fused
    assert fake_model.weight.dtype == torch.float16


def test_export_bakes_the_checkpoint_range_margin_into_the_graph_and_its_cache_key(tmp_path: Path) -> None:
    """Export the head the checkpoint was trained with, keeping the binding names."""
    trained: ZipDepthPrompt = ZipDepthPrompt(range_margin_m=3.9)
    checkpoint: Path = tmp_path / "final_model.pth"
    torch.save({"model_state_dict": trained.state_dict(), RANGE_MARGIN_M_KEY: 3.9}, checkpoint)
    exported_models: list[ZipDepthPrompt] = []

    def fake_export(model: nn.Module, _inputs: ExportInputs, out_path: Path, **kwargs: object) -> None:
        assert kwargs["input_names"] == [IMAGE_INPUT_NAME, PROMPT_INPUT_NAME]
        assert kwargs["output_names"] == ["depth", "min_depth", "max_depth"]
        assert isinstance(model, _ExportOutputs)
        inner: object = model.model
        assert isinstance(inner, ZipDepthPrompt)
        exported_models.append(inner)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"onnx")

    onnx_path: Path = export_zipdepth_prompt_onnx(
        checkpoint=checkpoint,
        image_hw=(64, 96),
        batch_size=1,
        cache_dir=tmp_path / "cache",
        export_fn=fake_export,
        device="cpu",
    )

    assert exported_models[0].range_margin_m == 3.9
    # The cache short-circuits before the model loads, so the margin has to be in the name.
    assert onnx_path.name.endswith("_m3.90.onnx")


def test_export_caches_one_graph_per_margin(tmp_path: Path) -> None:
    """Never serve a graph built for a different output range out of the cache."""
    checkpoint: Path = tmp_path / "final_model.pth"
    torch.save({"model_state_dict": ZipDepthPrompt().state_dict()}, checkpoint)
    fake_model = _FakePromptModel()

    def fake_export(_model: nn.Module, _inputs: ExportInputs, out_path: Path, **_kwargs: object) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"onnx")

    paths: list[Path] = [
        export_zipdepth_prompt_onnx(
            checkpoint=checkpoint,
            image_hw=(64, 96),
            batch_size=1,
            cache_dir=tmp_path / "cache",
            range_margin_m=margin_m,
            model_loader=_fake_loader(fake_model),
            export_fn=fake_export,
            device="cpu",
        )
        for margin_m in (0.0, 3.9)
    ]

    assert paths[0] != paths[1]


trt_parity = pytest.mark.skipif(
    os.environ.get("ZIPDEPTH_TRT_PARITY") != "1" or not torch.cuda.is_available(),
    reason="set ZIPDEPTH_TRT_PARITY=1 on a CUDA/TensorRT host",
)


@trt_parity
def test_holey_and_narrow_prompt_torch_onnx_trt_parity() -> None:
    """Compare the real fp16 graph across Torch, ONNX Runtime, and TensorRT."""
    from trtkit import TensorRtRuntime, TrtBuildConfig, ensure_engine, onnx_static_batch_size

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
    assert engine_path.is_file()

    inputs: prompted_trt.ParityInputs = prompted_trt._parity_inputs(batch_size, (768, 1024), torch.device("cuda"))
    parity: prompted_trt.TrtParityReport = prompted_trt.verify_three_backend_parity(
        checkpoint,
        onnx_path,
        (768, 1024),
        inputs,
        TensorRtRuntime(engine_path),
    )
    summaries: tuple[prompted_trt.ParityError, ...] = (
        parity.holey.onnx_vs_torch,
        parity.holey.trt_vs_torch,
        parity.narrow.onnx_vs_torch,
        parity.narrow.trt_vs_torch,
    )
    assert max(summary.median_abs_m for summary in summaries) < 0.003
    assert max(summary.max_abs_m for summary in summaries) < 0.02
