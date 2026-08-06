"""Three-backend parity: the same network must agree on torch, ONNX Runtime, and TensorRT.

Uses a tiny synthetic two-input/two-output convnet so the test is hermetic (no
checkpoint downloads) while still exercising the full contract: dict I/O,
static-batch padding, IOBinding, engine build + cache, and output slicing.
"""

from pathlib import Path

import pytest
import torch
from torch import Tensor

from posekit.runtimes import (
    OnnxCudaRuntime,
    RuntimeSpec,
    TensorRtRuntime,
    TensorSpec,
    TorchRuntime,
    TrtBuildConfig,
    ensure_engine,
    validate_runtime_inputs,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

STATIC_BATCH: int = 4
IMAGE_SPEC = TensorSpec(name="image", shape=(3, 16, 16), dtype=torch.float32)
MASK_SPEC = TensorSpec(name="mask", shape=(1, 16, 16), dtype=torch.float32)
FEATURES_SPEC = TensorSpec(name="features", shape=(8, 8, 8), dtype=torch.float32)
LOGITS_SPEC = TensorSpec(name="logits", shape=(5,), dtype=torch.float32)


class TinyNet(torch.nn.Module):
    """Two-input, two-output network exercising the dict runtime contract."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.conv = torch.nn.Conv2d(4, 8, 3, stride=2, padding=1)
        self.head = torch.nn.Linear(8, 5)

    def forward(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        features: Tensor = self.conv(torch.cat([image, mask], dim=1))
        logits: Tensor = self.head(features.mean(dim=(2, 3)))
        return features, logits


def _example_inputs(batch: int, device: str) -> dict[str, Tensor]:
    generator: torch.Generator = torch.Generator().manual_seed(3)
    return {
        "image": torch.rand((batch, 3, 16, 16), generator=generator).to(device),
        "mask": torch.rand((batch, 1, 16, 16), generator=generator).to(device),
    }


def _export_tiny_onnx(module: TinyNet, tmp_path: Path, *, dynamic_batch: bool = False) -> Path:
    onnx_path: Path = tmp_path / ("tinynet_dyn.onnx" if dynamic_batch else "tinynet_b4.onnx")
    dummy: dict[str, Tensor] = _example_inputs(STATIC_BATCH, "cuda")
    torch.onnx.export(
        module.cuda().eval(),
        (dummy["image"], dummy["mask"]),
        str(onnx_path),
        input_names=["image", "mask"],
        output_names=["features", "logits"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={name: {0: "batch"} for name in ("image", "mask", "features", "logits")} if dynamic_batch else None,
    )
    return onnx_path


def test_validate_runtime_inputs_errors() -> None:
    spec = RuntimeSpec(inputs=(IMAGE_SPEC, MASK_SPEC), outputs=(LOGITS_SPEC,), max_batch_size=STATIC_BATCH)
    inputs: dict[str, Tensor] = {"image": torch.zeros((2, 3, 16, 16)), "mask": torch.zeros((2, 1, 16, 16))}
    assert validate_runtime_inputs(spec, inputs) == 2
    with pytest.raises(ValueError, match="expects inputs"):
        validate_runtime_inputs(spec, {"image": inputs["image"]})
    with pytest.raises(ValueError, match="per-sample shape"):
        validate_runtime_inputs(spec, {"image": inputs["image"], "mask": torch.zeros((2, 1, 8, 8))})
    with pytest.raises(ValueError, match="share one batch"):
        validate_runtime_inputs(spec, {"image": inputs["image"], "mask": torch.zeros((3, 1, 16, 16))})
    with pytest.raises(ValueError, match="max batch"):
        validate_runtime_inputs(spec, _example_inputs(STATIC_BATCH + 1, "cpu"))


@cuda_only
def test_three_backend_parity(tmp_path: Path) -> None:
    module = TinyNet().cuda().eval()
    torch_runtime = TorchRuntime(
        module, input_specs=(IMAGE_SPEC, MASK_SPEC), output_specs=(FEATURES_SPEC, LOGITS_SPEC), max_batch_size=STATIC_BATCH
    )
    onnx_path: Path = _export_tiny_onnx(module, tmp_path)
    onnx_runtime = OnnxCudaRuntime(onnx_path)
    # TRT runs the dynamic-batch export: profile 1..STATIC_BATCH, true-batch execution.
    dynamic_onnx: Path = _export_tiny_onnx(module, tmp_path, dynamic_batch=True)
    engine_path: Path = ensure_engine(
        dynamic_onnx, TrtBuildConfig(max_batch_size=STATIC_BATCH, opt_batch_size=2, precision="fp32"), cache_dir=tmp_path / "trt"
    )
    trt_runtime = TensorRtRuntime(engine_path)
    assert onnx_runtime.spec.max_batch_size == STATIC_BATCH
    assert trt_runtime.spec.max_batch_size == STATIC_BATCH

    # Full static batch and a smaller batch exercising the padding paths.
    for batch in (STATIC_BATCH, STATIC_BATCH - 1):
        inputs: dict[str, Tensor] = _example_inputs(batch, "cuda")
        reference: dict[str, Tensor] = torch_runtime(inputs)
        for name, runtime in (("onnx", onnx_runtime), ("tensorrt", trt_runtime)):
            outputs: dict[str, Tensor] = runtime(inputs)
            for output_name in ("features", "logits"):
                assert outputs[output_name].shape[0] == batch
                assert outputs[output_name].device.type == "cuda"
                torch.testing.assert_close(
                    outputs[output_name], reference[output_name], rtol=1e-3, atol=1e-4, msg=f"{name}:{output_name} batch={batch}"
                )


@cuda_only
def test_tensorrt_cuda_graph_replay(tmp_path: Path) -> None:
    module = TinyNet().cuda().eval()
    onnx_path: Path = _export_tiny_onnx(module, tmp_path, dynamic_batch=True)
    engine_path: Path = ensure_engine(
        onnx_path, TrtBuildConfig(max_batch_size=STATIC_BATCH, opt_batch_size=2, precision="fp32"), cache_dir=tmp_path / "trt"
    )
    graph_runtime = TensorRtRuntime(engine_path, use_cuda_graph=True)
    plain_runtime = TensorRtRuntime(engine_path)
    for batch in (STATIC_BATCH, 2):
        inputs: dict[str, Tensor] = _example_inputs(batch, "cuda")
        expected: dict[str, Tensor] = {name: tensor.clone() for name, tensor in plain_runtime(inputs).items()}
        outputs: dict[str, Tensor] = graph_runtime(inputs)
        for output_name in ("features", "logits"):
            torch.testing.assert_close(outputs[output_name], expected[output_name], rtol=1e-4, atol=1e-5)

@cuda_only
def test_onnx_runtime_on_non_default_stream(tmp_path: Path) -> None:
    """Inputs produced on a side stream must be visible to the ORT run (and its
    outputs to the caller) — the cross-stream ordering contract."""
    module = TinyNet().cuda().eval()
    torch_runtime = TorchRuntime(
        module, input_specs=(IMAGE_SPEC, MASK_SPEC), output_specs=(FEATURES_SPEC, LOGITS_SPEC), max_batch_size=STATIC_BATCH
    )
    onnx_runtime = OnnxCudaRuntime(_export_tiny_onnx(module, tmp_path))
    side_stream: torch.cuda.Stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        inputs: dict[str, Tensor] = _example_inputs(STATIC_BATCH, "cuda")
        # A long device-side spin keeps the side stream busy so the staged
        # inputs genuinely arrive late — without the runtime's cross-stream
        # waits, ORT would read half-written buffers and the caller-side clone
        # below would read stale outputs.
        torch.cuda._sleep(50_000_000)
        inputs["image"] = (inputs["image"] * 1.0).contiguous()
        outputs: dict[str, Tensor] = onnx_runtime(inputs)
        logits: Tensor = outputs["logits"].clone()
    torch.cuda.synchronize()
    reference: dict[str, Tensor] = torch_runtime(inputs)
    torch.testing.assert_close(logits, reference["logits"], rtol=1e-3, atol=1e-4)
