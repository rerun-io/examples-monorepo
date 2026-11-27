"""ONNX Runtime on its CPU provider behind the trtkit runtime contract, and the ``auto`` backend that picks it off NVIDIA.

The CPU runtime is what engine-free tests and GPU-less hosts run: the same ONNX artifact the TensorRT engine is built
from, CPU tensors in and out.
"""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import Tensor

from trtkit import AutoBackendConfig, OnnxBackendConfig, OnnxRuntime, TensorRtBackendConfig, create_runtime_from_onnx, resolve_backend


def _affine_graph(path: Path, batch: int | str) -> Path:
    """``y = relu(x * 2 + 1)`` over ``[batch, 3]`` and ``s = sum(x)`` per sample, with ``s``'s length left symbolic."""
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["x", "two"], ["scaled"]),
            helper.make_node("Add", ["scaled", "one"], ["shifted"]),
            helper.make_node("Relu", ["shifted"], ["y"]),
            helper.make_node("ReduceSum", ["x", "last"], ["s"], keepdims=1),
        ],
        "affine",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 3]), helper.make_tensor_value_info("s", TensorProto.FLOAT, [batch, "k"])],
        initializer=[
            numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "last"),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10), path)
    return path


def test_the_cpu_runtime_runs_a_dynamic_batch_graph_on_cpu_tensors(tmp_path: Path) -> None:
    runtime: OnnxRuntime = OnnxRuntime(_affine_graph(tmp_path / "dynamic.onnx", "batch"), device="cpu", max_batch_size=4)
    assert runtime.spec.max_batch_size == 4
    assert [(spec.name, spec.shape) for spec in runtime.spec.outputs] == [("y", (3,)), ("s", (1,))], "the probe run resolves the symbolic dim"
    x: Tensor = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 0.5]])
    outputs: dict[str, Tensor] = runtime({"x": x})
    assert outputs["y"].device.type == "cpu"
    torch.testing.assert_close(outputs["y"], torch.relu(x * 2 + 1))
    torch.testing.assert_close(outputs["s"], x.sum(dim=1, keepdim=True))


def test_a_static_batch_graph_pads_smaller_batches(tmp_path: Path) -> None:
    runtime: OnnxRuntime = OnnxRuntime(_affine_graph(tmp_path / "static.onnx", 4), device="cpu")
    assert runtime.spec.max_batch_size == 4, "a baked batch is the graph's own"
    x: Tensor = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(runtime({"x": x})["y"], torch.relu(x * 2 + 1))


def test_inputs_on_another_device_are_refused(tmp_path: Path) -> None:
    runtime: OnnxRuntime = OnnxRuntime(_affine_graph(tmp_path / "dynamic.onnx", "batch"), device="cpu")
    with pytest.raises(ValueError, match="must live on cpu"):
        runtime({"x": torch.zeros((1, 3), dtype=torch.float32, device="meta")})


def test_auto_takes_tensorrt_on_nvidia_and_onnx_runtime_on_the_cpu_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    auto: AutoBackendConfig = AutoBackendConfig(max_batch_size=16, opt_batch_size=4)
    monkeypatch.setattr("trtkit.backends.tensorrt_available", lambda: True)
    assert resolve_backend(auto) == TensorRtBackendConfig(max_batch_size=16, opt_batch_size=4)
    monkeypatch.setattr("trtkit.backends.tensorrt_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_backend(auto) == OnnxBackendConfig(device="cpu", max_batch_size=16)


def test_auto_on_a_cuda_host_without_tensorrt_stays_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """The models feed CUDA frames wherever torch has CUDA, so ``auto`` keeps ONNX Runtime there (the old default)."""
    monkeypatch.setattr("trtkit.backends.tensorrt_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_backend(AutoBackendConfig(max_batch_size=16)) == OnnxBackendConfig(device="cuda", max_batch_size=16)
    explicit: OnnxBackendConfig = OnnxBackendConfig(device_id=1)
    assert resolve_backend(explicit) is explicit, "a concrete backend is the caller's choice"


def test_auto_off_nvidia_builds_the_cpu_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("trtkit.backends.tensorrt_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = create_runtime_from_onnx(_affine_graph(tmp_path / "dynamic.onnx", "batch"), AutoBackendConfig())
    assert isinstance(runtime, OnnxRuntime) and runtime.device == torch.device("cpu")
