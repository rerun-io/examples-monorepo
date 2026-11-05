"""The plan-driven Q/DQ rewrite replaces exactly the planned weights."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from trtkit.onnx_qdq import Fp8Conv, Fp8Linear, Nvfp4Linear, QdqPlan, quantize_qdq, save_with_external_data


def exported_graph() -> onnx.ModelProto:
    """Two linears reading one activation through torch's Transpose/Cast weight chain, and a Conv."""
    nodes: list[onnx.NodeProto] = []
    initializers: list[TensorProto] = []
    for name in ("fc1", "fc2"):
        initializers.append(numpy_helper.from_array(np.full((64, 64), 0.5, dtype=np.float32), f"inner.{name}.weight"))
        nodes.extend([
            helper.make_node("Cast", [f"inner.{name}.weight"], [f"{name}.cast"], to=TensorProto.FLOAT),
            helper.make_node("Transpose", [f"{name}.cast"], [f"{name}.t"], perm=[1, 0]),
            helper.make_node("MatMul", ["tokens", f"{name}.t"], [f"{name}.out"]),
        ])
    initializers.append(numpy_helper.from_array(np.ones((4, 3, 1, 1), dtype=np.float32), "inner.head.weight"))
    nodes.append(helper.make_node("Conv", ["image", "inner.head.weight"], ["head.out"]))
    graph = helper.make_graph(
        nodes, "exported",
        [helper.make_tensor_value_info("tokens", TensorProto.FLOAT, [2, 64]), helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in ("fc1.out", "fc2.out", "head.out")],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])


def test_planned_modules_are_quantized_and_shared_inputs_quantized_once(tmp_path) -> None:
    plan = QdqPlan({"fc1": Nvfp4Linear(2.0), "fc2": Nvfp4Linear(3.0), "head": Fp8Conv(1.0)}, ())
    model = quantize_qdq(exported_graph(), plan)
    types = {tensor.name: tensor.data_type for tensor in model.graph.initializer}
    assert not [name for name in types if name.startswith("inner.")]
    assert types["fc1.w_fp4"] == types["fc2.w_fp4"] == TensorProto.FLOAT4E2M1
    assert types["head.w_fp8"] == TensorProto.FLOAT8E4M3FN
    assert [node.op_type for node in model.graph.node].count("TRT_FP4DynamicQuantize") == 1
    assert any(opset.domain == "trt.plugins" for opset in model.opset_import)
    save_with_external_data(model, tmp_path / "out" / "model.onnx")
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == ["model.onnx", "model.onnx.data"]
    assert {tensor.name for tensor in onnx.load(tmp_path / "out" / "model.onnx").graph.initializer} == set(types)


def test_unplanned_linears_and_mismatched_precisions_are_rejected() -> None:
    with pytest.raises(ValueError, match="no precision for linear fc2"):
        quantize_qdq(exported_graph(), QdqPlan({"fc1": Fp8Linear(1.0)}, ()))
    with pytest.raises(ValueError, match="Fp8Linear does not apply to Conv head"):
        quantize_qdq(exported_graph(), QdqPlan({"fc1": Fp8Linear(1.0), "fc2": Fp8Linear(1.0), "head": Fp8Linear(1.0)}, ()))
    with pytest.raises(ValueError, match="missing_module"):
        quantize_qdq(exported_graph(), QdqPlan({"fc1": Fp8Linear(1.0), "fc2": Fp8Linear(1.0), "missing_module": Fp8Conv(1.0)}, ()))
