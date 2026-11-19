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


def test_plan_can_explicitly_preserve_linear_and_attention_precision() -> None:
    """A precision ablation retains the selected source operations and weights."""
    from trtkit.onnx_qdq import Fp8Attention, KeepPrecision

    model = exported_graph()
    model.graph.node.extend([
        helper.make_node("Attention", ["fc1.out", "fc1.out", "fc1.out"], ["attention.original"]),
        helper.make_node("Attention", ["fc2.out", "fc2.out", "fc2.out"], ["attention.quantized"]),
    ])
    model.graph.output.extend([
        helper.make_tensor_value_info("attention.original", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("attention.quantized", TensorProto.FLOAT, None),
    ])
    source_weight: bytes = model.graph.initializer[0].SerializeToString()
    plan = QdqPlan(
        {"fc1": KeepPrecision(), "fc2": Fp8Linear(1.0)},
        (KeepPrecision(), Fp8Attention("attention.second", (1.0, 1.0, 1.0))),
    )
    result = quantize_qdq(model, plan)
    weights = {tensor.name: tensor for tensor in result.graph.initializer}
    assert weights["inner.fc1.weight"].SerializeToString() == source_weight
    assert "inner.fc2.weight" not in weights
    assert weights["fc2.w_fp8"].data_type == TensorProto.FLOAT8E4M3FN
    attention = [node for node in result.graph.node if node.op_type == "Attention"]
    assert list(attention[0].input) == ["fc1.out"] * 3
    assert list(attention[1].input) == [f"attention.second.{axis}.a_dq8" for axis in "qkv"]


def test_fp16_scale_dtype_types_fp8_scales_for_fp16_graphs() -> None:
    """FP8 Q/DQ scales follow the plan's scale dtype, so fp16-compute exports stay strongly typed."""
    plan = QdqPlan({"fc1": Fp8Linear(2.0), "fc2": Fp8Linear(2.0), "head": Fp8Conv(1.0)}, (), scale_dtype="float16")
    types = {tensor.name: tensor.data_type for tensor in quantize_qdq(exported_graph(), plan).graph.initializer}
    assert types["fc1.a_s8"] == types["fc1.w_s8"] == types["head.w_s8"] == types["head.a_s8"] == TensorProto.FLOAT16
    assert types["fc1.w_fp8"] == TensorProto.FLOAT8E4M3FN
    with pytest.raises(ValueError, match="requires bfloat16 scales"):
        quantize_qdq(exported_graph(), QdqPlan({"fc1": Nvfp4Linear(1.0), "fc2": Nvfp4Linear(1.0)}, (), scale_dtype="float16"))


def test_a_weight_two_matmuls_apply_is_quantized_once_with_a_range_per_input() -> None:
    """A weight stored in the compute dtype (Transpose alone) that two MatMuls apply to different activations, as a
    two-image matcher does: one FP8 weight, one input scale per activation."""
    weight = numpy_helper.from_array(np.full((8, 8), 0.5, dtype=np.float16), "blocks.0.proj.weight")
    nodes = [helper.make_node("Transpose", ["blocks.0.proj.weight"], ["proj.t"], perm=[1, 0])]
    nodes += [helper.make_node("MatMul", [f"x{image}", "proj.t"], [f"y{image}"]) for image in (0, 1)]
    graph = helper.make_graph(
        nodes, "two_images",
        [helper.make_tensor_value_info(f"x{image}", TensorProto.FLOAT16, [2, 8]) for image in (0, 1)],
        [helper.make_tensor_value_info(f"y{image}", TensorProto.FLOAT16, [2, 8]) for image in (0, 1)],
        [weight],
    )
    plan = QdqPlan({"blocks.0.proj": Fp8Linear({"x0": 2.0, "x1": 4.0})}, (), scale_dtype="float16")
    result = quantize_qdq(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]), plan)
    tensors = {tensor.name: tensor for tensor in result.graph.initializer}
    assert [name for name, tensor in tensors.items() if tensor.data_type == TensorProto.FLOAT8E4M3FN and tensor.dims] == ["blocks.0.proj.w_fp8"]
    matmuls = [node for node in result.graph.node if node.op_type == "MatMul"]
    assert [node.input[1] for node in matmuls] == ["blocks.0.proj.w_t"] * 2
    assert [node.input[0] for node in matmuls] == ["blocks.0.proj.a_dq8", "blocks.0.proj@x1.a_dq8"]
    scales = [float(numpy_helper.to_array(tensors[name])) for name in ("blocks.0.proj.a_s8", "blocks.0.proj@x1.a_s8")]
    np.testing.assert_allclose(scales, [2.0 / 448.0, 4.0 / 448.0], rtol=1e-3)
    onnx.checker.check_model(result)


def test_softmax_attention_gets_fp8_q_k_probabilities_and_v() -> None:
    """MatMul(q, k^T) -> Softmax -> MatMul(p, v): Q/DQ before both MatMuls, the probabilities on a fixed [0, 1] range;
    an attention left out of the mapping keeps its precision."""
    from trtkit.onnx_qdq import SOFTMAX_AMAX, Fp8Attention, quantize_softmax_attention, softmax_attentions

    nodes: list[onnx.NodeProto] = []
    for block in (0, 1):
        nodes += [
            helper.make_node("MatMul", ["q", "kt"], [f"logits{block}"]),
            helper.make_node("Softmax", [f"logits{block}"], [f"p{block}"], axis=-1),
            helper.make_node("MatMul", [f"p{block}", "v"], [f"out{block}"]),
        ]
    graph = helper.make_graph(
        nodes, "attention",
        [helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape) for name, shape in (("q", [4, 8]), ("kt", [8, 4]), ("v", [4, 8]))],
        [helper.make_tensor_value_info(f"out{block}", TensorProto.BFLOAT16, [4, 8]) for block in (0, 1)],
    )
    assert [softmax.output[0] for _, softmax, _ in softmax_attentions(graph)] == ["p0", "p1"]
    quantize_softmax_attention(graph, {"p1": Fp8Attention("block1", (2.0, 3.0, 4.0))})
    quantized = [node.input[0] for node in graph.node if node.op_type == "QuantizeLinear"]
    assert quantized == ["q", "kt", "p1", "v"]
    scales = {t.name: float(numpy_helper.to_array(t)) for t in graph.initializer if t.name.endswith(".a_s8")}
    expected = {"block1.q_fp8.a_s8": 2.0, "block1.k_fp8.a_s8": 3.0, "block1.softmax_fp8.a_s8": SOFTMAX_AMAX, "block1.v_fp8.a_s8": 4.0}
    assert scales == {name: pytest.approx(amax / 448.0, rel=1e-2) for name, amax in expected.items()}
    assert list(graph.node[0].input) == ["q", "kt"]  # block 0 keeps its precision
    onnx.checker.check_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]))
    with pytest.raises(ValueError, match="Not the Softmax output"):
        quantize_softmax_attention(graph, {"logits0": Fp8Attention("block0", (1.0, 1.0, 1.0))})
