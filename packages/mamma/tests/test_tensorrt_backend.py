"""MammaNet FP8 graph surgery: the qkv split keeps the attention inputs exact."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from mamma.landmarks.tensorrt_backend import _quantize_attention, _split_qkv_linears

TOKENS: int = 5
HEADS: int = 2
HEAD_DIM: int = 3
WIDTH: int = HEADS * HEAD_DIM


def exported_attention_input() -> onnx.ModelProto:
    """The dynamo export of ``qkv(x).reshape(b, n, 3, h, -1).permute(2, 0, 3, 1, 4)`` and its three selects."""
    rng: np.random.Generator = np.random.default_rng(0)
    initializers: list[TensorProto] = [
        numpy_helper.from_array(rng.standard_normal((3 * WIDTH, WIDTH)).astype(np.float32), "inner.model.backbone.blocks.0.attn.qkv.weight"),
        numpy_helper.from_array(rng.standard_normal(3 * WIDTH).astype(np.float16), "qkv_bias"),
        numpy_helper.from_array(np.array([1, TOKENS, 3, HEADS, -1], dtype=np.int64), "qkv_shape"),
        *(numpy_helper.from_array(np.array(part, dtype=np.int64), f"select_{part}") for part in range(3)),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node("Cast", ["inner.model.backbone.blocks.0.attn.qkv.weight"], ["w16"], to=TensorProto.FLOAT16),
        helper.make_node("Transpose", ["w16"], ["wt"], perm=[1, 0]),
        helper.make_node("MatMul", ["x", "wt"], ["mm"]),
        helper.make_node("Add", ["mm", "qkv_bias"], ["linear"]),
        helper.make_node("Reshape", ["linear", "qkv_shape"], ["view"]),
        helper.make_node("Transpose", ["view"], ["permute"], perm=[2, 0, 3, 1, 4]),
        *(helper.make_node("Gather", ["permute", f"select_{part}"], [name], axis=0) for part, name in enumerate("qkv")),
    ]
    graph = helper.make_graph(
        nodes,
        "attention_input",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, TOKENS, WIDTH])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT16, None) for name in "qkv"],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def run_reference(model: onnx.ModelProto, x: np.ndarray) -> list[np.ndarray]:
    """Evaluate the graph with onnx's reference runtime."""
    outputs = ReferenceEvaluator(model).run(None, {"x": x})
    assert isinstance(outputs, list)
    return outputs


def test_split_qkv_linears_matches_the_fused_projection() -> None:
    x: np.ndarray = np.random.default_rng(1).standard_normal((1, TOKENS, WIDTH)).astype(np.float16)
    fused: onnx.ModelProto = exported_attention_input()
    expected: list[np.ndarray] = run_reference(fused, x)

    split: onnx.ModelProto = exported_attention_input()
    assert _split_qkv_linears(split) == 1
    weights: set[str] = {tensor.name for tensor in split.graph.initializer}
    assert {f"inner.model.backbone.blocks.0.attn.{part}.weight" for part in "qkv"} <= weights
    assert not [node for node in split.graph.node if node.op_type == "Gather"]
    actual: list[np.ndarray] = run_reference(split, x)
    for got, want in zip(actual, expected, strict=True):
        assert got.shape == want.shape == (1, HEADS, TOKENS, HEAD_DIM)
        np.testing.assert_array_equal(got, want)


def exported_split_attention() -> onnx.ModelProto:
    """``softmax(q * scale @ k^T) @ v`` over the head tensors ``_split_qkv_linears`` writes, plus a decoder-style softmax."""
    heads: dict[str, str] = {part: f"inner.model.backbone.blocks.0.attn.{part}.heads" for part in "qkv"}
    nodes: list[onnx.NodeProto] = [
        *(helper.make_node("Transpose", [heads[part]], [part], perm=[0, 2, 1, 3]) for part in "qkv"),
        helper.make_node("Mul", ["q", "scale"], ["q_scaled"]),
        helper.make_node("Transpose", ["k"], ["k_t"], perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["q_scaled", "k_t"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
        helper.make_node("MatMul", ["probs", "v"], ["context"]),
        helper.make_node("Softmax", ["context"], ["decoder_probs"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes,
        "split_attention",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, TOKENS, HEADS, HEAD_DIM]) for name in heads.values()],
        [helper.make_tensor_value_info("decoder_probs", TensorProto.FLOAT16, [1, HEADS, TOKENS, HEAD_DIM])],
        [numpy_helper.from_array(np.array(HEAD_DIM**-0.5, dtype=np.float16), "scale")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def test_quantize_attention_adds_fp8_qdq_to_vit_attention_operands_only() -> None:
    model: onnx.ModelProto = exported_split_attention()
    assert _quantize_attention(model, {"backbone.blocks.0.attn": (2.0, 3.0, 4.0)}) == 1
    quantized: list[str] = [node.input[0] for node in model.graph.node if node.op_type == "QuantizeLinear"]
    assert quantized == ["q_scaled", "k_t", "probs", "v"]
    onnx.checker.check_model(model)
    rng: np.random.Generator = np.random.default_rng(2)
    feeds: dict[str, np.ndarray] = {
        tensor.name: rng.uniform(-1.0, 1.0, (1, TOKENS, HEADS, HEAD_DIM)).astype(np.float16) for tensor in model.graph.input
    }
    reference: list[np.ndarray] | dict = ReferenceEvaluator(exported_split_attention()).run(None, feeds)
    fp8: list[np.ndarray] | dict = ReferenceEvaluator(model).run(None, feeds)
    assert isinstance(reference, list) and isinstance(fp8, list)
    np.testing.assert_allclose(fp8[0].astype(np.float32), reference[0].astype(np.float32), atol=0.05)
