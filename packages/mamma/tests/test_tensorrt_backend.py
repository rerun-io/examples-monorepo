"""MammaNet FP8 graph surgery: the qkv split keeps the attention inputs exact."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from mamma.landmarks.tensorrt_backend import _split_qkv_linears

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
