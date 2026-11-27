"""The FlexGlue matcher rewrite: contrib attention and Gelu become standard ops (a bf16 attention core), early stops become selects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

pytest.importorskip("trtkit", reason="colsfm.flexglue_trt imports trtkit's runtime")

from colsfm.flexglue_trt import _lower_early_stop, _rewrite  # noqa: E402

# After colsfm, whose import claims the ONNX Runtime provider bridge for pycolmap (colsfm/__init__.py).
ort = pytest.importorskip("onnxruntime", reason="the rewrite is checked against ONNX Runtime's own contrib kernels")

HIDDEN: int = 16
HEADS: int = 4
THRESHOLD: float = 0.2


def _toy_matcher(if_name: str = "early_stop_layer2_if") -> onnx.ModelProto:
    """x -> Gelu -> MultiHeadAttention(x, x, x), then the pinned graph's early-stop shape: `Less(score, threshold)` picks
    the no-match branch, the else branch matches with a branch-local weight."""
    then_branch = helper.make_graph(
        [
            helper.make_node("Shape", ["x"], ["stop_shape"], start=0, end=2),
            helper.make_node("ConstantOfShape", ["stop_shape"], ["stop_matches"], value=helper.make_tensor("v", TensorProto.INT64, [1], [-1])),
            helper.make_node("ConstantOfShape", ["stop_shape"], ["stop_scores"], value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.0])),
            helper.make_node("Identity", ["score"], ["stop_covisibility"]),
        ],
        "then",
        [],
        [
            helper.make_tensor_value_info("stop_matches", TensorProto.INT64, None),
            helper.make_tensor_value_info("stop_scores", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("stop_covisibility", TensorProto.FLOAT, None),
        ],
    )
    else_branch = helper.make_graph(
        [
            helper.make_node("Add", ["attended", "late_weight"], ["late"]),
            helper.make_node("ArgMax", ["late"], ["full_matches"], axis=2, keepdims=0),
            helper.make_node("ReduceMax", ["late", "last_axis"], ["full_scores"], keepdims=0),
            helper.make_node("Add", ["score", "late_bonus"], ["full_covisibility"]),
        ],
        "else",
        [],
        [
            helper.make_tensor_value_info("full_matches", TensorProto.INT64, None),
            helper.make_tensor_value_info("full_scores", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("full_covisibility", TensorProto.FLOAT, None),
        ],
        initializer=[
            numpy_helper.from_array(np.full((HIDDEN,), 0.25, dtype=np.float32), "late_weight"),
            numpy_helper.from_array(np.array([0.5], dtype=np.float32), "late_bonus"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "last_axis"),
        ],
    )
    graph = helper.make_graph(
        [
            helper.make_node("Gelu", ["x"], ["activated"], domain="com.microsoft"),
            helper.make_node("MultiHeadAttention", ["activated", "activated", "activated"], ["attended"], domain="com.microsoft", num_heads=HEADS),
            helper.make_node("Less", ["score", "threshold"], ["stop"]),
            helper.make_node(
                "If", ["stop"], ["matches0", "mscores0", "covisibility_score"], name=if_name, then_branch=then_branch, else_branch=else_branch
            ),
        ],
        "toy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 5, HIDDEN]), helper.make_tensor_value_info("score", TensorProto.FLOAT, [1])],
        [
            helper.make_tensor_value_info("matches0", TensorProto.INT64, [1, 5]),
            helper.make_tensor_value_info("mscores0", TensorProto.FLOAT, [1, 5]),
            helper.make_tensor_value_info("covisibility_score", TensorProto.FLOAT, [1]),
        ],
        initializer=[numpy_helper.from_array(np.array(THRESHOLD, dtype=np.float32), "threshold")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)], ir_version=10)


def _run(model: onnx.ModelProto, feeds: dict[str, np.ndarray], tmp_path: Path, name: str) -> list[np.ndarray]:
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feeds)


@pytest.mark.parametrize(
    ("score", "stops"),
    [(0.05, True), (np.nextafter(np.float32(THRESHOLD), np.float32(0.0)), True), (THRESHOLD, False), (0.9, False)],
    ids=["stops", "just-below-gate", "at-gate", "continues"],
)
def test_the_rewritten_matcher_returns_what_the_conditional_graph_returns(tmp_path: Path, score: float, stops: bool) -> None:
    """Standard-op attention and erf Gelu reproduce ONNX Runtime's contrib kernels, and the select keeps the stop decision:
    a stopped pair (down to one float below the gate) gets the no-match outputs, a pair at or above the gate the full ones."""
    feeds = {"x": np.random.default_rng(0).normal(size=(1, 5, HIDDEN)).astype(np.float32), "score": np.array([score], dtype=np.float32)}
    reference = _run(_toy_matcher(), feeds, tmp_path, "reference")
    rewritten = _toy_matcher()
    assert _rewrite(rewritten.graph) == (1, 1)
    assert _lower_early_stop(rewritten.graph) == 1
    assert {node.op_type for node in rewritten.graph.node}.isdisjoint({"If", "MultiHeadAttention", "Gelu"})
    # The attention core is bf16 for TensorRT (`_attention_nodes`); ONNX Runtime's CPU kernels have no bf16 MatMul, so
    # this run widens those casts to fp32 and checks the arithmetic around them.
    core_casts = [n for n in rewritten.graph.node if n.op_type == "Cast" and helper.get_attribute_value(n.attribute[0]) == TensorProto.BFLOAT16]
    assert len(core_casts) == 3
    for cast in core_casts:
        cast.attribute[0].i = TensorProto.FLOAT
    del rewritten.opset_import[:]
    rewritten.opset_import.append(helper.make_opsetid("", 18))
    lowered = _run(rewritten, feeds, tmp_path, "rewritten")
    assert (reference[0] == -1).all() == stops
    np.testing.assert_array_equal(lowered[0], reference[0])
    for got, expected in zip(lowered[1:], reference[1:], strict=True):
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_a_conditional_outside_the_pinned_early_stop_structure_is_refused() -> None:
    """Only `early_stop_layer2_if` may sit at the top level; anything else would be silently lowered otherwise."""
    with pytest.raises(ValueError, match="unexpected conditional 'some_other_if'"):
        _lower_early_stop(_toy_matcher("some_other_if").graph)
