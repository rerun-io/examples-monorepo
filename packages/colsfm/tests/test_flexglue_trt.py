"""The FlexGlue matcher rewrite: contrib attention and Gelu become standard ops, early-stop branches are inlined."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

ort = pytest.importorskip("onnxruntime", reason="the rewrite is checked against ONNX Runtime's own contrib kernels")
pytest.importorskip("trtkit", reason="colsfm.flexglue_trt imports trtkit's runtime")

from colsfm.flexglue_trt import _flatten_early_stop, _rewrite  # noqa: E402

HIDDEN: int = 16
HEADS: int = 4


def _toy_matcher() -> onnx.ModelProto:
    """x -> Gelu -> MultiHeadAttention(x, x, x), then an early-stop If whose else branch adds a branch-local weight."""
    then_branch = helper.make_graph(
        [helper.make_node("Identity", ["attended"], ["stopped"])], "then", [], [helper.make_tensor_value_info("stopped", TensorProto.FLOAT, None)]
    )
    else_branch = helper.make_graph(
        [helper.make_node("Add", ["attended", "late_weight"], ["continued"])],
        "else",
        [],
        [helper.make_tensor_value_info("continued", TensorProto.FLOAT, None)],
        initializer=[numpy_helper.from_array(np.full((HIDDEN,), 0.25, dtype=np.float32), "late_weight")],
    )
    graph = helper.make_graph(
        [
            helper.make_node("Gelu", ["x"], ["activated"], domain="com.microsoft"),
            helper.make_node("MultiHeadAttention", ["activated", "activated", "activated"], ["attended"], domain="com.microsoft", num_heads=HEADS),
            helper.make_node("If", ["stop"], ["y"], then_branch=then_branch, else_branch=else_branch),
        ],
        "toy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 5, HIDDEN]), helper.make_tensor_value_info("stop", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 5, HIDDEN])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)], ir_version=10)


def _run(model: onnx.ModelProto, feeds: dict[str, np.ndarray], tmp_path: Path, name: str) -> np.ndarray:
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {k: v for k, v in feeds.items() if k in {i.name for i in session.get_inputs()}})[0]


def test_the_rewritten_matcher_computes_what_the_contrib_kernels_compute(tmp_path: Path) -> None:
    """Standard-op attention and erf Gelu reproduce ONNX Runtime's contrib kernels; the inlined graph takes the else branch."""
    x = np.random.default_rng(0).normal(size=(1, 5, HIDDEN)).astype(np.float32)
    reference = _run(_toy_matcher(), {"x": x, "stop": np.array(False)}, tmp_path, "reference")
    rewritten = _toy_matcher()
    assert _rewrite(rewritten.graph) == (1, 1)
    assert _flatten_early_stop(rewritten.graph) == 1
    assert {node.op_type for node in rewritten.graph.node}.isdisjoint({"If", "MultiHeadAttention", "Gelu"})
    del rewritten.opset_import[:]
    rewritten.opset_import.append(helper.make_opsetid("", 18))
    np.testing.assert_allclose(_run(rewritten, {"x": x, "stop": np.array(False)}, tmp_path, "rewritten"), reference, rtol=1e-5, atol=1e-5)
