"""FlexGlue's FP8 matcher: which MatMuls go to FP8, the static scales baked into the graph, the calibration pairs, and
the refusal of a missing or stale FP8 graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

pytest.importorskip("trtkit", reason="colsfm.flexglue_fp8 uses trtkit's Q/DQ helpers")

from colsfm.flexglue_fp8 import (  # noqa: E402
    SOURCE_PROPERTY,
    calibration_graph,
    calibration_pairs,
    check_fp8_graph,
    fp8_targets,
    quantize,
)

WIDTH: int = 8
FP8_MAX: float = 448.0


def _toy_matcher() -> onnx.ModelProto:
    """One transformer linear feeding two others (shared input, as the q/k/v projections), one head linear, and one
    attention core (bf16 q.k^T, softmax, p.v) in the prepared matcher's spelling."""
    weights = {
        name: numpy_helper.from_array(np.full((WIDTH, WIDTH), value, dtype=np.float16), name)
        for name, value in (
            ("matcher.transformers.0.self_attn.Wqkv.weight", 0.5),
            ("matcher.transformers.0.self_attn.out.weight", -0.25),
            ("matcher.heads.0.proj.weight", 1.0),
        )
    }

    def linear(weight: str, activation: str, output: str) -> list[onnx.NodeProto]:
        return [
            helper.make_node("Transpose", [weight], [f"{output}_wt"], perm=[1, 0]),
            helper.make_node("MatMul", [activation, f"{output}_wt"], [output], name=f"mm_{output}"),
        ]

    nodes = [
        *linear("matcher.transformers.0.self_attn.Wqkv.weight", "x", "q"),
        *linear("matcher.transformers.0.self_attn.out.weight", "x", "k"),
        helper.make_node("Cast", ["q"], ["q_bf16"], to=TensorProto.BFLOAT16),
        helper.make_node("Cast", ["k"], ["k_bf16"], to=TensorProto.BFLOAT16),
        helper.make_node("MatMul", ["q_bf16", "k_bf16"], ["logits"], name="mm_logits"),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=-1),
        helper.make_node("MatMul", ["probabilities", "k_bf16"], ["context_bf16"], name="mm_context"),
        helper.make_node("Cast", ["context_bf16"], ["context"], to=TensorProto.FLOAT16),
        *linear("matcher.heads.0.proj.weight", "context", "y"),
    ]
    graph = helper.make_graph(
        nodes,
        "toy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4, WIDTH])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4, WIDTH])],
        initializer=list(weights.values()),
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_only_transformer_linears_and_attention_cores_go_to_fp8() -> None:
    """The head linear keeps its precision; the shared projection input and the core operands are calibrated once each,
    the softmax probabilities not at all (their range is fixed)."""
    targets = fp8_targets(_toy_matcher().graph)
    assert [node for node, _, _ in targets.linears] == ["mm_q", "mm_k"]
    assert [node for node, _, _ in targets.cores] == ["mm_logits", "mm_context"]
    assert targets.probabilities == {"probabilities"}
    assert targets.calibrated() == ["x", "q_bf16", "k_bf16"]
    extra = calibration_graph(_toy_matcher(), targets).graph.output[1:]
    assert [(v.name, v.type.tensor_type.elem_type) for v in extra] == [
        ("x", TensorProto.FLOAT16),
        ("q_bf16", TensorProto.BFLOAT16),
        ("k_bf16", TensorProto.BFLOAT16),
    ]


def _operand_scales(model: onnx.ModelProto) -> dict[tuple[str, int], tuple[float, int]]:
    """The static scale and its dtype behind every dequantized MatMul operand, by `(node, input slot)`."""
    scales = {t.name: (float(numpy_helper.to_array(t)), t.data_type) for t in model.graph.initializer if t.name.endswith(".a_s8")}
    return {
        (node.name, slot): scales[name.removesuffix(".a_dq8") + ".a_s8"]
        for node in model.graph.node
        if node.op_type == "MatMul"
        for slot, name in enumerate(node.input)
        if name.endswith(".a_dq8")
    }


def test_the_scales_baked_into_the_graph_are_the_calibrated_maxima_with_headroom() -> None:
    """Linear inputs: amax x 2 / 448 in fp16, one Q/DQ shared by both projections of x; core operands: amax x 2 / 448
    in bf16; probabilities: 1 / 448. Weights become FP8 with per-channel scales, the fp16 weights they replace are
    dropped, and the head stays as it was."""
    model = _toy_matcher()
    quantized = quantize(model, fp8_targets(model.graph), {"x": 3.0, "q_bf16": 10.0, "k_bf16": 20.0})
    ops = [node.op_type for node in quantized.graph.node]
    assert ops.count("QuantizeLinear") == 1 + 4  # x once, two operands per core
    assert ops.count("DequantizeLinear") == 1 + 4 + 2  # plus the two FP8 weights
    fp16, bf16 = TensorProto.FLOAT16, TensorProto.BFLOAT16
    expected = {
        ("mm_q", 0): (3.0 * 2 / FP8_MAX, fp16),
        ("mm_k", 0): (3.0 * 2 / FP8_MAX, fp16),
        ("mm_logits", 0): (10.0 * 2 / FP8_MAX, bf16),
        ("mm_logits", 1): (20.0 * 2 / FP8_MAX, bf16),
        ("mm_context", 0): (1 / FP8_MAX, bf16),
        ("mm_context", 1): (20.0 * 2 / FP8_MAX, bf16),
    }
    scales = _operand_scales(quantized)
    assert scales.keys() == expected.keys()
    for operand, (scale, dtype) in expected.items():
        assert scales[operand] == (pytest.approx(scale, rel=1e-2), dtype), operand
    weights = {t.name for t in quantized.graph.initializer if t.data_type == TensorProto.FLOAT8E4M3FN and t.dims}
    assert len(weights) == 2 and all("transformers" in name for name in weights)
    assert [t.name for t in quantized.graph.initializer if t.name.endswith(".weight")] == ["matcher.heads.0.proj.weight"]
    assert {"q_wt", "k_wt"}.isdisjoint(o for node in quantized.graph.node for o in node.output)
    head = next(node for node in quantized.graph.node if node.name == "mm_y")
    assert head.input[0] == "context"
    assert next(o.version for o in quantized.opset_import if o.domain == "") == 21
    assert model.graph.node[0].op_type == "Transpose" and len(model.graph.node) == 12  # the input model is untouched


@pytest.mark.parametrize("amax", [{"x": 3.0, "q_bf16": 10.0}, {"x": 3.0, "q_bf16": 10.0, "k_bf16": 0.0}], ids=["missing", "zero"])
def test_a_missing_or_zero_maximum_is_refused(amax: dict[str, float]) -> None:
    """A scale of zero (an activation the calibration never exercised) would flush every value; the rewrite stops."""
    model = _toy_matcher()
    with pytest.raises(ValueError, match="no usable maximum"):
        quantize(model, fp8_targets(model.graph), amax)


def test_calibration_pairs_mix_neighbours_ten_apart_and_unrelated_frames(tmp_path: Path) -> None:
    """In turn per pair: a neighbour, a frame ten later, a frame of another sequence; the same draw for the same seed."""
    sequences = [[tmp_path / f"{s}_{i:03d}.png" for i in range(30)] for s in ("a", "b")]
    pairs = calibration_pairs(sequences, 12)
    assert pairs == calibration_pairs(sequences, 12)
    index = {path: (path.name[0], int(path.stem[2:])) for frames in sequences for path in frames}
    for k, (first, second) in enumerate(pairs[:6]):
        (sa, ia), (sb, ib) = index[first], index[second]
        if k % 3 == 0:
            assert (sa, ib - ia) == (sb, 1)
        elif k % 3 == 1:
            assert (sa, ib - ia) == (sb, 10)
    with pytest.raises(ValueError, match="at least 12 frames"):
        calibration_pairs([sequences[0][:11]], 3)


def test_a_missing_or_stale_fp8_graph_is_refused(tmp_path: Path) -> None:
    """The FP8 graph records the prepared matcher it was calibrated from; another one (an older rewrite, other options)
    must be recalibrated, not reused."""
    prepared = tmp_path / "matcher_v3.onnx"
    fp8 = tmp_path / "flexglue_matcher_fp8.onnx"
    with pytest.raises(FileNotFoundError, match="calibrate it once"):
        check_fp8_graph(fp8, prepared)
    model = _toy_matcher()
    helper.set_model_props(model, {SOURCE_PROPERTY: "matcher_v2.onnx"})
    onnx.save(model, fp8)
    with pytest.raises(FileNotFoundError, match="recalibrate"):
        check_fp8_graph(fp8, prepared)
    helper.set_model_props(model, {SOURCE_PROPERTY: prepared.name})
    onnx.save(model, fp8)
    check_fp8_graph(fp8, prepared)
