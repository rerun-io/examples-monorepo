"""fp32_accumulate widens exactly the selected fp16 matrix products and keeps the graph's types and values."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from trtkit.onnx_precision import fp32_accumulate

F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


def fp16_graph() -> onnx.ModelProto:
    """fp32 I/O around an fp16 body: a linear with a wide weight (exported Cast + Transpose chain), a narrow linear and
    an activation-by-activation product."""
    rng = np.random.default_rng(0)
    initializers: list[TensorProto] = [
        numpy_helper.from_array(rng.standard_normal((32, 8)).astype(np.float32), "wide.weight"),
        numpy_helper.from_array(rng.standard_normal((8, 32)).astype(np.float32), "narrow.weight"),
    ]
    nodes: list[onnx.NodeProto] = [helper.make_node("Cast", ["x"], ["xh"], to=F16)]
    for name, source in (("wide", "xh"), ("narrow", "wide.out")):
        nodes += [
            helper.make_node("Cast", [f"{name}.weight"], [f"{name}.wh"], to=F16),
            helper.make_node("Transpose", [f"{name}.wh"], [f"{name}.wt"], perm=[1, 0]),
            helper.make_node("MatMul", [source, f"{name}.wt"], [f"{name}.out"], name=f"{name}_matmul"),
        ]
    nodes += [
        helper.make_node("Transpose", ["narrow.out"], ["narrow.t"], perm=[1, 0]),
        helper.make_node("MatMul", ["narrow.out", "narrow.t"], ["gram"], name="gram_matmul"),
        helper.make_node("Cast", ["gram"], ["y"], to=F32),
    ]
    graph = helper.make_graph(nodes, "fp16", [helper.make_tensor_value_info("x", F32, [4, 8])], [helper.make_tensor_value_info("y", F32, [4, 4])], initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_selected_products_read_fp32_casts_and_write_back_fp16() -> None:
    model = fp16_graph()
    seen: dict[str, tuple[int, ...] | None] = {}

    def wide(node: onnx.NodeProto, weight: TensorProto | None) -> bool:
        seen[node.name] = None if weight is None else tuple(weight.dims)
        return weight is not None and tuple(weight.dims) == (32, 8)

    assert fp32_accumulate(model, wide) == 1
    assert seen == {"wide_matmul": (32, 8), "narrow_matmul": (8, 32), "gram_matmul": None}
    onnx.checker.check_model(model, full_check=True)
    nodes = {node.name: node for node in model.graph.node}
    producer = {output: node for node in model.graph.node for output in node.output}
    matmul = nodes["wide_matmul"]
    assert [(producer[name].op_type, producer[name].attribute[0].i) for name in matmul.input] == [("Cast", F32), ("Cast", F32)]
    back = producer["wide.out"]
    assert back.op_type == "Cast" and back.input[0] == matmul.output[0] and back.attribute[0].i == F16
    assert list(nodes["narrow_matmul"].input) == ["wide.out", "narrow.wt"]
    assert list(nodes["gram_matmul"].input) == ["narrow.out", "narrow.t"]
    inferred = onnx.shape_inference.infer_shapes(model).graph
    types = {v.name: v.type.tensor_type.elem_type for v in inferred.value_info}
    assert types["wide.out"] == F16 and types[matmul.output[0]] == F32


def test_rewritten_graph_computes_the_fp32_accumulated_product() -> None:
    """The reference evaluator gives the fp16 rounding of the fp32 product of the fp16 operands."""
    x = np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32)
    model = fp16_graph()
    assert fp32_accumulate(model, lambda node, weight: True) == 3
    weights = {tensor.name: numpy_helper.to_array(tensor).astype(np.float16).astype(np.float32) for tensor in model.graph.initializer}
    wide = (x.astype(np.float16).astype(np.float32) @ weights["wide.weight"].T).astype(np.float16).astype(np.float32)
    narrow = (wide @ weights["narrow.weight"].T).astype(np.float16).astype(np.float32)
    expected = (narrow @ narrow.T).astype(np.float16).astype(np.float32)
    (got,) = ReferenceEvaluator(model).run(None, {"x": x})
    np.testing.assert_array_equal(got, expected)


def test_a_selection_without_matches_is_an_error() -> None:
    with pytest.raises(ValueError, match="matched no fp16"):
        fp32_accumulate(fp16_graph(), lambda node, weight: False)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tensorrt_engine_error_drops_with_fp32_accumulation(tmp_path: Path) -> None:
    """One K=4096 fp16 MatMul: the strongly typed engine accumulates in fp16 unless the graph says fp32."""
    from trtkit import TensorRtRuntime, TrtBuildConfig, ensure_engine

    rows, inner, cols = 2048, 4096, 1024
    generator = torch.Generator().manual_seed(0)
    weight = (torch.randn(inner, cols, generator=generator) / inner**0.5).half()
    nodes = [helper.make_node("Cast", ["x"], ["xh"], to=F16), helper.make_node("MatMul", ["xh", "w"], ["yh"]), helper.make_node("Cast", ["yh"], ["y"], to=F32)]
    graph = helper.make_graph(
        nodes,
        "matmul",
        [helper.make_tensor_value_info("x", F32, [1, rows, inner])],
        [helper.make_tensor_value_info("y", F32, [1, rows, cols])],
        [numpy_helper.from_array(weight.numpy(), "w")],
    )
    x = (torch.randn(1, rows, inner, generator=generator) * 2).half().float().cuda()
    truth = x.double() @ weight.cuda().double()
    errors: dict[str, float] = {}
    for name, widen in (("fp16", False), ("fp32acc", True)):
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        if widen:
            fp32_accumulate(model, lambda node, weight: True)
        path = tmp_path / f"{name}.onnx"
        onnx.save(model, str(path))
        config = TrtBuildConfig(max_batch_size=1, opt_batch_size=1, workspace_gib=1.0, allow_tf32=not widen)
        got = TensorRtRuntime(ensure_engine(path, config, cache_dir=tmp_path))({"x": x})["y"].double()
        errors[name] = ((got - truth).norm() / truth.norm()).item()
    rounding = ((truth.half().double() - truth).norm() / truth.norm()).item()
    assert errors["fp32acc"] < 1.5 * rounding < errors["fp16"], (errors, rounding)


def test_export_onnx_applies_the_rewrite_before_publishing(tmp_path: Path) -> None:
    from trtkit import export_onnx

    def fake_export_fn(model: object, inputs: object, path: str, **kwargs: object) -> None:
        onnx.save(fp16_graph(), path)

    out_path = tmp_path / "model.onnx"
    export_onnx(
        torch.nn.Identity(), (torch.zeros(1),), out_path, input_names=["x"], output_names=["y"], export_fn=fake_export_fn,
        rewrite=lambda model: fp32_accumulate(model, lambda node, weight: node.name == "gram_matmul"),
    )
    assert "gram__fp32acc" in {o for n in onnx.load(out_path).graph.node for o in n.output}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.onnx"]
