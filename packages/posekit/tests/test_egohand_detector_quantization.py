"""Detector FP8 QDQ: only the encoder input projections, with the FP32 box-decoding tail and public outputs preserved."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from posekit.models.egohand_detector_quantization import quantize_detector

PROJECTION: str = "/model/encoder/input_proj.0/conv/Conv"


def detector_graph() -> onnx.ModelProto:
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["images", "onnx::Conv_1", "bias"], ["features"], name="/model/encoder/input_proj.0/conv/Conv"),
                helper.make_node("ReduceMean", ["features"], ["boxes"], axes=[2, 3], keepdims=0),
            ],
            "detector",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
            [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 8])],
            initializer=[
                numpy_helper.from_array(np.full((8, 3, 3, 3), 0.2, dtype=np.float32), "onnx::Conv_1"),
                numpy_helper.from_array(np.zeros(8, dtype=np.float32), "bias"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 16)],
        ir_version=10,
    )


def test_fp8_projection_preserves_fp32_outputs_and_tail() -> None:
    source: onnx.ModelProto = detector_graph()
    model: onnx.ModelProto = quantize_detector(source, {PROJECTION: 1.0})
    onnx.checker.check_model(model)
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert any(t.data_type == TensorProto.FLOAT8E4M3FN for t in model.graph.initializer)
    assert any(n.op_type == "QuantizeLinear" for n in model.graph.node)
    assert all(n.op_type != "QuantizeLinear" for n in source.graph.node)
    reduce: onnx.NodeProto = next(n for n in model.graph.node if n.op_type == "ReduceMean")
    cast: onnx.NodeProto = next(n for n in model.graph.node if reduce.input[0] in n.output)
    assert cast.op_type == "Cast"
    assert next(a.i for a in cast.attribute if a.name == "to") == TensorProto.FLOAT


def test_detector_quantization_requires_every_projection_range() -> None:
    with pytest.raises(ValueError, match="range"):
        quantize_detector(detector_graph(), {})


def test_only_projections_are_quantized() -> None:
    source: onnx.ModelProto = detector_graph()
    source.graph.node.insert(0, helper.make_node("Conv", ["raw", "stem.weight", "bias"], ["images"], name="/model/backbone/conv1/Conv"))
    source.graph.input[0].name = "raw"
    source.graph.initializer.append(numpy_helper.from_array(np.full((3, 3, 1, 1), 0.1, dtype=np.float32), "stem.weight"))
    model: onnx.ModelProto = quantize_detector(source, {PROJECTION: 1.0})
    onnx.checker.check_model(model)
    assert sum(node.op_type == "QuantizeLinear" for node in model.graph.node) == 1
    stem: onnx.NodeProto = next(node for node in model.graph.node if node.name == "/model/backbone/conv1/Conv")
    weight: TensorProto = next(tensor for tensor in model.graph.initializer if tensor.name == stem.input[1])
    assert weight.data_type == TensorProto.FLOAT16
