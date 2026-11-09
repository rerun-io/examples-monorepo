"""Detector FP8 QDQ must preserve the FP32 box-decoding tail and public outputs."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from posekit.models.egohand_detector_quantization import quantize_detector


def detector_graph() -> onnx.ModelProto:
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["images", "onnx::Conv_1", "bias"], ["features"], name="/backbone/conv/Conv"),
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


def test_fp8_convolution_preserves_fp32_outputs_and_tail() -> None:
    source: onnx.ModelProto = detector_graph()
    model: onnx.ModelProto = quantize_detector(source, {"/backbone/conv/Conv": 1.0})
    onnx.checker.check_model(model)
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert any(t.data_type == TensorProto.FLOAT8E4M3FN for t in model.graph.initializer)
    assert any(n.op_type == "QuantizeLinear" for n in model.graph.node)
    assert all(n.op_type != "QuantizeLinear" for n in source.graph.node)
    reduce: onnx.NodeProto = next(n for n in model.graph.node if n.op_type == "ReduceMean")
    cast: onnx.NodeProto = next(n for n in model.graph.node if reduce.input[0] in n.output)
    assert cast.op_type == "Cast"
    assert next(a.i for a in cast.attribute if a.name == "to") == TensorProto.FLOAT


def test_detector_quantization_requires_every_conv_range() -> None:
    with pytest.raises(ValueError, match="range"):
        quantize_detector(detector_graph(), {})


def test_backbone_scope_keeps_stem_and_neck_in_original_precision() -> None:
    source: onnx.ModelProto = detector_graph()
    source.graph.node[0].name = "/model/backbone/res_layers.0/conv/Conv"
    source.graph.node.insert(1, helper.make_node("Conv", ["features", "neck.weight", "bias"], ["neck_features"], name="/model/encoder/conv/Conv"))
    source.graph.node[2].input[0] = "neck_features"
    source.graph.initializer.append(numpy_helper.from_array(np.full((8, 8, 1, 1), 0.1, dtype=np.float32), "neck.weight"))
    model: onnx.ModelProto = quantize_detector(source, {source.graph.node[0].name: 1.0}, scope="backbone")
    onnx.checker.check_model(model)
    assert sum(node.op_type == "QuantizeLinear" for node in model.graph.node) == 1
    neck: onnx.NodeProto = next(node for node in model.graph.node if node.name == "/model/encoder/conv/Conv")
    weight: TensorProto = next(tensor for tensor in model.graph.initializer if tensor.name == neck.input[1])
    assert weight.data_type == TensorProto.FLOAT16
    np.testing.assert_array_equal(numpy_helper.to_array(weight), np.full((8, 8, 1, 1), 0.1, dtype=np.float16))


def test_late_backbone_scope_does_not_quantize_early_features() -> None:
    source: onnx.ModelProto = detector_graph()
    source.graph.node[0].name = "/model/backbone/res_layers.0/conv/Conv"
    source.graph.node.insert(1, helper.make_node("Conv", ["features", "late.weight", "bias"], ["late_features"], name="/model/backbone/res_layers.3/conv/Conv"))
    source.graph.node[2].input[0] = "late_features"
    source.graph.initializer.append(numpy_helper.from_array(np.full((8, 8, 1, 1), 0.1, dtype=np.float32), "late.weight"))
    model: onnx.ModelProto = quantize_detector(source, {source.graph.node[1].name: 1.0}, scope="late_backbone")
    onnx.checker.check_model(model)
    assert sum(node.op_type == "QuantizeLinear" for node in model.graph.node) == 1
    early: onnx.NodeProto = next(node for node in model.graph.node if node.name == source.graph.node[0].name)
    assert any(tensor.name == early.input[1] and tensor.data_type == TensorProto.FLOAT16 for tensor in model.graph.initializer)


def test_projection_scope_quantizes_only_feature_projections() -> None:
    source: onnx.ModelProto = detector_graph()
    source.graph.node.insert(1, helper.make_node("Conv", ["features", "projection.weight", "bias"], ["projected"], name="/model/encoder/input_proj.0/conv/Conv"))
    source.graph.node[2].input[0] = "projected"
    source.graph.initializer.append(numpy_helper.from_array(np.full((8, 8, 1, 1), 0.1, dtype=np.float32), "projection.weight"))
    model: onnx.ModelProto = quantize_detector(source, {source.graph.node[1].name: 1.0}, scope="projections")
    onnx.checker.check_model(model)
    assert sum(node.op_type == "QuantizeLinear" for node in model.graph.node) == 1
    backbone: onnx.NodeProto = next(node for node in model.graph.node if node.name == source.graph.node[0].name)
    assert any(tensor.name == backbone.input[1] and tensor.data_type == TensorProto.FLOAT16 for tensor in model.graph.initializer)
