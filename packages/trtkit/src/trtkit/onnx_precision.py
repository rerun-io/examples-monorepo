"""Mixed-precision rewrites of fp32 ONNX graphs for strongly typed TensorRT builds.

TensorRT 11 builds only strongly typed networks: every tensor runs in the type the
graph declares, so an fp32 third-party graph (the OpenMMLab deploy zips) runs in
fp32/TF32 until the graph itself says fp16.
"""

from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FP32_ONLY_INPUTS: dict[str, frozenset[int]] = {"Resize": frozenset({1, 2})}
"""Float inputs ONNX types independently of the data: Resize's roi and scales (scales is tensor(float))."""


def fp16_conv_trunk(source: onnx.ModelProto) -> onnx.ModelProto:
    """Run the convolutional trunk in fp16 and keep the tail and the graph I/O in fp32.

    A node joins the trunk when it is a Conv or feeds one; everything after the
    last Conv (norms, attention, SimCC heads, box decode) stays fp32, which is
    where fp16 overflows (sums of squares) and loses resolution (pixel
    coordinates). Weights read by the trunk become fp16, keeping an fp32 copy
    when the tail reads them too; activations crossing the boundary get a Cast.

    Args:
        source: fp32 graph; left unmodified.

    Returns:
        The rewritten, shape-inferred, checked graph.

    Raises:
        ValueError: If a graph output comes from the trunk (no fp32 tail).
    """
    model: onnx.ModelProto = onnx.shape_inference.infer_shapes(source)
    graph = model.graph
    nodes: list[onnx.NodeProto] = list(graph.node)
    constants: dict[str, TensorProto] = {tensor.name: tensor for tensor in graph.initializer}
    constants.update({node.output[0]: next(a.t for a in node.attribute if a.name == "value") for node in nodes if node.op_type == "Constant"})
    elem_type: dict[str, int] = {v.name: v.type.tensor_type.elem_type for v in [*graph.value_info, *graph.input, *graph.output]}
    elem_type.update({name: tensor.data_type for name, tensor in constants.items()})

    producer: dict[str, int] = {name: index for index, node in enumerate(nodes) for name in node.output}
    trunk: set[int] = set()
    stack: list[int] = [index for index, node in enumerate(nodes) if node.op_type == "Conv"]
    while stack:
        index = stack.pop()
        if index not in trunk:
            trunk.add(index)
            stack.extend(producer[name] for name in nodes[index].input if name in producer)
    trunk = {index for index in trunk if nodes[index].op_type != "Constant"}  # weights, not compute
    if any(producer.get(output.name) in trunk for output in graph.output):
        raise ValueError("A graph output comes from the fp16 trunk; fp16_conv_trunk needs an fp32 tail after the last Conv")

    first_reader: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            first_reader.setdefault(name, index)
    renamed: dict[tuple[str, int], str] = {}
    half_weights: list[TensorProto] = []
    casts: dict[int, list[onnx.NodeProto]] = defaultdict(list)  # node index -> casts inserted before it
    for index, node in enumerate(nodes):
        for slot, name in enumerate(node.input):
            if not name or elem_type.get(name) != TensorProto.FLOAT or slot in FP32_ONLY_INPUTS.get(node.op_type, frozenset()):
                continue
            to: int = TensorProto.FLOAT16 if index in trunk else TensorProto.FLOAT
            if (producer.get(name) in trunk) == (to == TensorProto.FLOAT16):
                continue
            if (name, to) not in renamed:
                target = f"{name}__{'fp16' if to == TensorProto.FLOAT16 else 'fp32'}"
                if name in constants and to == TensorProto.FLOAT16:
                    half_weights.append(numpy_helper.from_array(numpy_helper.to_array(constants[name]).astype(np.float16), name=target))
                else:
                    casts[first_reader[name]].append(helper.make_node("Cast", [name], [target], to=to, name=f"Cast_{target}"))
                renamed[(name, to)] = target
            node.input[slot] = renamed[(name, to)]

    del graph.node[:]
    graph.node.extend(cast for index, node in enumerate(nodes) for cast in [*casts[index], node])
    graph.initializer.extend(half_weights)
    read: set[str] = {name for node in graph.node for name in node.input}
    kept: list[TensorProto] = [tensor for tensor in graph.initializer if tensor.name in read]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model
