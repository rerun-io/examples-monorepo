"""Mixed-precision rewrites of fp32 ONNX graphs for strongly typed TensorRT builds.

TensorRT 11 builds only strongly typed networks: every tensor runs in the type the
graph declares, so an fp32 third-party graph (the OpenMMLab deploy zips) runs in
fp32/TF32 until the graph itself says fp16.
"""

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from trtkit.artifacts import build_once
from trtkit.trt_builder import onnx_content_hash

FP32_ONLY_INPUTS: dict[str, frozenset[int]] = {"Resize": frozenset({1, 2})}
"""Float inputs ONNX types independently of the data: Resize's roi and scales (scales is tensor(float))."""


def _constant_tensor(node: onnx.NodeProto) -> TensorProto:
    """A Constant node's value as a tensor, whichever attribute holds it (value, value_float(s), value_int(s)).

    torch.onnx writes scalars as `value_float` / `value_ints`; reading only `value` raised StopIteration on those.
    """
    attribute: onnx.AttributeProto = node.attribute[0]
    if attribute.name == "value":
        return attribute.t
    data = np.asarray(helper.get_attribute_value(attribute), dtype=np.float32 if attribute.name.startswith("value_float") else np.int64)
    return numpy_helper.from_array(data, name=node.output[0])


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
    constants.update({node.output[0]: _constant_tensor(node) for node in nodes if node.op_type == "Constant"})
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


def fp32_accumulate(model: onnx.ModelProto, select: Callable[[onnx.NodeProto, TensorProto | None], bool]) -> int:
    """Make the selected fp16 MatMul / Gemm nodes accumulate in fp32, in place.

    A strongly typed TensorRT build runs an fp16 matrix product with fp16 accumulation, which drifts at large inner
    dimensions (one K=4096 MatMul: relative error 2.4e-3 against cuBLAS's 2.1e-4). Each selected node reads its fp16
    operands through a Cast to fp32 placed directly before it and writes its fp32 result through a Cast back to fp16,
    so the graph outside it keeps its types. TensorRT folds the Casts into an fp16 tensor-core kernel with fp32
    accumulation; build with ``TrtBuildConfig(allow_tf32=False)`` so it does not pick a TF32 kernel instead.

    Args:
        model: Graph to rewrite; its weights may stay in external data (``load_external_data=False``).
        select: Called with each fp16 MatMul / Gemm and the initializer its second operand reads (through Cast and
            Transpose nodes), or ``None`` when that operand is an activation; True rewrites the node.

    Returns:
        The number of rewritten nodes.

    Raises:
        ValueError: If ``select`` matches no fp16 MatMul / Gemm.
    """
    graph = model.graph
    inferred = onnx.shape_inference.infer_shapes(model).graph
    elem_type: dict[str, int] = {v.name: v.type.tensor_type.elem_type for v in [*inferred.value_info, *inferred.input, *inferred.output]}
    elem_type.update({tensor.name: tensor.data_type for tensor in graph.initializer})
    elem_type.update({node.output[0]: _constant_tensor(node).data_type for node in graph.node if node.op_type == "Constant"})
    initializers: dict[str, TensorProto] = {tensor.name: tensor for tensor in graph.initializer}
    producer: dict[str, onnx.NodeProto] = {name: node for node in graph.node for name in node.output}

    def weight(name: str) -> TensorProto | None:
        while name in producer and producer[name].op_type in ("Cast", "Transpose"):
            name = producer[name].input[0]
        return initializers.get(name)

    nodes: list[onnx.NodeProto] = []
    widened: dict[str, str] = {}
    rewritten: int = 0
    for node in graph.node:
        if node.op_type not in ("MatMul", "Gemm") or elem_type.get(node.output[0]) != TensorProto.FLOAT16 or not select(node, weight(node.input[1])):
            nodes.append(node)
            continue
        for slot, name in enumerate(node.input):
            if not name or elem_type.get(name) != TensorProto.FLOAT16:
                continue
            if name not in widened:
                widened[name] = f"{name}__fp32"
                nodes.append(helper.make_node("Cast", [name], [widened[name]], to=TensorProto.FLOAT, name=f"Cast_{name}__fp32"))
            node.input[slot] = widened[name]
        output: str = node.output[0]
        node.output[0] = f"{output}__fp32acc"
        nodes.append(node)
        nodes.append(helper.make_node("Cast", [node.output[0]], [output], to=TensorProto.FLOAT16, name=f"Cast_{output}__fp16"))
        rewritten += 1
    if rewritten == 0:
        raise ValueError("fp32_accumulate: the selection matched no fp16 MatMul / Gemm")
    del graph.node[:]
    graph.node.extend(nodes)
    return rewritten


def fp16_trunk(source: Path, cache_dir: Path) -> Path:
    """Cache the fp16 convolutional rewrite by source content and transform version; preserve the source."""
    directory = cache_dir.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    digest: str = onnx_content_hash(source, cache_dir=directory)[:16]
    dest: Path = directory / f"{source.stem}_{digest}_fp16trunk_v1.onnx"

    def rewrite() -> None:
        with TemporaryDirectory(prefix=".fp16-trunk-", dir=directory) as work:
            staged: Path = Path(work) / dest.name
            onnx.save(fp16_conv_trunk(onnx.load(source)), str(staged))
            staged.replace(dest)

    return build_once(dest, rewrite)

