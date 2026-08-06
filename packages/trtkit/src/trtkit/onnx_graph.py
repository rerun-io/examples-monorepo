"""Generic ONNX graph surgery shared by the accelerated backends.

Model-agnostic transforms on ONNX interchange files: reading static batch
sizes and rewriting detector graphs to end before their baked-in NMS. Artifact
*acquisition* (zoo downloads, checkpoint resolution) stays model-side.
"""

from pathlib import Path
from typing import Any


def strip_detector_nms(onnx_path: Path) -> Path:
    """Rewrite a detector ONNX so it ends at the NMS inputs (boxes + scores).

    MMDeploy SDK detector exports bake ``TopK`` + ``NonMaxSuppression`` into the
    graph, which forces batch-1 data-dependent output shapes, blocks static
    TensorRT engines, and can exceed TensorRT's TopK limit (K <= 3840). trtkit
    instead ends the graph at the full decoded per-anchor boxes/scores (before
    the TopK selection cluster) and leaves thresholding + torchvision NMS to the
    caller on GPU, so ONNX Runtime and TensorRT share one static-shape artifact
    and one postprocess.

    Args:
        onnx_path: Detector ONNX containing a ``NonMaxSuppression`` node.

    Returns:
        Path to the cached ``<stem>_nonms.onnx`` whose outputs are the decoded
        boxes and scores tensors (in that order).

    Raises:
        RuntimeError: If the graph contains no ``NonMaxSuppression`` node.
    """
    import onnx
    from onnx import shape_inference

    stripped_path: Path = onnx_path.with_name(f"{onnx_path.stem}_nonms.onnx")
    if stripped_path.exists():
        return stripped_path
    model: Any = onnx.load(str(onnx_path))
    nms_nodes: list[Any] = [node for node in model.graph.node if node.op_type == "NonMaxSuppression"]
    if not nms_nodes:
        raise RuntimeError(f"No NonMaxSuppression node found in {onnx_path}.")
    node_producers: dict[str, Any] = {output: node for node in model.graph.node for output in node.output}
    boxes_name: str = _bypass_topk_selection(str(nms_nodes[0].input[0]), node_producers)
    scores_name: str = _bypass_topk_selection(str(nms_nodes[0].input[1]), node_producers)
    inferred: Any = shape_inference.infer_shapes(model)
    value_infos: dict[str, Any] = {info.name: info for info in [*inferred.graph.value_info, *inferred.graph.output, *inferred.graph.input]}
    for name in (boxes_name, scores_name):
        if name not in value_infos:
            value_infos[name] = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
    # Keep only nodes the two new outputs (transitively) depend on.
    producers: dict[str, Any] = {output: node for node in model.graph.node for output in node.output}
    needed: set[int] = set()
    frontier: list[str] = [boxes_name, scores_name]
    while frontier:
        name: str = frontier.pop()
        node: Any = producers.get(name)
        if node is None or id(node) in needed:
            continue
        needed.add(id(node))
        frontier.extend(node.input)
    kept_order: list[Any] = [node for node in model.graph.node if id(node) in needed]
    del model.graph.node[:]
    model.graph.node.extend(kept_order)
    del model.graph.output[:]
    model.graph.output.extend([value_infos[boxes_name], value_infos[scores_name]])
    onnx.save(model, str(stripped_path))
    return stripped_path


def _bypass_topk_selection(name: str, node_producers: dict[str, Any]) -> str:
    """Walk upstream past a TopK candidate-selection cluster if present.

    MMDeploy detector graphs select top-K anchors (``TopK`` -> ``Gather``,
    possibly with layout ops) right before NMS. The full per-anchor tensor
    feeding that cluster is what trtkit wants — thresholding downstream makes
    the K-selection redundant.

    Args:
        name: Tensor name feeding NMS (boxes or scores).
        node_producers: Map from tensor name to its producing node.

    Returns:
        The upstream full-anchor tensor name, or ``name`` unchanged when no
        TopK selection cluster is found.
    """
    current: str = name
    node: Any = node_producers.get(current)
    while node is not None and node.op_type in ("Transpose", "Unsqueeze", "Squeeze"):
        current = str(node.input[0])
        node = node_producers.get(current)
    if node is not None and node.op_type in ("Gather", "GatherElements") and len(node.input) > 1:
        indices_node: Any = node_producers.get(str(node.input[1]))
        for _ in range(4):
            if indices_node is None:
                break
            if indices_node.op_type == "TopK":
                return str(node.input[0])
            indices_node = node_producers.get(str(indices_node.input[0])) if len(indices_node.input) > 0 else None
    return name


def onnx_static_batch_size(onnx_path: Path) -> int | None:
    """Read the static batch size baked into an ONNX graph's inputs, if any.

    Args:
        onnx_path: ONNX model file.

    Returns:
        The fixed leading dimension shared by the graph inputs, or ``None`` when
        the batch dimension is symbolic/dynamic.
    """
    import onnx

    model: Any = onnx.load(str(onnx_path), load_external_data=False)
    initializers: set[str] = {init.name for init in model.graph.initializer}
    for graph_input in model.graph.input:
        if graph_input.name in initializers:
            continue
        dims: Any = graph_input.type.tensor_type.shape.dim
        if len(dims) > 0 and dims[0].HasField("dim_value"):
            return int(dims[0].dim_value)
    return None
