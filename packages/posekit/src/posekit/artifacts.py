"""ONNX artifact acquisition and caching.

ONNX files are the portable interchange for the accelerated backends: the ONNX
Runtime backend loads them directly and the TensorRT backend builds machine-local
engines from them. Artifacts come either from OpenMMLab deploy zips (the rtmlib
convention for RTMPose/RTMW/YOLOX) or from a torch export done by a model family.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_ONNX_CACHE_DIR: Path = Path(os.environ.get("POSEKIT_ONNX_CACHE", "~/.cache/posekit/onnx")).expanduser()
"""Portable ONNX artifact cache; override with the ``POSEKIT_ONNX_CACHE`` env var."""


def fetch_openmmlab_onnx(zip_url: str, *, cache_dir: Path = DEFAULT_ONNX_CACHE_DIR) -> Path:
    """Download an OpenMMLab SDK deploy zip and return its cached ``end2end.onnx``.

    Args:
        zip_url: ``download.openmmlab.com`` deploy-zip URL (contains ``end2end.onnx``).
        cache_dir: ONNX artifact cache root.

    Returns:
        Path to the extracted ONNX file, named after the zip stem.

    Raises:
        RuntimeError: If the zip does not contain an ONNX file.
    """
    stem: str = Path(zip_url).stem
    onnx_path: Path = cache_dir / f"{stem}.onnx"
    if onnx_path.exists():
        return onnx_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path: Path = cache_dir / f"{stem}.zip"
    if not zip_path.exists():
        print(f"[posekit] downloading ONNX artifact: {zip_url}")
        tmp_path: Path = zip_path.with_suffix(".zip.part")
        urllib.request.urlretrieve(zip_url, tmp_path)  # noqa: S310
        tmp_path.rename(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        onnx_members: list[str] = [name for name in archive.namelist() if name.endswith(".onnx")]
        if not onnx_members:
            raise RuntimeError(f"No .onnx file inside {zip_url}.")
        with archive.open(onnx_members[0]) as src:
            onnx_path.write_bytes(src.read())
    zip_path.unlink()
    return onnx_path


def strip_detector_nms(onnx_path: Path) -> Path:
    """Rewrite a detector ONNX so it ends at the NMS inputs (boxes + scores).

    MMDeploy SDK detector exports bake ``TopK`` + ``NonMaxSuppression`` into the
    graph, which forces batch-1 data-dependent output shapes, blocks static
    TensorRT engines, and can exceed TensorRT's TopK limit (K <= 3840). posekit
    instead ends the graph at the full decoded per-anchor boxes/scores (before
    the TopK selection cluster) and runs thresholding + torchvision NMS on GPU,
    so ONNX Runtime and TensorRT share one static-shape artifact and one
    postprocess.

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
    kept_nodes: list[Any] = []
    while frontier:
        name: str = frontier.pop()
        node: Any = producers.get(name)
        if node is None or id(node) in needed:
            continue
        needed.add(id(node))
        kept_nodes.append(node)
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
    feeding that cluster is what posekit wants — thresholding downstream makes
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
