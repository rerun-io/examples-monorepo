"""Generic ONNX graph introspection shared by the accelerated backends.

Model-agnostic reads on ONNX interchange files. Model-family-specific graph
surgery (e.g. posekit's detector NMS stripping) lives in the model packages.
"""

from pathlib import Path
from typing import Any


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
