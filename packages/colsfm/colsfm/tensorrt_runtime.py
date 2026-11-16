"""The TensorRT plumbing the RaCo extractor and the LightGlue+ matcher share.

Engines are built by trtkit (`trtkit.ensure_engine`): TensorRT 11 builds strongly typed networks only, so the
precision lives in the graph. `resolve_engine` hands trtkit the fp32 graph (TF32 allowed), or optionally its fp16
conv-trunk rewrite (`trtkit.onnx_precision.fp16_trunk`); trtkit keys the cached engine by graph content, profiles,
builder settings, TensorRT version and GPU, and builds it once per machine.

Execution is trtkit's `TensorRtDynamicRuntime` with `direct_binding=True` for every engine (RaCo-ALIKED, LightGlue+,
FlexGlue's own two use its default copying mode): device inputs bound in place, outputs sized per call, views valid
until the next call. colsfm converts to host only where a value goes into the COLMAP database.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, TypeAlias

from trtkit import onnx_static_batch_size
from trtkit.onnx_precision import fp16_trunk
from trtkit.trt_builder import DEFAULT_TRT_CACHE_DIR, InputShapeProfile, TrtBuildConfig, ensure_engine

ShapeProfile: TypeAlias = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
"""One optimisation profile for one input: its minimum, optimal and maximum shape."""

EnginePrecision: TypeAlias = Literal["fp16-trunk", "fp32"]
"""How a graph is typed before the strongly typed build: its conv trunk in fp16, or fp32 (TF32) throughout."""

DEFAULT_WORKSPACE_GIB: Final[int] = 24
"""TensorRT tactic workspace ceiling (a cap, not an allocation), trtkit's default. 8 GiB, the fork's value, leaves TensorRT 11.2
with no tactic for the shape-dynamic RaCo graph's fused convolution block ("Could not find any implementation")."""

DEFAULT_OPTIMIZATION_LEVEL: Final[int] = 3
"""TensorRT builder search level; 3 is the default and what the shipped engines used."""

FP16_GRAPH_DIR: Final[Path] = DEFAULT_TRT_CACHE_DIR.parent / "onnx-fp16"
"""Where the fp16-trunk rewrites of the ONNX graphs are cached."""


def resolve_engine(
    onnx_path: Path,
    profiles: Mapping[str, ShapeProfile] | None = None,
    *,
    precision: EnginePrecision = "fp32",
    workspace_gib: int = DEFAULT_WORKSPACE_GIB,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
) -> Path:
    """Return trtkit's cached engine for one graph and one set of build settings, building it on a miss.

    Args:
        onnx_path: The fp32 ONNX graph.
        profiles: Optimisation profile per dynamic input, as `(minimum, optimal, maximum)` shapes. Without one the
            graph must be fully static.
        precision: `fp32` keeps the graph as it is (TF32 allowed), the default; `fp16-trunk` runs the convolutional trunk
            in fp16 through trtkit's generic rewrite, which RaCo-ALIKED's graph does not survive yet (a Float/Half `Add`
            after an `einsum` fails to parse under TensorRT 11.2).
        workspace_gib: Tactic workspace ceiling in GiB.
        optimization_level: TensorRT builder search level, 0 to 5.

    Returns:
        The engine file.

    Raises:
        FileNotFoundError: When the ONNX graph is missing.
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"No ONNX graph at {onnx_path}")
    graph: Path = fp16_trunk(onnx_path, FP16_GRAPH_DIR) if precision == "fp16-trunk" else onnx_path
    shape_profiles: tuple[InputShapeProfile, ...] = tuple(
        InputShapeProfile(name=name, min_shape=minimum, opt_shape=optimal, max_shape=maximum)
        for name, (minimum, optimal, maximum) in (profiles or {}).items()
    )
    static_batch: int | None = None if shape_profiles else onnx_static_batch_size(graph)
    config: TrtBuildConfig = TrtBuildConfig(
        max_batch_size=static_batch or 1,
        opt_batch_size=static_batch or 1,
        workspace_gib=float(workspace_gib),
        builder_optimization_level=optimization_level,
        shape_profiles=shape_profiles,
    )
    return ensure_engine(graph, config)
