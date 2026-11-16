"""The TensorRT plumbing the RaCo extractor and the LightGlue+ matcher share.

Engines are built by trtkit (`trtkit.ensure_engine`): TensorRT 11 builds strongly typed networks only, so the
precision lives in the graph. `resolve_engine` hands trtkit the fp32 graph (TF32 allowed), or optionally its fp16
conv-trunk rewrite (`trtkit.onnx_precision.fp16_trunk`); trtkit keys the cached engine by graph content, profiles,
builder settings, TensorRT version and GPU, and builds it once per machine.

This module keeps the execution side: one generic `TensorRTSession` over `cuda.bindings` with growing device buffers
(LightGlue has four inputs and two outputs, taken by name), which also binds a `DeviceTensor` produced on its stream.

TensorRT and `cuda.bindings` are imported at module scope, so importing this module is itself the availability probe:
`colsfm.features` and `colsfm.matching` import their engine backends inside the dispatch branch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Self, TypeAlias

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from numpy import ndarray
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

PREPROCESS_IMAGE_U8_BINDING: Final[str] = "image_u8"
"""The preprocessing engine's input: `[batch, height, width, 3]` uint8 BGR, as `cv2.imread` gives it."""

PREPROCESS_IMAGE_BINDING: Final[str] = "image"
"""The preprocessing engine's output, named to match what both ALIKED graphs call their input."""

PREPROCESS_ENGINE_DIR: Final[Path] = DEFAULT_TRT_CACHE_DIR / "colsfm-preprocess"
"""Where the generated preprocessing engines are cached: machine-local, beside trtkit's engines."""

FP16_GRAPH_DIR: Final[Path] = DEFAULT_TRT_CACHE_DIR.parent / "onnx-fp16"
"""Where the fp16-trunk rewrites of the ONNX graphs are cached."""


PUBLISHED_FILE_MODE: Final[int] = 0o644
"""Permissions a published engine or timing cache carries.

`tempfile.mkstemp` opens at 0600; these are build products in a shared cache
directory, so they are made world-readable on the way out."""





@dataclass(frozen=True, slots=True)
class DeviceTensor:
    """One tensor that already lives in device memory, ready to bind as an input.

    The point is to feed an engine from device memory without a round trip through
    the host: RaCo extraction builds its batch with torch on the session's stream
    (`colsfm.features_native._device_batch`) and `TensorRTSession.run` binds that
    address directly.
    """

    pointer: int
    """Device address, as `cudaMalloc` and `set_tensor_address` both spell it."""
    shape: tuple[int, ...]
    """The shape to declare for the binding this tensor is passed as."""
    dtype: np.dtype[Any]
    """Element type of the memory at `pointer`.

    Carried so that `TensorRTSession.run` can check a device input against the
    binding's dtype exactly as it checks a host array. Without it the device path was
    the one input the session bound unexamined, and a preprocessor rebuilt at another
    precision would have been read as whatever the engine expected."""
    capacity_bytes: int
    """Bytes the allocation behind `pointer` holds.

    The producer owns a fixed buffer sized for its largest batch and returns a view of
    the first `count` frames, so `shape` is normally smaller than this. The session
    checks `nbytes <= capacity_bytes`, which is the device equivalent of the
    contiguity check a host array gets."""
    stream: int
    """The CUDA stream the producing work was queued on.

    A device input is bound with no copy and no synchronisation, so it is only safe
    when the consumer executes on the same stream. That was a sentence in `run`'s
    docstring; it is a field the session can check."""

    @property
    def nbytes(self) -> int:
        """Bytes `shape` at `dtype` occupies.

        Returns:
            The size of the view this tensor declares, which the allocation must hold.
        """
        return int(np.prod(self.shape)) * self.dtype.itemsize




@dataclass(slots=True)
class DeviceBufferPool:
    """The device buffers one execution context owns, one per binding name.

    Buffers only ever grow, so a run whose shapes change from call to call pays
    one allocation per new high-water mark rather than one per call.

    Ownership is exception-safe in both directions. `ensure` allocates the
    replacement **before** it gives up the old one, so a failed `cudaMalloc`
    leaves the pool owning exactly what it owned before — still valid, still
    bindable — rather than a freed address recorded at its old capacity. `close`
    drops each record before freeing it, so no pointer can be freed twice even if
    one `cudaFree` fails.
    """

    pointers: dict[str, int] = field(default_factory=dict)
    """Device address per binding name; only the bindings that have a buffer."""
    capacities: dict[str, int] = field(default_factory=dict)
    """Bytes owned per binding name; always the same keys as `pointers`."""

    def ensure(self, name: str, byte_count: int) -> int:
        """Guarantee a buffer of at least `byte_count` bytes for one binding.

        Args:
            name: The binding.
            byte_count: Bytes the current call needs; zero is rounded up to one,
                because `cudaMalloc(0)` has no useful address to bind.

        Returns:
            The device pointer, reallocated when the buffer had to grow.

        Raises:
            RuntimeError: When `cudaMalloc` fails. The pool then still owns the
                buffer it had, unchanged.
        """
        wanted: int = max(byte_count, 1)
        if self.capacities.get(name, 0) >= wanted:
            return self.pointers[name]
        pointer: int = int(check_cuda(cudart.cudaMalloc(wanted), f"cudaMalloc({name})")[0])
        previous: int | None = self.pointers.get(name)
        self.pointers[name] = pointer
        self.capacities[name] = wanted
        if previous is not None:
            check_cuda(cudart.cudaFree(previous), f"cudaFree({name})")
        return pointer

    def close(self) -> None:
        """Free every buffer this pool owns; safe to call twice.

        Raises:
            RuntimeError: When a `cudaFree` fails. The buffers freed before it
                stay freed and forgotten; the ones after it are still owned.
        """
        while self.pointers:
            name: str = next(iter(self.pointers))
            pointer: int = self.pointers.pop(name)
            self.capacities.pop(name, None)
            check_cuda(cudart.cudaFree(pointer), f"cudaFree({name})")


def check_cuda(result: tuple[Any, ...] | Any, operation: str) -> tuple[Any, ...]:
    """Raise on a failed `cudart` call and return its payload.

    Every `cuda.bindings.runtime` entry point returns `(status, *values)`, so the
    status has to be stripped before the payload is usable; dropping it silently
    turns a failed allocation into a null pointer used several lines later.

    Args:
        operation: What was attempted, for the error message.
        result: Whatever the `cudart` call returned.

    Returns:
        The payload after the status word.

    Raises:
        RuntimeError: When the status is not `cudaSuccess`.
    """
    values: tuple[Any, ...] = result if isinstance(result, tuple) else (result,)
    if values[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {values[0]}")
    return values[1:]


def compute_capability(device_index: int = 0) -> str:
    """The GPU's compute capability as `<major>_<minor>`.

    Args:
        device_index: CUDA device to query.

    Returns:
        `"<major>_<minor>"`, e.g. `"12_0"` for an RTX 5090.
    """
    properties: Any = check_cuda(cudart.cudaGetDeviceProperties(device_index), "cudaGetDeviceProperties")[0]
    return f"{int(properties.major)}_{int(properties.minor)}"








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


class TensorRTSession:
    """One deserialised engine, its execution context and its device buffers.

    Buffers are allocated lazily and only ever grow, so a run whose shapes change
    from pair to pair — LightGlue, whose keypoint counts differ per image — pays
    one allocation per new high-water mark rather than one per call.

    Use it as a context manager; `close` frees the device memory and the stream.
    """

    def __init__(self, engine_path: Path, *, reuse_output_buffers: bool = False) -> None:
        """Deserialise one engine.

        Args:
            engine_path: A serialised TensorRT engine built for this GPU and
                this TensorRT version.
            reuse_output_buffers: Keep one host array per output binding and hand
                back views into it instead of allocating per call. Off by default
                because it invalidates the previous call's arrays; the feature
                backends turn it on because they copy what they keep out of the
                arrays before the next call.

        Raises:
            FileNotFoundError: When the engine file is missing.
            RuntimeError: When TensorRT refuses to deserialise it.
        """
        if not engine_path.is_file():
            raise FileNotFoundError(f"No TensorRT engine at {engine_path}")
        logger: trt.Logger = trt.Logger(trt.Logger.ERROR)
        if not trt.init_libnvinfer_plugins(logger, ""):
            raise RuntimeError("Could not initialize TensorRT's standard plugin registry")
        self.runtime: trt.Runtime = trt.Runtime(logger)
        engine: trt.ICudaEngine | None = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}; it was built for another GPU or TensorRT")
        self.engine: trt.ICudaEngine = engine
        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        self.input_names: tuple[str, ...] = tuple(
            name
            for name in (self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_names: tuple[str, ...] = tuple(
            name
            for name in (self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        self.dtypes: dict[str, np.dtype[Any]] = {
            self.engine.get_tensor_name(index): np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.engine.get_tensor_name(index))))
            for index in range(self.engine.num_io_tensors)
        }
        self.buffers: DeviceBufferPool = DeviceBufferPool()
        self.reuse_output_buffers: bool = reuse_output_buffers
        self.host_outputs: dict[str, ndarray] = {}
        self.stream: int = int(check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")[0])
        self.closed: bool = False

    def _host_output(self, name: str, shape: tuple[int, ...]) -> ndarray:
        """A host array of `shape` for one output binding, reused when allowed.

        Args:
            name: The output binding.
            shape: The shape TensorRT resolved for this call.

        Returns:
            A writable array. With `reuse_output_buffers` it is the same memory
            every call, so the previous call's view is invalidated; without it a
            fresh `np.empty`, whose first-touch page faults cost about 1 ms per
            megabyte of descriptors (docs/raco-speed-investigation.md §4).
        """
        if not self.reuse_output_buffers:
            return np.empty(shape, dtype=self.dtypes[name])
        cached: ndarray | None = self.host_outputs.get(name)
        if cached is None or cached.size < int(np.prod(shape)):
            cached = np.zeros(int(np.prod(shape)), dtype=self.dtypes[name])
            self.host_outputs[name] = cached
        return cached[: int(np.prod(shape))].reshape(shape)

    def run(self, inputs: Mapping[str, ndarray | DeviceTensor]) -> dict[str, ndarray]:
        """Execute the engine once and return every output as a host array.

        Args:
            inputs: Per input binding, either one contiguous host array already
                in the binding's dtype — which is copied to the device here — or
                a `DeviceTensor` that some earlier work on **this session's
                stream** already produced, which is bound in place with no copy.

        Returns:
            One host array per output binding, at the shape TensorRT resolved.
            With `reuse_output_buffers` these are views into session-owned memory
            and are only valid until the next `run`.

        Raises:
            RuntimeError: When the session is closed, when a profile rejects a
                shape, or when execution fails.
            ValueError: When an input is missing or mistyped, when a host array is
                not contiguous, or when a `DeviceTensor` was produced on another
                stream or declares more bytes than its allocation holds.
        """
        if self.closed:
            raise RuntimeError("This TensorRTSession is closed")
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"TensorRT engine input {name} was not supplied")
            supplied: ndarray | DeviceTensor = inputs[name]
            if supplied.dtype != self.dtypes[name]:
                raise ValueError(f"Input {name} must be {self.dtypes[name]}, got {supplied.dtype}")
            if isinstance(supplied, DeviceTensor):
                if supplied.stream != self.stream:
                    raise ValueError(f"Device input {name} was produced on another stream; binding it in place is unsafe")
                if supplied.nbytes > supplied.capacity_bytes:
                    raise ValueError(
                        f"Device input {name} declares {supplied.nbytes} bytes but its allocation holds {supplied.capacity_bytes}"
                    )
                input_shape: tuple[int, ...] = supplied.shape
            else:
                if not supplied.flags.c_contiguous:
                    raise ValueError(f"Input {name} must be C-contiguous")
                input_shape = tuple(supplied.shape)
            if not self.context.set_input_shape(name, input_shape):
                raise RuntimeError(f"TensorRT profile rejected shape {input_shape} for {name}")
        for name in self.input_names:
            supplied = inputs[name]
            if isinstance(supplied, DeviceTensor):
                self.context.set_tensor_address(name, supplied.pointer)
                continue
            pointer: int = self.buffers.ensure(name, supplied.nbytes)
            self.context.set_tensor_address(name, pointer)
            check_cuda(
                cudart.cudaMemcpyAsync(
                    pointer,
                    supplied.ctypes.data,
                    supplied.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                ),
                f"cudaMemcpyAsync(H2D {name})",
            )
        outputs: dict[str, ndarray] = {}
        for name in self.output_names:
            shape: tuple[int, ...] = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"TensorRT left output {name} with an unresolved shape {shape}")
            host: ndarray = self._host_output(name, shape)
            outputs[name] = host
            self.context.set_tensor_address(name, self.buffers.ensure(name, max(host.nbytes, 1)))
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execution failed")
        for name, host in outputs.items():
            if host.nbytes == 0:
                continue
            check_cuda(
                cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    self.buffers.pointers[name],
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f"cudaMemcpyAsync(D2H {name})",
            )
        check_cuda(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        return outputs

    def close(self) -> None:
        """Free the device buffers and the stream; safe to call twice."""
        if self.closed:
            return
        self.closed = True
        check_cuda(cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
        self.buffers.close()

    def __enter__(self) -> Self:
        """Return the session itself, so `with TensorRTSession(...) as session` works."""
        return self

    def __exit__(self, *_arguments: object) -> None:
        """Free the device buffers and the stream."""
        self.close()






