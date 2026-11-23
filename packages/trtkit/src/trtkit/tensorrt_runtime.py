"""TensorRT backend for the trtkit runtime contract.

Unifies the TensorRT runners previously copied between ``wilor-nano``,
``sapiens2-pose``/``sapiens-coco133-pose``, ``prompt-da``, and ``mamma``:
persistent torch-tensor I/O bound via ``set_tensor_address`` (no host copies),
``execute_async_v3``, and optional CUDA-graph capture/replay for
launch-overhead-critical loops (the mamma pattern). Dynamic-batch engines
execute at the caller's true batch size; static-batch engines (fixed-batch
ONNX exports) zero-pad up to their baked batch.
"""

import math
from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import Tensor

from trtkit.base import RuntimeSpec, TensorSpec, validate_runtime_inputs


class _TrtEngine:
    """Deserialized engine, execution context, private stream, and CUDA-graph capture shared by both runtimes."""

    def __init__(self, engine_path: Path, *, use_cuda_graph: bool, user_managed_memory: bool = False) -> None:
        """Deserialize a machine-local engine and create its execution context.

        ``user_managed_memory``: the context gets no activation memory of its own; the subclass binds it per call.

        Raises:
            RuntimeError: If CUDA is unavailable or the engine cannot be deserialized.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT execution requires CUDA.")
        import tensorrt

        self._trt: Any = cast(Any, tensorrt)  # the compiled bindings have incomplete stubs
        engine: Any = self._trt.Runtime(self._trt.Logger(self._trt.Logger.WARNING)).deserialize_cuda_engine(engine_path.expanduser().read_bytes())
        strategy: Any = (
            self._trt.ExecutionContextAllocationStrategy.USER_MANAGED if user_managed_memory else self._trt.ExecutionContextAllocationStrategy.STATIC
        )
        context: Any = None if engine is None else engine.create_execution_context(strategy)
        if engine is None or context is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        self._engine: Any = engine
        self._context: Any = context
        self._device: torch.device = torch.device("cuda")
        self._stream: torch.cuda.Stream = torch.cuda.Stream(device=self._device)
        self._use_cuda_graph: bool = use_cuda_graph

    def close(self) -> None:
        """Release the CUDA graphs, I/O buffers, execution context and engine now, not at the next garbage collection.

        Idempotent; a closed runtime refuses calls. ``with Runtime(...) as runtime:`` closes at the block's end.
        """
        if getattr(self, "_engine", None) is None:
            return
        self._stream.synchronize()
        # Captured launches go before the context and engine they replay.
        for graph in getattr(self, "_graphs", {}).values():
            graph.reset()
        self._graphs = {}
        self._input_buffers = {}
        self._output_buffers = {}
        self._activation_memory = None
        self._context = None
        self._engine = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        """Close on collection when the owner did not."""
        self.close()

    def _require_open(self) -> None:
        if self._context is None:
            raise RuntimeError("this TensorRT runtime is closed")

    def _execute(self) -> None:
        """Launch the engine on the private stream, fenced against torch's current stream.

        Raises:
            RuntimeError: If TensorRT reports a failed ``execute_async_v3`` call.
        """
        current: torch.cuda.Stream = torch.cuda.current_stream(self._device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            ok: bool = bool(self._context.execute_async_v3(stream_handle=int(self._stream.cuda_stream)))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        current.wait_stream(self._stream)

    def _capture_graph(self) -> torch.cuda.CUDAGraph:
        """Warm up and capture a single engine launch into a CUDA graph.

        The capture is valid for the currently active input shapes; dynamic
        engines keep one graph per batch size.

        Returns:
            Captured graph whose replay re-runs the engine on the persistent buffers.

        Raises:
            RuntimeError: If the warmup or capture launch fails.
        """
        warmup: torch.cuda.Stream = torch.cuda.Stream(device=self._device)
        warmup.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(warmup):
            if not bool(self._context.execute_async_v3(stream_handle=int(warmup.cuda_stream))):
                raise RuntimeError("TensorRT warmup launch failed.")
        torch.cuda.current_stream(self._device).wait_stream(warmup)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            stream: torch.cuda.Stream = torch.cuda.current_stream(self._device)
            if not bool(self._context.execute_async_v3(stream_handle=int(stream.cuda_stream))):
                raise RuntimeError("TensorRT capture launch failed.")
        return graph


class TensorRtRuntime(_TrtEngine):
    """TensorRT engine implementing the trtkit runtime contract."""

    def __init__(self, engine_path: Path, *, use_cuda_graph: bool = False) -> None:
        """Deserialize a machine-local engine and bind persistent I/O buffers.

        Args:
            engine_path: Path to a TensorRT engine built for this machine/GPU
                (see :mod:`trtkit.trt_builder`).
            use_cuda_graph: Capture ``execute_async_v3`` launches into CUDA
                graphs (one per distinct batch size on dynamic engines) and
                replay them afterwards. Worth it for small/latency-bound
                engines called in tight loops.

        Raises:
            RuntimeError: If CUDA is unavailable or the engine cannot be
                deserialized.
            ValueError: If the engine has dynamic non-batch dims or its inputs
                disagree on batch size.
        """
        super().__init__(engine_path, use_cuda_graph=use_cuda_graph)
        trt: Any = self._trt
        engine: Any = self._engine
        input_specs: list[TensorSpec] = []
        output_specs: list[TensorSpec] = []
        max_batches: set[int] = set()
        self._dynamic: bool = False
        for idx in range(int(engine.num_io_tensors)):
            name: str = str(engine.get_tensor_name(idx))
            shape: tuple[int, ...] = tuple(int(dim) for dim in engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape[1:]):
                raise ValueError(f"Engine tensor {name!r} has dynamic non-batch dims {shape}; trtkit requires static per-sample shapes.")
            spec = TensorSpec(name=name, shape=shape[1:], dtype=_torch_dtype(engine.get_tensor_dtype(name), trt))
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_specs.append(spec)
                if shape[0] < 0:
                    self._dynamic = True
                    profile_max: tuple[int, ...] = tuple(int(dim) for dim in engine.get_tensor_profile_shape(name, 0)[2])
                    max_batches.add(profile_max[0])
                else:
                    max_batches.add(shape[0])
            else:
                output_specs.append(spec)
        if len(max_batches) != 1:
            raise ValueError(f"Engine inputs disagree on batch size: {sorted(max_batches)}.")
        max_batch: int = max_batches.pop()
        self._spec = RuntimeSpec(inputs=tuple(input_specs), outputs=tuple(output_specs), max_batch_size=max_batch)
        # Persistent buffers: stable addresses are required both for one-time
        # set_tensor_address binding and for CUDA-graph capture/replay.
        self._input_buffers: dict[str, Tensor] = {
            spec.name: torch.zeros((max_batch, *spec.shape), dtype=spec.dtype, device=self._device) for spec in self._spec.inputs
        }
        self._active_batch: int = -1
        self._set_active_batch(max_batch)
        # Output buffers size from the context's concrete max-batch shapes, not
        # (max_batch, *per_sample): an extra materialized intermediate (see
        # TrtBuildConfig.extra_output_patterns) may scale its leading dim by more
        # than the batch (e.g. batch*views), and undersized buffers corrupt memory.
        self._output_buffers: dict[str, Tensor] = {
            spec.name: torch.empty(tuple(int(dim) for dim in self._context.get_tensor_shape(spec.name)), dtype=spec.dtype, device=self._device)
            for spec in self._spec.outputs
        }
        # Static engines report the baked max batch through the context at every
        # call, so padded rows must be sliced off by per-sample ratio; dynamic
        # engines report exact shapes per active batch and owe no divisibility.
        # The slice assumes sample-major rows (leading dim = batch * k); a
        # k-major intermediate would yield the wrong rows — none exists today.
        self._static_rows_per_sample: dict[str, int] = {}
        if not self._dynamic:
            for out_spec in self._spec.outputs:
                rows_at_max: int = int(self._output_buffers[out_spec.name].shape[0])
                if rows_at_max % max_batch != 0:
                    raise ValueError(
                        f"Static engine output {out_spec.name!r} has {rows_at_max} rows at batch {max_batch}; "
                        "padded rows cannot be sliced off a non-multiple. Constant-shaped outputs usually "
                        "come from TrtBuildConfig.extra_output_patterns intermediates — rebuild without "
                        "materializing that tensor, or use a dynamic-batch engine."
                    )
                self._static_rows_per_sample[out_spec.name] = rows_at_max // max_batch
        for name, tensor in {**self._input_buffers, **self._output_buffers}.items():
            self._context.set_tensor_address(name, int(tensor.data_ptr()))
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    @property
    def spec(self) -> RuntimeSpec:
        """Static I/O contract of this runtime."""
        return self._spec

    def _set_active_batch(self, batch_size: int) -> None:
        """Point a dynamic engine's input shapes at the given batch size."""
        if not self._dynamic or batch_size == self._active_batch:
            self._active_batch = batch_size
            return
        for tensor_spec in self._spec.inputs:
            self._context.set_input_shape(tensor_spec.name, (batch_size, *tensor_spec.shape))
        self._active_batch = batch_size

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run one batch (true batch on dynamic engines, zero-padded on static).

        Args:
            inputs: CUDA tensors keyed by ``spec.inputs`` names.

        Returns:
            CUDA tensors keyed by ``spec.outputs`` names, sliced to the submitted
            batch size. Buffers are reused across calls — clone before mutating
            or holding results past the next call.
        """
        self._require_open()
        batch_size: int = validate_runtime_inputs(self._spec, inputs)
        for tensor_spec in self._spec.inputs:
            tensor: Tensor = inputs[tensor_spec.name].to(dtype=tensor_spec.dtype)
            buffer: Tensor = self._input_buffers[tensor_spec.name]
            buffer[:batch_size].copy_(tensor)
            if not self._dynamic and batch_size < self._spec.max_batch_size:
                buffer[batch_size:] = 0
        self._set_active_batch(batch_size)
        if self._use_cuda_graph:
            graph_key: int = batch_size if self._dynamic else self._spec.max_batch_size
            if graph_key not in self._graphs:
                self._graphs[graph_key] = self._capture_graph()
            self._graphs[graph_key].replay()
        else:
            self._execute()
        # Dynamic engines report exact per-batch output shapes; static engines
        # always report the baked max batch, so padded rows are sliced off by
        # the per-sample ratio recorded at construction.
        if self._dynamic:
            return {name: tensor[: int(self._context.get_tensor_shape(name)[0])] for name, tensor in self._output_buffers.items()}
        return {name: tensor[: self._static_rows_per_sample[name] * batch_size] for name, tensor in self._output_buffers.items()}


class TensorRtDynamicRuntime(_TrtEngine):
    """TensorRT engine whose inputs may be dynamic in any dimension (one optimization profile).

    Companion to :class:`TensorRtRuntime` for engines built with
    ``TrtBuildConfig.shape_profiles``: inputs need not share a batch dim, and any
    dim may vary within the profile. Buffers are allocated once at the profile's
    max shapes; each call pins the concrete input shapes, copies inputs into the
    buffer prefixes, and returns output views shaped by the execution context.
    CUDA graphs are captured per distinct input-shape signature and kept for the
    runtime's lifetime, so a caller that walks many shapes should pass
    ``use_cuda_graph=False``.

    ``direct_binding=True`` is the contract for callers that already hold their
    inputs on the device and whose shapes change from call to call:

    * **Inputs** are bound by address, not copied: each must be a contiguous CUDA
      tensor of the binding's dtype. They are read on the runtime's stream after it
      waits on the caller's current stream, so produce them on that stream; the
      runtime records them on its stream, so the caching allocator keeps them until
      the engine has read them even if the caller drops its reference.
    * **Outputs** live in runtime-owned device buffers sized to the shape TensorRT
      resolves for the call, grown (never shrunk) when a call needs more; nothing is
      allocated at the profile's max shapes.
    * **Ownership**: the returned tensors are views into those buffers, valid until
      the next call, which may overwrite or reallocate them. Copy what you keep
      (``.clone()``, or a host copy at the boundary that needs one).

    CUDA graphs need fixed addresses, so ``direct_binding`` and ``use_cuda_graph``
    exclude each other.

    ``per_shape_activation_memory=True`` sizes the context's activation memory to each call's shapes instead of the
    profile's max: for engines built with ``TrtBuildConfig.runtime_activation_resize``, whose max corner can need
    several times the memory of the shapes callers send. The buffer grows, never shrinks, and is re-sized only when
    the input shapes change. It moves when it grows, so it excludes CUDA graphs.
    """

    def __init__(
        self, engine_path: Path, *, use_cuda_graph: bool = False, direct_binding: bool = False, per_shape_activation_memory: bool = False
    ) -> None:
        """Deserialize an engine and allocate its I/O buffers.

        Args:
            engine_path: Machine-local engine built with shape profiles.
            use_cuda_graph: Capture and replay one CUDA graph per input-shape signature.
            direct_binding: Bind device inputs in place and size outputs to each call (see the class docstring)
                instead of copying into max-shape buffers.
            per_shape_activation_memory: Size activation memory to each call's shapes (see the class docstring).

        Raises:
            RuntimeError: If CUDA is unavailable or the engine cannot be deserialized.
            ValueError: If ``use_cuda_graph`` is set together with ``direct_binding`` or ``per_shape_activation_memory``.
        """
        if direct_binding and use_cuda_graph:
            raise ValueError("direct_binding binds a new address per call; CUDA graph replay needs fixed addresses.")
        if per_shape_activation_memory and use_cuda_graph:
            raise ValueError("per_shape_activation_memory moves activation memory when it grows; CUDA graph replay needs fixed addresses.")
        super().__init__(engine_path, use_cuda_graph=use_cuda_graph, user_managed_memory=per_shape_activation_memory)
        self.direct_binding: bool = direct_binding
        self.per_shape_activation_memory: bool = per_shape_activation_memory
        self._activation_memory: Tensor | None = None
        self._activation_shapes: tuple[tuple[str, tuple[int, ...]], ...] | None = None
        """Input shapes the bound activation memory was sized for."""
        self.device_index: int = torch.cuda.current_device()
        """The CUDA device the context and stream were created on; direct-binding inputs must live there."""
        trt: Any = self._trt
        engine: Any = self._engine
        self.min_input_shapes: dict[str, tuple[int, ...]] = {}
        """Smallest accepted shape per input."""
        self.max_input_shapes: dict[str, tuple[int, ...]] = {}
        """Largest accepted shape per input; the persistent buffer size."""
        self._input_dtypes: dict[str, torch.dtype] = {}
        self._output_dtypes: dict[str, torch.dtype] = {}
        for idx in range(int(engine.num_io_tensors)):
            name: str = str(engine.get_tensor_name(idx))
            dtype: torch.dtype = _torch_dtype(engine.get_tensor_dtype(name), trt)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                self._output_dtypes[name] = dtype
                continue
            shape: tuple[int, ...] = tuple(int(dim) for dim in engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                bounds: Any = engine.get_tensor_profile_shape(name, 0)
                self.min_input_shapes[name] = tuple(int(dim) for dim in bounds[0])
                self.max_input_shapes[name] = tuple(int(dim) for dim in bounds[2])
            else:
                self.min_input_shapes[name] = shape
                self.max_input_shapes[name] = shape
            self._input_dtypes[name] = dtype
        self._active_shapes: dict[str, tuple[int, ...]] = {}
        for name, shape in self.max_input_shapes.items():
            self._set_input_shape(name, shape)
        self.max_output_shapes: dict[str, tuple[int, ...]] = {
            name: tuple(int(dim) for dim in self._context.get_tensor_shape(name)) for name in self._output_dtypes
        }
        """Output shapes at the profile's max input shapes; the persistent buffer size unless ``direct_binding``."""
        self._input_buffers: dict[str, Tensor] = {}
        self._output_buffers: dict[str, Tensor] = {}
        if not direct_binding:
            self._input_buffers = {
                name: torch.zeros(math.prod(shape), dtype=self._input_dtypes[name], device=self._device)
                for name, shape in self.max_input_shapes.items()
            }
            self._output_buffers = {
                name: torch.empty(math.prod(shape), dtype=self._output_dtypes[name], device=self._device)
                for name, shape in self.max_output_shapes.items()
            }
            for name, tensor in {**self._input_buffers, **self._output_buffers}.items():
                self._context.set_tensor_address(name, int(tensor.data_ptr()))
        self._graphs: dict[tuple[tuple[str, tuple[int, ...]], ...], torch.cuda.CUDAGraph] = {}
        self.device_memory_bytes: int = int(engine.device_memory_size_v2)
        """Activation memory TensorRT reserves for the execution context (sized at the profile's max shapes)."""

    def _set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        """Pin one input's concrete shape on the context when it changes."""
        if self._active_shapes.get(name) == shape:
            return
        if not bool(self._context.set_input_shape(name, shape)):
            raise ValueError(
                f"TensorRT rejected shape {shape} for input {name!r} (profile {self.min_input_shapes[name]}..{self.max_input_shapes[name]})."
            )
        self._active_shapes[name] = shape

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run one call at the inputs' concrete shapes.

        Args:
            inputs: CUDA tensors keyed by input name, each within its profile bounds.

        Returns:
            Output views into runtime-owned buffers (overwritten by the next call), shaped by the context.

        Raises:
            ValueError: If names, ranks, or shapes fall outside the engine's profile, an output shape is not resolved
                by the input shapes (a negative, data-dependent dimension), or, with ``direct_binding``, an input is not
                a contiguous CUDA tensor of its binding's dtype on the runtime's device.
        """
        self._require_open()
        if set(inputs) != set(self._input_dtypes):
            raise ValueError(f"Runtime expects inputs {tuple(self._input_dtypes)}, got {tuple(inputs)}.")
        for name, tensor in inputs.items():
            shape: tuple[int, ...] = tuple(int(dim) for dim in tensor.shape)
            lower: tuple[int, ...] = self.min_input_shapes[name]
            upper: tuple[int, ...] = self.max_input_shapes[name]
            if len(shape) != len(upper) or any(dim < low or dim > high for dim, low, high in zip(shape, lower, upper, strict=True)):
                raise ValueError(f"Input {name!r} shape {shape} is outside the engine profile {lower}..{upper}.")
            self._set_input_shape(name, shape)
        # Resolved once, before anything is bound or allocated: a negative dimension is data-dependent, and no buffer
        # sized here could hold it (that needs TensorRT's output allocator, which this runtime does not implement).
        output_shapes: dict[str, tuple[int, ...]] = {
            name: tuple(int(dim) for dim in self._context.get_tensor_shape(name)) for name in self._output_dtypes
        }
        unresolved: dict[str, tuple[int, ...]] = {name: shape for name, shape in output_shapes.items() if any(dim < 0 for dim in shape)}
        if unresolved:
            raise ValueError(f"Output shapes {unresolved} are not resolved by the input shapes; data-dependent outputs are not supported.")
        for name, tensor in inputs.items():
            if self.direct_binding:
                if tensor.dtype != self._input_dtypes[name] or not tensor.is_cuda or not tensor.is_contiguous():
                    raise ValueError(
                        f"Input {name!r} must be a contiguous CUDA {self._input_dtypes[name]} tensor to bind in place, got {tensor.dtype}."
                    )
                if tensor.device.index != self.device_index:
                    raise ValueError(f"Input {name!r} is on {tensor.device}; the engine's context and stream are on cuda:{self.device_index}.")
                tensor.record_stream(self._stream)
                self._context.set_tensor_address(name, int(tensor.data_ptr()))
                continue
            numel: int = math.prod(tensor.shape)
            self._input_buffers[name][:numel].copy_(tensor.reshape(-1).to(dtype=self._input_dtypes[name]))
        if self.direct_binding:
            for name, dtype in self._output_dtypes.items():
                # A resolved zero-size output still gets one element, so its address is a real allocation.
                needed: int = max(math.prod(output_shapes[name]), 1)
                buffer: Tensor | None = self._output_buffers.get(name)
                if buffer is None or buffer.numel() < needed:
                    self._output_buffers[name] = torch.empty(needed, dtype=dtype, device=self._device)
                self._context.set_tensor_address(name, int(self._output_buffers[name].data_ptr()))
        active_shapes: tuple[tuple[str, tuple[int, ...]], ...] = tuple(sorted(self._active_shapes.items()))
        if self.per_shape_activation_memory and active_shapes != self._activation_shapes:
            needed_bytes: int = max(int(self._context.update_device_memory_size_for_shapes()), 1)
            if self._activation_memory is None or self._activation_memory.numel() < needed_bytes:
                self._stream.synchronize()  # the previous call may still read the old buffer
                self._activation_memory = None
                self._activation_memory = torch.empty(needed_bytes, dtype=torch.uint8, device=self._device)
            self._context.set_device_memory(int(self._activation_memory.data_ptr()), int(self._activation_memory.numel()))
            self._activation_shapes = active_shapes
        if self._use_cuda_graph:
            if active_shapes not in self._graphs:
                self._graphs[active_shapes] = self._capture_graph()
            self._graphs[active_shapes].replay()
        else:
            self._execute()
        return {name: buffer[: math.prod(output_shapes[name])].view(output_shapes[name]) for name, buffer in self._output_buffers.items()}


def _torch_dtype(dtype: Any, trt: Any) -> torch.dtype:
    """Map TensorRT dtypes to torch dtypes.

    Args:
        dtype: TensorRT dtype object returned by the engine.
        trt: Imported TensorRT module.

    Returns:
        Equivalent ``torch.dtype``.

    Raises:
        TypeError: If the dtype is not supported by this runtime.
    """
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if hasattr(trt, "bfloat16") and dtype == trt.bfloat16:
        return torch.bfloat16
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.int32:
        return torch.int32
    raise TypeError(f"Unsupported TensorRT tensor dtype: {dtype}")
