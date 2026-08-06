"""TensorRT backend for the trtkit runtime contract.

Unifies the TensorRT runners previously copied between ``wilor-nano``,
``sapiens2-pose``/``sapiens-coco133-pose``, ``prompt-da``, and ``mamma``:
persistent torch-tensor I/O bound via ``set_tensor_address`` (no host copies),
``execute_async_v3``, and optional CUDA-graph capture/replay for
launch-overhead-critical loops (the mamma pattern). Dynamic-batch engines
execute at the caller's true batch size; static-batch engines (fixed-batch
ONNX exports) zero-pad up to their baked batch.
"""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from trtkit.base import RuntimeSpec, TensorSpec, validate_runtime_inputs


class TensorRtRuntime:
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
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT execution requires CUDA.")
        import tensorrt

        trt: Any = tensorrt  # the compiled bindings have incomplete stubs
        engine: Any = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_path.expanduser().read_bytes())
        context: Any = None if engine is None else engine.create_execution_context()
        if engine is None or context is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        self._trt: Any = trt
        self._engine: Any = engine
        self._context: Any = context
        self._device: torch.device = torch.device("cuda")
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
        self._output_buffers: dict[str, Tensor] = {
            spec.name: torch.empty((max_batch, *spec.shape), dtype=spec.dtype, device=self._device) for spec in self._spec.outputs
        }
        for name, tensor in {**self._input_buffers, **self._output_buffers}.items():
            self._context.set_tensor_address(name, int(tensor.data_ptr()))
        self._active_batch: int = -1
        self._set_active_batch(max_batch)
        self._stream: torch.cuda.Stream = torch.cuda.Stream(device=self._device)
        self._use_cuda_graph: bool = use_cuda_graph
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
        return {name: tensor[:batch_size] for name, tensor in self._output_buffers.items()}

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
