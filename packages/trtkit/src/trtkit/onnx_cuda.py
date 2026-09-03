"""ONNX Runtime CUDA backend with IOBinding onto torch tensor memory.

Inputs and outputs are bound directly to torch CUDA tensor ``data_ptr``s so no
numpy/host copies happen anywhere in the hot path — the gap that makes the
stock rtmlib ONNX path slow. When available, the session is attached to the
current torch CUDA stream (``user_compute_stream``) so execution is fully
stream-ordered with surrounding torch preprocessing/decode kernels.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from beartype.roar import BeartypeException
from torch import Tensor

from trtkit.base import RuntimeSpec, TensorSpec, validate_runtime_inputs

_ORT_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int64)": torch.int64,
    "tensor(int32)": torch.int32,
    "tensor(uint8)": torch.__dict__["uint8"],
    "tensor(bool)": torch.bool,
}

_TORCH_TO_NUMPY_DTYPE: dict[torch.dtype, type] = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.__dict__["uint8"]: np.uint8,
    torch.bool: np.bool_,
}


class OnnxCudaRuntime:
    """GPU-resident ONNX Runtime session implementing the trtkit runtime contract."""

    def __init__(self, onnx_path: Path, *, device_id: int = 0, max_batch_size: int = 32) -> None:
        """Load an ONNX graph into a CUDA execution-provider session.

        Args:
            onnx_path: Path to the ONNX model file.
            device_id: CUDA device ordinal for the execution provider.
            max_batch_size: Batch cap for graphs with a dynamic batch dimension.
                Graphs with a static batch dimension use that value instead and
                pad smaller batches up to it.

        Raises:
            RuntimeError: If onnxruntime or its CUDA provider is unavailable.
            ValueError: If the graph has symbolic non-batch dimensions.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed in this Pixi environment.") from exc
        # The PyPI wheel dlopens cuDNN/cuBLAS/cudart from the conda prefix. Under
        # `pixi run` the cuda feature's LD_LIBRARY_PATH covers that; a bare
        # `.pixi/envs/<env>/bin/python` does not, so point ORT at the prefix
        # explicitly (its default search only knows torch on Windows and the
        # nvidia-* pip packages).
        ort.preload_dlls(directory=str(Path(sys.prefix) / "lib"))
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("onnxruntime CUDAExecutionProvider is unavailable; install onnxruntime-gpu.")
        self._device: torch.device = torch.device("cuda", device_id)
        # Dedicated session stream, NEVER torch's current stream: the default
        # stream's raw handle is 0, which ORT parses as "no user stream" and
        # silently runs on its own non-blocking internal stream — unfenced
        # against torch, so inputs race (intermittent garbage inferences). A
        # torch-owned side stream has a real handle ORT honors, and __call__
        # fences it against the caller's stream on both sides of the run.
        self._torch_stream: torch.cuda.Stream = torch.cuda.Stream(device=self._device)
        provider_options: dict[str, Any] = {"device_id": device_id, "user_compute_stream": str(int(self._torch_stream.cuda_stream))}
        session_options: Any = ort.SessionOptions()
        session_options.log_severity_level = 3
        try:
            self._session: Any = ort.InferenceSession(
                str(onnx_path), sess_options=session_options, providers=[("CUDAExecutionProvider", provider_options)]
            )
            self._stream_ordered: bool = True
        except BeartypeException:
            raise
        except Exception as error:
            # Perf-relevant mode switch: every call will host-sync the caller's
            # stream instead of being stream-ordered — say so, don't hide it.
            print(f"[trtkit] ORT user_compute_stream unavailable ({error}); falling back to a host-synchronized session.")
            self._session = ort.InferenceSession(
                str(onnx_path), sess_options=session_options, providers=[("CUDAExecutionProvider", {"device_id": device_id})]
            )
            self._stream_ordered = False
        self._device_id: int = device_id
        static_batch: int | None = None
        input_specs: list[TensorSpec] = []
        for node in self._session.get_inputs():
            spec, node_batch = _tensor_spec_from_node(node)
            input_specs.append(spec)
            static_batch = node_batch if node_batch is not None else static_batch
        output_specs: list[TensorSpec] = self._resolve_output_specs(tuple(input_specs), static_batch)
        self._static_batch: int | None = static_batch
        self._spec = RuntimeSpec(
            inputs=tuple(input_specs),
            outputs=tuple(output_specs),
            max_batch_size=static_batch if static_batch is not None else max_batch_size,
        )
        self._input_buffers: dict[str, Tensor] = {}
        self._output_buffers: dict[str, Tensor] | None = None

    @property
    def spec(self) -> RuntimeSpec:
        """Static I/O contract of this runtime."""
        return self._spec

    def _resolve_output_specs(self, input_specs: tuple[TensorSpec, ...], static_batch: int | None) -> list[TensorSpec]:
        """Determine concrete output shapes, probing the session once if needed.

        Zoo exports (e.g. RTMW) sometimes leave statically-determined output
        dims symbolic in the graph. A single dummy-input run with ORT-allocated
        outputs recovers the concrete shapes so the hot path can use
        preallocated torch buffers.

        Args:
            input_specs: Fully parsed input contract.
            static_batch: Static graph batch size, or ``None`` when dynamic.

        Returns:
            Output specs with fully concrete per-sample shapes.
        """
        output_nodes: list[Any] = list(self._session.get_outputs())
        try:
            return [_tensor_spec_from_node(node)[0] for node in output_nodes]
        except ValueError:
            pass
        probe_batch: int = static_batch if static_batch is not None else 1
        binding: Any = self._session.io_binding()
        for tensor_spec in input_specs:
            dummy: Tensor = torch.zeros((probe_batch, *tensor_spec.shape), dtype=tensor_spec.dtype, device=self._device)
            binding.bind_input(
                name=tensor_spec.name,
                device_type="cuda",
                device_id=self._device_id,
                element_type=_TORCH_TO_NUMPY_DTYPE[tensor_spec.dtype],
                shape=tuple(int(dim) for dim in dummy.shape),
                buffer_ptr=int(dummy.data_ptr()),
            )
        for node in output_nodes:
            binding.bind_output(node.name, device_type="cuda", device_id=self._device_id)
        self._session.run_with_iobinding(binding)
        specs: list[TensorSpec] = []
        for node, ort_value in zip(output_nodes, binding.get_outputs(), strict=True):
            shape: tuple[int, ...] = tuple(int(dim) for dim in ort_value.shape())
            dtype: torch.dtype | None = _ORT_TO_TORCH_DTYPE.get(str(node.type))
            if dtype is None:
                raise ValueError(f"Unsupported ONNX tensor type {node.type!r} for {node.name!r}.")
            specs.append(TensorSpec(name=str(node.name), shape=shape[1:], dtype=dtype))
        return specs

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run one batch through ONNX Runtime with zero host copies.

        Args:
            inputs: CUDA tensors keyed by ``spec.inputs`` names.

        Returns:
            CUDA tensors keyed by ``spec.outputs`` names, sliced to the submitted
            batch size. Buffers are reused across calls — clone before mutating
            or holding results past the next call.
        """
        batch_size: int = validate_runtime_inputs(self._spec, inputs)
        bound_batch: int = self._static_batch if self._static_batch is not None else batch_size
        binding: Any = self._session.io_binding()
        for tensor_spec in self._spec.inputs:
            tensor: Tensor = self._bound_input(tensor_spec, inputs[tensor_spec.name], bound_batch)
            binding.bind_input(
                name=tensor_spec.name,
                device_type="cuda",
                device_id=self._device_id,
                element_type=_TORCH_TO_NUMPY_DTYPE[tensor_spec.dtype],
                shape=tuple(int(dim) for dim in tensor.shape),
                buffer_ptr=int(tensor.data_ptr()),
            )
        if self._output_buffers is None or int(next(iter(self._output_buffers.values())).shape[0]) != bound_batch:
            self._output_buffers = {
                tensor_spec.name: torch.empty((bound_batch, *tensor_spec.shape), dtype=tensor_spec.dtype, device=self._device)
                for tensor_spec in self._spec.outputs
            }
        for tensor_spec in self._spec.outputs:
            output: Tensor = self._output_buffers[tensor_spec.name]
            binding.bind_output(
                name=tensor_spec.name,
                device_type="cuda",
                device_id=self._device_id,
                element_type=_TORCH_TO_NUMPY_DTYPE[tensor_spec.dtype],
                shape=tuple(int(dim) for dim in output.shape),
                buffer_ptr=int(output.data_ptr()),
            )
        # The session executes on the construction-time stream; when the caller
        # is on a different stream, order the two on both sides of the run so
        # ORT never reads half-written inputs and torch never reads stale
        # outputs. The non-stream-ordered fallback must drain the CALLER's
        # stream (ORT runs on its own internal stream there).
        current_stream: torch.cuda.Stream = torch.cuda.current_stream(self._device)
        if self._stream_ordered:
            if current_stream != self._torch_stream:
                self._torch_stream.wait_stream(current_stream)
            self._session.run_with_iobinding(binding)
            if current_stream != self._torch_stream:
                current_stream.wait_stream(self._torch_stream)
        else:
            current_stream.synchronize()
            self._session.run_with_iobinding(binding)
        return {name: buffer[:batch_size] for name, buffer in self._output_buffers.items()}

    def _bound_input(self, tensor_spec: TensorSpec, tensor: Tensor, bound_batch: int) -> Tensor:
        """Return a contiguous CUDA tensor padded to the bound batch size.

        Args:
            tensor_spec: Input contract being bound.
            tensor: Caller-provided input for this binding.
            bound_batch: Batch size the session will actually execute.

        Returns:
            The caller tensor when it already matches, otherwise a reusable
            zero-padded staging buffer.

        Raises:
            ValueError: If the tensor is not on the session's CUDA device.
        """
        if tensor.device != self._device:
            raise ValueError(f"Input {tensor_spec.name!r} must live on {self._device}, got {tensor.device}.")
        tensor = tensor.to(dtype=tensor_spec.dtype).contiguous()
        if int(tensor.shape[0]) == bound_batch:
            return tensor
        buffer: Tensor | None = self._input_buffers.get(tensor_spec.name)
        if buffer is None or tuple(int(dim) for dim in buffer.shape) != (bound_batch, *tensor_spec.shape):
            buffer = torch.zeros((bound_batch, *tensor_spec.shape), dtype=tensor_spec.dtype, device=self._device)
            self._input_buffers[tensor_spec.name] = buffer
        buffer[: int(tensor.shape[0])].copy_(tensor)
        buffer[int(tensor.shape[0]) :] = 0
        return buffer


def _tensor_spec_from_node(node: Any) -> tuple[TensorSpec, int | None]:
    """Convert an ORT graph node into a trtkit tensor spec.

    Args:
        node: ``onnxruntime.NodeArg`` from ``get_inputs()``/``get_outputs()``.

    Returns:
        The tensor spec (per-sample shape, torch dtype) and the static batch
        size when the leading dimension is a fixed integer (else ``None``).

    Raises:
        ValueError: If a non-batch dimension is symbolic or the dtype is unsupported.
    """
    dims: list[Any] = list(node.shape)
    per_sample: list[int] = []
    for dim in dims[1:]:
        if not isinstance(dim, int):
            raise ValueError(f"Graph tensor {node.name!r} has symbolic non-batch dim {dim!r}; trtkit requires static per-sample shapes.")
        per_sample.append(int(dim))
    dtype: torch.dtype | None = _ORT_TO_TORCH_DTYPE.get(str(node.type))
    if dtype is None:
        raise ValueError(f"Unsupported ONNX tensor type {node.type!r} for {node.name!r}.")
    static_batch: int | None = int(dims[0]) if dims and isinstance(dims[0], int) else None
    return TensorSpec(name=str(node.name), shape=tuple(per_sample), dtype=dtype), static_batch
