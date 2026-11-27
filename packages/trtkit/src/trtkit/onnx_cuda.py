"""ONNX Runtime backend (CUDA or CPU provider) with IOBinding onto torch tensor memory.

Inputs and outputs are bound directly to torch tensor ``data_ptr``s so no
numpy/host copies happen anywhere in the hot path — the gap that makes the
stock rtmlib ONNX path slow. On CUDA, the session is attached to a torch
stream (``user_compute_stream``) when available so execution is
stream-ordered with surrounding torch preprocessing/decode kernels. On the
CPU, the same contract runs on CPU tensors: engine-free tests and hosts
without NVIDIA run the very artifact a TensorRT engine is built from.
"""

import functools
import sys
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
from torch import Tensor

from trtkit.base import RuntimeSpec, TensorSpec, validate_runtime_inputs

# Beartype runs only under pytest (pytest-beartype imports it); its violations must escape broad handlers then.
BeartypeException: tuple[type[Exception], ...] = (sys.modules["beartype.roar"].BeartypeException,) if "beartype.roar" in sys.modules else ()

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


def _import_ort() -> Any:
    """The ``onnxruntime`` module, with an error that names the env.

    Raises:
        RuntimeError: If onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed in this Pixi environment.") from exc
    return ort


def _session_options(ort: Any) -> Any:
    """The session options every trtkit ONNX Runtime session takes."""
    options: Any = ort.SessionOptions()
    options.log_severity_level = 3
    return options


def _torch_dtype(node: Any) -> torch.dtype:
    """The torch dtype of an ``onnxruntime.NodeArg``.

    Raises:
        ValueError: If the dtype is unsupported.
    """
    dtype: torch.dtype | None = _ORT_TO_TORCH_DTYPE.get(str(node.type))
    if dtype is None:
        raise ValueError(f"Unsupported ONNX tensor type {node.type!r} for {node.name!r}.")
    return dtype


@functools.cache
def require_cuda_kernels(device_id: int) -> None:
    """Run one Relu on ONNX Runtime's CUDA provider, once per device and process (a failure raises and is not cached).

    Listing the provider is not enough: conda-forge's CUDA 13 build registers on the GB10 (sm_121) and creates
    sessions there, but has no kernel image for it, so only the first inference fails.

    Args:
        device_id: CUDA device ordinal.

    Raises:
        RuntimeError: If onnxruntime or its CUDA provider is missing, or the provider does not take the session or
            cannot run the kernel.
    """
    ort: Any = _import_ort()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("onnxruntime CUDAExecutionProvider is unavailable: the env needs the cuda feature's onnxruntime build.")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "cuda_kernel_probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
    )
    model: bytes = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10).SerializeToString()
    try:
        session: Any = ort.InferenceSession(
            model, sess_options=_session_options(ort), providers=[("CUDAExecutionProvider", {"device_id": device_id})]
        )
        # ORT falls back to the CPU provider when the CUDA one cannot be created; a CPU run would prove nothing.
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"the session fell back to {session.get_providers()}")
        session.run(None, {"x": np.zeros(4, dtype=np.float32)})
    except BeartypeException:
        raise
    except Exception as error:
        raise RuntimeError(
            f"ONNX Runtime's CUDA provider cannot run a kernel on CUDA device {device_id} ({error}): run TensorRtBackendConfig "
            "on this GPU, or ONNX Runtime on the CPU."
        ) from error


def _open_cuda_session(ort: Any, onnx_path: Path, session_options: Any, device: torch.device) -> tuple[Any, torch.cuda.Stream | None]:
    """Create the CUDA session, stream-ordered on a torch side stream when ORT accepts one.

    Returns:
        The session, and the stream it is ordered on (``None`` for the host-synchronized fallback).

    Raises:
        RuntimeError: If the CUDA provider is unavailable or cannot run a kernel on this GPU.
    """
    device_id: int = device.index or 0
    require_cuda_kernels(device_id)
    # Dedicated session stream, NEVER torch's current stream: the default
    # stream's raw handle is 0, which ORT parses as "no user stream" and
    # silently runs on its own non-blocking internal stream — unfenced
    # against torch, so inputs race (intermittent garbage inferences). A
    # torch-owned side stream has a real handle ORT honors, and __call__
    # fences it against the caller's stream on both sides of the run.
    torch_stream: torch.cuda.Stream = torch.cuda.Stream(device=device)
    provider_options: dict[str, Any] = {"device_id": device_id, "user_compute_stream": str(int(torch_stream.cuda_stream))}
    try:
        return ort.InferenceSession(
            str(onnx_path), sess_options=session_options, providers=[("CUDAExecutionProvider", provider_options)]
        ), torch_stream
    except BeartypeException:
        raise
    except Exception as error:
        # Perf-relevant mode switch: every call will host-sync the caller's
        # stream instead of being stream-ordered — say so, don't hide it.
        print(f"[trtkit] ORT user_compute_stream unavailable ({error}); falling back to a host-synchronized session.")
        return ort.InferenceSession(
            str(onnx_path), sess_options=session_options, providers=[("CUDAExecutionProvider", {"device_id": device_id})]
        ), None


OrtDevice: TypeAlias = Literal["cuda", "cpu"]
"""Which ONNX Runtime execution provider a session runs on."""


class OnnxRuntime:
    """ONNX Runtime session implementing the trtkit runtime contract on CUDA or CPU tensors."""

    def __init__(self, onnx_path: Path, *, device: OrtDevice = "cuda", device_id: int = 0, max_batch_size: int = 32) -> None:
        """Load an ONNX graph into a CUDA or CPU execution-provider session.

        Args:
            onnx_path: Path to the ONNX model file.
            device: ``cuda`` (GPU-resident, stream-ordered) or ``cpu``.
            device_id: CUDA device ordinal for the execution provider; unused on the CPU.
            max_batch_size: Batch cap for graphs with a dynamic batch dimension.
                Graphs with a static batch dimension use that value instead and
                pad smaller batches up to it.

        Raises:
            RuntimeError: If onnxruntime or its CUDA provider is unavailable.
            ValueError: If the graph has symbolic non-batch dimensions.
        """
        ort: Any = _import_ort()
        session_options: Any = _session_options(ort)
        self._device: torch.device = torch.device("cuda", device_id) if device == "cuda" else torch.device("cpu")
        self._session: Any
        # None on the CPU and on the host-synchronized CUDA fallback; else the stream the session is ordered on.
        self._torch_stream: torch.cuda.Stream | None
        if device == "cpu":
            self._session, self._torch_stream = (
                ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"]),
                None,
            )
        else:
            self._session, self._torch_stream = _open_cuda_session(ort, onnx_path, session_options, self._device)
        spec_and_batch: tuple[RuntimeSpec, int | None] = self._read_spec(max_batch_size)
        self._spec: RuntimeSpec = spec_and_batch[0]
        self._static_batch: int | None = spec_and_batch[1]
        self._input_buffers: dict[str, Tensor] = {}
        self._output_buffers: dict[str, Tensor] | None = None

    @property
    def device(self) -> torch.device:
        """The torch device inputs must live on and outputs are returned on."""
        return self._device

    @property
    def _bound_on(self) -> dict[str, Any]:
        """IOBinding's ``device_type`` / ``device_id`` for this runtime's tensors."""
        return {"device_type": self._device.type, "device_id": self._device.index or 0}

    def _read_spec(self, max_batch_size: int) -> tuple[RuntimeSpec, int | None]:
        """The I/O contract read from the session, and the graph's static batch (``None`` when dynamic)."""
        static_batch: int | None = None
        input_specs: list[TensorSpec] = []
        for node in self._session.get_inputs():
            spec, node_batch = _tensor_spec_from_node(node)
            input_specs.append(spec)
            static_batch = node_batch if node_batch is not None else static_batch
        output_specs: list[TensorSpec] = self._resolve_output_specs(tuple(input_specs), static_batch)
        runtime_spec: RuntimeSpec = RuntimeSpec(
            inputs=tuple(input_specs),
            outputs=tuple(output_specs),
            max_batch_size=static_batch if static_batch is not None else max_batch_size,
        )
        return runtime_spec, static_batch

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
                **self._bound_on,
                element_type=_TORCH_TO_NUMPY_DTYPE[tensor_spec.dtype],
                shape=tuple(int(dim) for dim in dummy.shape),
                buffer_ptr=int(dummy.data_ptr()),
            )
        for node in output_nodes:
            binding.bind_output(node.name, **self._bound_on)
        self._session.run_with_iobinding(binding)
        specs: list[TensorSpec] = []
        for node, ort_value in zip(output_nodes, binding.get_outputs(), strict=True):
            shape: tuple[int, ...] = tuple(int(dim) for dim in ort_value.shape())
            specs.append(TensorSpec(name=str(node.name), shape=shape[1:], dtype=_torch_dtype(node)))
        return specs

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run one batch through ONNX Runtime with zero host copies.

        Args:
            inputs: Tensors on :attr:`device` keyed by ``spec.inputs`` names.

        Returns:
            Tensors on :attr:`device` keyed by ``spec.outputs`` names, sliced to the submitted
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
                **self._bound_on,
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
                **self._bound_on,
                element_type=_TORCH_TO_NUMPY_DTYPE[tensor_spec.dtype],
                shape=tuple(int(dim) for dim in output.shape),
                buffer_ptr=int(output.data_ptr()),
            )
        if self._device.type == "cpu":
            self._session.run_with_iobinding(binding)
        elif self._torch_stream is not None:
            # The session executes on the construction-time stream; when the
            # caller is on a different stream, order the two on both sides of
            # the run so ORT never reads half-written inputs and torch never
            # reads stale outputs.
            current_stream: torch.cuda.Stream = torch.cuda.current_stream(self._device)
            if current_stream != self._torch_stream:
                self._torch_stream.wait_stream(current_stream)
            self._session.run_with_iobinding(binding)
            if current_stream != self._torch_stream:
                current_stream.wait_stream(self._torch_stream)
        else:
            # The host-synchronized fallback runs on ORT's own internal stream: drain the caller's first.
            torch.cuda.current_stream(self._device).synchronize()
            self._session.run_with_iobinding(binding)
        return {name: buffer[:batch_size] for name, buffer in self._output_buffers.items()}

    def _bound_input(self, tensor_spec: TensorSpec, tensor: Tensor, bound_batch: int) -> Tensor:
        """Return a contiguous tensor on the session's device, padded to the bound batch size.

        Args:
            tensor_spec: Input contract being bound.
            tensor: Caller-provided input for this binding.
            bound_batch: Batch size the session will actually execute.

        Returns:
            The caller tensor when it already matches, otherwise a reusable
            zero-padded staging buffer.

        Raises:
            ValueError: If the tensor is not on the session's device.
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
    static_batch: int | None = int(dims[0]) if dims and isinstance(dims[0], int) else None
    return TensorSpec(name=str(node.name), shape=tuple(per_sample), dtype=_torch_dtype(node)), static_batch


class OnnxDynamicRuntime:
    """A graph on ONNX Runtime's CPU provider whose inputs may change any dimension on every call.

    The CPU counterpart of :class:`trtkit.tensorrt_runtime.TensorRtDynamicRuntime`'s contract: a dict of named CPU
    tensors of the graph's dtypes in (as direct binding takes them: no cast), a dict of named CPU tensors out (fresh
    arrays, safe to keep), ``max_input_shapes`` with ``-1`` for a dynamic dimension, and a context manager. Engine-free
    tests and hosts without NVIDIA run the very ONNX graph the TensorRT engine is built from. Unlike
    :class:`OnnxRuntime`, which lets only the batch change and keeps its buffers bound, nothing here is kept between
    calls.
    """

    def __init__(self, onnx_path: Path) -> None:
        """Load an ONNX graph into a CPU execution-provider session.

        Args:
            onnx_path: Path to the ONNX model file.

        Raises:
            RuntimeError: If onnxruntime is not installed.
            ValueError: If an input has an unsupported dtype.
        """
        ort: Any = _import_ort()
        session_options: Any = _session_options(ort)
        # Shapes change on every call: the CPU arena keeps each new peak until the session closes, which grew one colsfm
        # test run past 64 GB. Allocate per call instead.
        session_options.enable_cpu_mem_arena = False
        self._session: Any = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
        self._input_dtypes: dict[str, torch.dtype] = {str(node.name): _torch_dtype(node) for node in self._session.get_inputs()}
        self._output_names: tuple[str, ...] = tuple(str(node.name) for node in self._session.get_outputs())
        # Each input's declared shape, -1 where the graph leaves a dimension dynamic (the TensorRT runtime's attribute).
        self.max_input_shapes: dict[str, tuple[int, ...]] = {
            str(node.name): tuple(dim if isinstance(dim, int) else -1 for dim in node.shape) for node in self._session.get_inputs()
        }

    @property
    def device(self) -> torch.device:
        """The torch device inputs must live on and outputs are returned on."""
        return torch.device("cpu")

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run once.

        Args:
            inputs: CPU tensors keyed by the graph's input names, in any shape the graph accepts.

        Returns:
            CPU tensors keyed by the graph's output names.

        Raises:
            ValueError: If the names differ from the graph's inputs, or an input is not a CPU tensor of its input's dtype.
        """
        if set(inputs) != set(self._input_dtypes):
            raise ValueError(f"Runtime expects inputs {tuple(self._input_dtypes)}, got {tuple(inputs)}.")
        feeds: dict[str, np.ndarray] = {}
        for name, tensor in inputs.items():
            if tensor.device.type != "cpu" or tensor.dtype != self._input_dtypes[name]:
                raise ValueError(f"Input {name!r} must be a CPU {self._input_dtypes[name]} tensor, got {tensor.dtype} on {tensor.device}.")
            feeds[name] = tensor.detach().contiguous().numpy()
        values: list[np.ndarray] = self._session.run(self._output_names, feeds)
        return {name: torch.from_numpy(value) for name, value in zip(self._output_names, values, strict=True)}

    def __enter__(self) -> "OnnxDynamicRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self._session = None


OnnxCudaRuntime = OnnxRuntime
"""The CUDA spelling: ``OnnxCudaRuntime(path, device_id=...)`` is ``OnnxRuntime(path, device="cuda", device_id=...)``."""
