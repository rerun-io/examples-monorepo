"""Small TensorRT runners used by the optimized WiLor video path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import torch
from jaxtyping import Float
from torch import Tensor

FULL_WILOR_INPUT_NAME: str = "img_patches"
FULL_WILOR_OUTPUT_NAMES: tuple[str, ...] = ("global_orient", "hand_pose", "betas", "pred_cam", "pred_keypoints_3d", "pred_vertices")
FULL_WILOR_INPUT_SHAPE: tuple[int, int, int] = (256, 256, 3)
DETECTOR_INPUT_NAME: str = "images"
DETECTOR_OUTPUT_NAME: str = "output0"
DETECTOR_INPUT_SHAPE: tuple[int, int, int] = (3, 512, 416)


class WiLorOutput(TypedDict):
    """TensorRT full-WiLor output tensors returned before CPU/Rerun conversion."""

    global_orient: Float[Tensor, "batch 1 3"]
    hand_pose: Float[Tensor, "batch 15 3"]
    betas: Float[Tensor, "batch 10"]
    pred_cam: Float[Tensor, "batch 3"]
    pred_keypoints_3d: Float[Tensor, "batch 21 3"]
    pred_vertices: Float[Tensor, "batch 778 3"]


class _StaticTensorRtRunner:
    """Static-batch TensorRT runner with reusable padded input/output buffers."""

    def __init__(self, engine_path: Path, *, input_name: str, output_names: tuple[str, ...], static_input_shape: tuple[int, ...], static_batch_size: int) -> None:
        """Load a machine-local TensorRT engine and bind named IO tensors.

        Args:
            engine_path: Path to the TensorRT engine file built for the current
                machine/GPU.
            input_name: Expected TensorRT input tensor name.
            output_names: Expected TensorRT output tensor names.
            static_input_shape: Per-sample input shape, excluding batch.
            static_batch_size: Batch size baked into the static engine.

        Raises:
            ValueError: If CUDA is unavailable or ``static_batch_size`` is not
                positive.
            RuntimeError: If the engine cannot be deserialized or does not have
                the requested input/output tensors.
        """
        if not torch.cuda.is_available() or static_batch_size <= 0:
            raise ValueError("TensorRT requires CUDA and a positive static batch size.")
        trt: Any = _import_tensorrt()
        engine: Any = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_path.expanduser().read_bytes())
        context: Any = None if engine is None else engine.create_execution_context()
        if engine is None or context is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        self._trt: Any = trt
        self._engine: Any = engine
        self._context: Any = context
        self._input_name: str = _tensor_name(engine, trt, input_name, input=True)
        self._output_names: tuple[str, ...] = tuple(_tensor_name(engine, trt, name, input=False) for name in output_names)
        self._static_batch_size: int = static_batch_size
        self._static_input_shape: tuple[int, ...] = static_input_shape
        self._stream: torch.cuda.Stream | None = None
        self._input_buffer: Tensor | None = None
        self._outputs: dict[str, Tensor] | None = None

    def run(self, inputs: Tensor) -> tuple[int, dict[str, Tensor]]:
        """Execute TensorRT with static-batch padding when needed.

        Args:
            inputs: CUDA tensor whose first dimension is less than or equal to
                the engine's static batch size.

        Returns:
            A tuple containing the original unpadded batch size and the reusable
            output buffer dictionary. Callers slice outputs back to the returned
            batch size.
        """
        padded: Tensor = self._prepare_input(inputs)
        if self._outputs is None:
            self._outputs = self._allocate_outputs(padded)
        self._context.set_tensor_address(self._input_name, int(cast(Any, padded).data_ptr()))
        cast(Any, padded).record_stream(self._stream_for(padded.device))
        self._execute(padded.device)
        return int(inputs.shape[0]), self._outputs

    def _prepare_input(self, inputs: Tensor) -> Tensor:
        """Return a static-batch input tensor compatible with the engine.

        Args:
            inputs: Runtime input tensor. Its shape after the batch dimension
                must match ``self._static_input_shape``.

        Returns:
            The original contiguous input if it already has the static batch
            size, otherwise a reusable padded tensor with zeros after the real
            batch.

        Raises:
            ValueError: If the input shape is incompatible with the engine or
                the runtime batch is larger than the static batch size.
        """
        shape: tuple[int, ...] = tuple(int(dim) for dim in inputs.shape)
        batch_size: int = shape[0]
        if shape[1:] != self._static_input_shape:
            raise ValueError(f"Expected input shape (*, {self._static_input_shape}), got {shape}.")
        if batch_size > self._static_batch_size:
            raise ValueError(f"Static TensorRT batch is {self._static_batch_size}, got {batch_size}.")
        if batch_size == self._static_batch_size:
            return inputs.contiguous()
        static_shape: tuple[int, ...] = (self._static_batch_size, *self._static_input_shape)
        if self._input_buffer is None or tuple(int(dim) for dim in self._input_buffer.shape) != static_shape or self._input_buffer.dtype != inputs.dtype:
            self._input_buffer = torch.empty(static_shape, dtype=inputs.dtype, device=inputs.device)
        self._input_buffer[:batch_size].copy_(inputs)
        self._input_buffer[batch_size:] = 0
        return self._input_buffer

    def _allocate_outputs(self, example_inputs: Tensor) -> dict[str, Tensor]:
        """Allocate reusable output buffers and bind their TensorRT addresses.

        Args:
            example_inputs: Static-batch input tensor used to set dynamic input
                shape metadata if the engine exposes dynamic dimensions.

        Returns:
            Mapping from output tensor name to CUDA output buffer.
        """
        if any(int(dim) < 0 for dim in self._engine.get_tensor_shape(self._input_name)):
            self._context.set_input_shape(self._input_name, tuple(int(dim) for dim in example_inputs.shape))
        outputs: dict[str, Tensor] = {}
        for name in self._output_names:
            shape: tuple[int, ...] = tuple(int(dim) for dim in self._context.get_tensor_shape(name))
            output: Tensor = torch.empty(shape, dtype=_torch_dtype(self._engine.get_tensor_dtype(name), self._trt), device=example_inputs.device)
            self._context.set_tensor_address(name, int(cast(Any, output).data_ptr()))
            outputs[name] = output
        return outputs

    def _stream_for(self, device: torch.device) -> torch.cuda.Stream:
        """Return the lazily created CUDA stream used for TensorRT execution.

        Args:
            device: CUDA device for the input/output tensors.

        Returns:
            A CUDA stream reused by this runner instance.
        """
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        return self._stream

    def _execute(self, device: torch.device) -> None:
        """Run the TensorRT context and synchronize with PyTorch's current stream.

        Args:
            device: CUDA device where execution should be scheduled.

        Raises:
            RuntimeError: If TensorRT reports a failed ``execute_async_v3`` call.
        """
        stream: torch.cuda.Stream = self._stream_for(device)
        current: torch.cuda.Stream = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            ok: bool = bool(self._context.execute_async_v3(stream_handle=int(stream.cuda_stream)))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        current.wait_stream(stream)


class TensorRtFullWilorRunner:
    """Callable wrapper for the static-batch full WiLor TensorRT engine."""

    def __init__(self, engine_path: Path, *, static_batch_size: int, device: Literal["cuda"] = "cuda") -> None:
        """Create a full-WiLor TensorRT runner.

        Args:
            engine_path: Path to the static full-WiLor TensorRT engine.
            static_batch_size: Batch size baked into the engine.
            device: Execution device. Only ``"cuda"`` is supported.

        Raises:
            ValueError: If ``device`` is not ``"cuda"``.
        """
        if device != "cuda":
            raise ValueError("Full WiLor TensorRT inference requires device='cuda'.")
        self._runner = _StaticTensorRtRunner(
            engine_path,
            input_name=FULL_WILOR_INPUT_NAME,
            output_names=FULL_WILOR_OUTPUT_NAMES,
            static_input_shape=FULL_WILOR_INPUT_SHAPE,
            static_batch_size=static_batch_size,
        )

    def __call__(self, inputs: Float[Tensor, "batch 256 256 3"]) -> WiLorOutput:
        """Run full-WiLor inference on NHWC crop tensors.

        Args:
            inputs: CUDA float16 crops as ``Float[Tensor, "batch 256 256 3"]``.

        Returns:
            A ``WiLorOutput`` dictionary sliced back to the original unpadded
            batch size.

        Raises:
            ValueError: If ``inputs`` is not a CUDA float16 tensor.
        """
        if inputs.device.type != "cuda" or inputs.dtype != torch.float16:
            raise ValueError("Full WiLor TensorRT expects CUDA float16 NHWC inputs.")
        batch_size, outputs = self._runner.run(inputs)
        return cast(WiLorOutput, {name: outputs[name][:batch_size] for name in FULL_WILOR_OUTPUT_NAMES})


class TensorRtRawDetectorRunner:
    """Callable wrapper for the static-batch raw YOLO detector TensorRT engine."""

    def __init__(self, engine_path: Path, *, static_batch_size: int, device: Literal["cuda"] = "cuda") -> None:
        """Create a raw detector TensorRT runner.

        Args:
            engine_path: Path to the static detector TensorRT engine.
            static_batch_size: Batch size baked into the engine.
            device: Execution device. Only ``"cuda"`` is supported.

        Raises:
            ValueError: If ``device`` is not ``"cuda"``.
        """
        if device != "cuda":
            raise ValueError("Raw detector TensorRT inference requires device='cuda'.")
        self._runner = _StaticTensorRtRunner(
            engine_path,
            input_name=DETECTOR_INPUT_NAME,
            output_names=(DETECTOR_OUTPUT_NAME,),
            static_input_shape=DETECTOR_INPUT_SHAPE,
            static_batch_size=static_batch_size,
        )

    def __call__(self, inputs: Float[Tensor, "batch 3 512 416"]) -> Tensor:
        """Run raw detector inference on NCHW detector tensors.

        Args:
            inputs: CUDA float32 detector input as
                ``Float[Tensor, "batch 3 512 416"]``.

        Returns:
            Raw detector output tensor sliced back to the original unpadded batch
            size.

        Raises:
            ValueError: If ``inputs`` is not a CUDA float32 tensor.
        """
        if inputs.device.type != "cuda" or inputs.dtype != torch.float32:
            raise ValueError("Raw detector TensorRT expects CUDA float32 NCHW inputs.")
        batch_size, outputs = self._runner.run(inputs)
        return outputs[DETECTOR_OUTPUT_NAME][:batch_size]


def _tensor_name(engine: Any, trt: Any, requested: str, *, input: bool) -> str:
    """Resolve and validate a named TensorRT input or output tensor.

    Args:
        engine: Deserialized TensorRT engine.
        trt: Imported TensorRT module.
        requested: Tensor name expected by the WiLor runtime wrapper.
        input: Whether to look for an input tensor instead of an output tensor.

    Returns:
        The matching TensorRT tensor name.

    Raises:
        RuntimeError: If the engine does not expose the requested tensor with
            the expected IO mode.
    """
    mode: Any = trt.TensorIOMode.INPUT if input else trt.TensorIOMode.OUTPUT
    for idx in range(int(engine.num_io_tensors)):
        name: str = str(engine.get_tensor_name(idx))
        if name == requested and engine.get_tensor_mode(name) == mode:
            return name
    raise RuntimeError(f"TensorRT engine has no {'input' if input else 'output'} tensor named {requested!r}.")


def _torch_dtype(dtype: Any, trt: Any) -> torch.dtype:
    """Map TensorRT dtypes to PyTorch dtypes.

    Args:
        dtype: TensorRT dtype object returned by the engine.
        trt: Imported TensorRT module.

    Returns:
        Equivalent ``torch.dtype``.

    Raises:
        TypeError: If the dtype is not supported by this runner.
    """
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if hasattr(trt, "bfloat16") and dtype == trt.bfloat16:
        return torch.bfloat16
    raise TypeError(f"Unsupported TensorRT output dtype: {dtype}")


def _import_tensorrt() -> Any:
    """Import TensorRT lazily so non-TRT tests can import this module.

    Returns:
        Imported TensorRT Python module.

    Raises:
        RuntimeError: If TensorRT bindings are not installed in the active Pixi
            environment.
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are not installed in this Pixi environment.") from exc
    return trt
