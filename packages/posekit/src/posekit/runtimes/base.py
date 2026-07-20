"""Backend-neutral tensor-function contract shared by torch/ONNX/TensorRT.

A :class:`TensorRuntime` is a pure batched tensor function: CUDA tensors in, CUDA
tensors out. Everything model-specific (crop generation, normalization, heatmap
or SimCC decode) happens *outside* the runtime in shared torch ops, which is what
lets one model definition run on all three backends with bit-comparable pre- and
postprocessing.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Name, per-sample shape, and dtype of one runtime input or output."""

    name: str
    """Binding name in the underlying graph/engine."""
    shape: tuple[int, ...]
    """Per-sample shape excluding the leading batch dimension."""
    dtype: torch.dtype
    """Tensor dtype expected/produced by the runtime."""


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """Static I/O contract of a loaded runtime."""

    inputs: tuple[TensorSpec, ...]
    """Input bindings in graph order."""
    outputs: tuple[TensorSpec, ...]
    """Output bindings in graph order."""
    max_batch_size: int
    """Largest batch a single call may submit (static engines pad up to this)."""


@runtime_checkable
class TensorRuntime(Protocol):
    """Batched CUDA-tensor function backing one neural network."""

    @property
    def spec(self) -> RuntimeSpec:
        """Static I/O contract of this runtime."""
        ...

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run one batch.

        Args:
            inputs: CUDA tensors keyed by ``spec.inputs`` names. All inputs share
                the same leading batch size, which must be ``<= spec.max_batch_size``.

        Returns:
            CUDA tensors keyed by ``spec.outputs`` names, sliced to the submitted
            batch size.
        """
        ...


def run_chunked(runtime: TensorRuntime, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Run a batch of any size through a runtime, chunking to ``max_batch_size``.

    This owns the buffer-safety contract every caller would otherwise have to
    remember: ONNX/TensorRT runtimes hand back views of persistent buffers, so
    each chunk's outputs are cloned before the next chunk overwrites them.

    Args:
        runtime: Backend runtime to invoke.
        inputs: Full-batch input tensors keyed by ``spec.inputs`` names.

    Returns:
        Full-batch output tensors keyed by ``spec.outputs`` names.
    """
    batch_size: int = int(next(iter(inputs.values())).shape[0])
    chunk: int = runtime.spec.max_batch_size
    chunk_outputs: list[dict[str, Tensor]] = []
    for start in range(0, batch_size, chunk):
        outputs: dict[str, Tensor] = runtime({name: tensor[start : start + chunk] for name, tensor in inputs.items()})
        chunk_outputs.append({name: value.clone() for name, value in outputs.items()})
    if len(chunk_outputs) == 1:
        return chunk_outputs[0]
    return {name: torch.cat([outputs[name] for outputs in chunk_outputs], dim=0) for name in chunk_outputs[0]}


def validate_runtime_inputs(spec: RuntimeSpec, inputs: dict[str, Tensor]) -> int:
    """Validate a runtime input dict against a spec and return the batch size.

    Args:
        spec: Runtime I/O contract to validate against.
        inputs: Candidate input tensors keyed by binding name.

    Returns:
        The shared leading batch size of all inputs.

    Raises:
        ValueError: If names, per-sample shapes, or batch sizes are inconsistent
            with the spec, or the batch exceeds ``spec.max_batch_size``.
    """
    expected_names: tuple[str, ...] = tuple(tensor_spec.name for tensor_spec in spec.inputs)
    if tuple(sorted(inputs)) != tuple(sorted(expected_names)):
        raise ValueError(f"Runtime expects inputs {expected_names}, got {tuple(inputs)}.")
    batch_sizes: set[int] = set()
    for tensor_spec in spec.inputs:
        tensor: Tensor = inputs[tensor_spec.name]
        shape: tuple[int, ...] = tuple(int(dim) for dim in tensor.shape)
        if shape[1:] != tensor_spec.shape:
            raise ValueError(f"Input {tensor_spec.name!r} expects per-sample shape {tensor_spec.shape}, got {shape[1:]}.")
        batch_sizes.add(shape[0])
    if len(batch_sizes) != 1:
        raise ValueError(f"All runtime inputs must share one batch size, got {sorted(batch_sizes)}.")
    batch_size: int = batch_sizes.pop()
    if batch_size > spec.max_batch_size:
        raise ValueError(f"Runtime max batch is {spec.max_batch_size}, got {batch_size} — chunk upstream.")
    return batch_size
