"""Eager PyTorch backend for the posekit runtime contract."""

from collections.abc import Sequence

import torch
from torch import Tensor

from posekit.runtimes.base import RuntimeSpec, TensorSpec, validate_runtime_inputs


class TorchRuntime:
    """Wraps an ``nn.Module`` as a :class:`posekit.runtimes.base.TensorRuntime`.

    The module is called positionally in ``input_specs`` order and its (tuple of)
    outputs are mapped to ``output_specs`` names, so the same model class can
    swap this for the ONNX/TensorRT runtimes without touching pre/postprocessing.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        input_specs: Sequence[TensorSpec],
        output_specs: Sequence[TensorSpec],
        max_batch_size: int,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        """Create a torch runtime around an eval-mode module.

        Args:
            module: Network to run. Moved to eval mode; the caller picks its device.
            input_specs: Positional input contract (names are for the dict API).
            output_specs: Output names/dtypes; must match the module's return arity.
            max_batch_size: Largest batch a single call may submit.
            autocast_dtype: Optional CUDA autocast dtype (e.g. ``torch.float16``)
                matching the precision a TensorRT engine of this model would use.
        """
        self._module: torch.nn.Module = module.eval()
        self._spec = RuntimeSpec(inputs=tuple(input_specs), outputs=tuple(output_specs), max_batch_size=max_batch_size)
        self._autocast_dtype: torch.dtype | None = autocast_dtype

    @property
    def spec(self) -> RuntimeSpec:
        """Static I/O contract of this runtime."""
        return self._spec

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the wrapped module on one batch.

        Args:
            inputs: CUDA tensors keyed by ``spec.inputs`` names.

        Returns:
            Module outputs keyed by ``spec.outputs`` names, cast to the declared
            output dtypes.
        """
        validate_runtime_inputs(self._spec, inputs)
        ordered: list[Tensor] = [inputs[tensor_spec.name].to(dtype=tensor_spec.dtype) for tensor_spec in self._spec.inputs]
        with torch.inference_mode():
            if self._autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=self._autocast_dtype):
                    raw = self._module(*ordered)
            else:
                raw = self._module(*ordered)
        outputs: tuple[Tensor, ...] = raw if isinstance(raw, tuple) else (raw,)
        if len(outputs) != len(self._spec.outputs):
            raise ValueError(f"Module returned {len(outputs)} outputs, spec declares {len(self._spec.outputs)}.")
        return {tensor_spec.name: tensor.to(dtype=tensor_spec.dtype) for tensor_spec, tensor in zip(self._spec.outputs, outputs, strict=True)}
