"""The shared ONNX export recipe for strongly-typed TensorRT builds.

TensorRT 11 removed weak typing, so an engine's compute dtype is whatever the
ONNX graph says. This module owns the one recipe every export in the monorepo
follows: keep the I/O contract fp32, trace the model under autocast so the
low-precision compute (and its fp32 islands) is baked into the graph, pick the
opset the dtype requires, and publish atomically. Model packages pass their
network — or a thin adapter that shapes outputs (flatten a dict, add
fusion-breaker outputs) — and nothing else.
"""

import os
from collections.abc import Callable
from pathlib import Path

import torch

ExportFn = Callable[..., object]


class _AutocastWrapper(torch.nn.Module):
    """fp32 I/O boundary with autocast compute for strongly-typed graphs.

    Floating-point outputs are cast back to fp32 so every consumer (runtime
    buffers, parity harnesses) sees the same contract as the eager model.
    """

    def __init__(self, inner: torch.nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.inner = inner
        self.dtype = dtype

    def forward(self, *inputs: torch.Tensor):
        with torch.autocast("cuda", dtype=self.dtype):
            outputs = self.inner(*inputs)
        if isinstance(outputs, tuple):
            return tuple(o.float() if isinstance(o, torch.Tensor) and o.is_floating_point() else o for o in outputs)
        return outputs.float()


def export_onnx(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    out_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    compute_dtype: torch.dtype | None = None,
    dynamic_batch_max: int | None = None,
    export_fn: ExportFn = torch.onnx.export,
) -> None:
    """Export a model to ONNX with the strongly-typed-TRT recipe.

    Args:
        model: Network (or output-shaping adapter around it) in eval mode.
        example_inputs: Positional example tensors, traced as given.
        out_path: Final ONNX path, published atomically (pid-unique temp +
            rename) so a killed export can never leave a truncated file that
            later runs silently reuse.
        input_names: ONNX input names, matching ``example_inputs`` order.
        output_names: ONNX output names, matching the model's output order.
        compute_dtype: ``torch.float16``/``torch.bfloat16`` traces the model
            under autocast with fp32 I/O boundaries; ``None`` exports the
            model's own dtypes unchanged.
        dynamic_batch_max: When set, dim 0 of every input is dynamic in
            ``1..dynamic_batch_max``; ``None`` bakes the example batch in.
        export_fn: ``torch.onnx.export``-compatible hook for tests.

    Raises:
        ValueError: If ``compute_dtype`` is not fp16/bf16/None.
    """
    if compute_dtype not in (None, torch.float16, torch.bfloat16):
        raise ValueError(f"compute_dtype must be float16, bfloat16, or None, got {compute_dtype}.")
    # bf16 Conv/ConvTranspose only exist in ONNX from opset 22; below that,
    # TRT parsers accept the invalid graph and silently miscompile it. 18 is
    # the dynamo baseline for everything else; TRT 11 parses up to 24.
    opset_version: int = 23 if compute_dtype == torch.bfloat16 else 18
    export_model: torch.nn.Module = _AutocastWrapper(model, compute_dtype).eval() if compute_dtype is not None else model

    kwargs: dict[str, object] = {}
    if dynamic_batch_max is not None:
        batch_dim = torch.export.Dim("batch", min=1, max=dynamic_batch_max)
        # Positional form: keys by arg name break on adapters with different
        # parameter names, and dynamic_axes with dynamo=True is a lossy path.
        per_input = tuple({0: batch_dim} for _ in example_inputs)
        # The autocast wrapper's forward is ``*inputs``, so torch.export sees
        # ONE varargs parameter holding the tuple — mirror that nesting.
        kwargs["dynamic_shapes"] = (per_input,) if compute_dtype is not None else per_input

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path = out_path.with_name(f"{out_path.name}.part{os.getpid()}")
    with torch.inference_mode():
        export_fn(
            export_model,
            example_inputs,
            str(tmp_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=True,
            **kwargs,
        )
    tmp_path.rename(out_path)
