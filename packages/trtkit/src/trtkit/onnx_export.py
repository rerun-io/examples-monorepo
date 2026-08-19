"""The shared ONNX export recipe for strongly-typed TensorRT builds.

TensorRT 11 removed weak typing, so an engine's compute dtype is whatever the
ONNX graph says. This module owns the one recipe every export in the monorepo
follows: keep the I/O contract fp32, trace the model under autocast so the
low-precision compute (and its fp32 islands) is baked into the graph, pick the
opset the dtype requires, and publish atomically. Model packages pass their
network — or a thin adapter that shapes outputs (flatten a dict, add
fusion-breaker outputs) — and nothing else.
"""

import copy
import os
import shutil
import time
from collections.abc import Callable, Collection
from pathlib import Path

import torch
from torch.nn.modules.conv import _ConvTransposeNd

ExportFn = Callable[..., object]


class _Fp32Island(torch.nn.Module):
    """Runs its inner module in fp32 inside an autocast region.

    TensorRT's strongly-typed builds cannot type a BF16 ConvTranspose — an
    open type-inference-rule gap (NVIDIA/TensorRT-Incubator#565, verified
    failing on TRT 11.2.1.2) — so bf16 exports keep transposed convolutions
    fp32. Weak typing made this exact fallback implicitly before TRT 11
    removed it. Retest on TRT bumps; delete when the upstream bug is fixed.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            return self.inner(*(t.float() if t.is_floating_point() else t for t in inputs))


def _with_fp32_transposed_convs(module: torch.nn.Module) -> torch.nn.Module:
    """Return a structural copy whose transposed convs sit in fp32 islands.

    Parameters and buffers are shared with the original; the caller's module
    tree is never mutated, so eager parity references stay honest.
    """
    if isinstance(module, _Fp32Island):
        return module
    if isinstance(module, _ConvTransposeNd):
        return _Fp32Island(module)
    replaced: dict[str, torch.nn.Module] = {
        name: _with_fp32_transposed_convs(child) for name, child in module.named_children()
    }
    if all(replaced[name] is child for name, child in module.named_children()):
        return module
    clone: torch.nn.Module = copy.copy(module)
    clone._modules = dict(module._modules)
    clone._modules.update(replaced)
    return clone


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


def sweep_stale_onnx_exports(
    directory: Path,
    filename_prefix: str,
    *,
    keep_paths: Collection[Path],
    partial_grace_seconds: float = 3600.0,
) -> list[Path]:
    """Remove obsolete exports while preserving current and in-flight files.

    Args:
        directory: Directory containing model-specific ONNX exports.
        filename_prefix: Prefix shared only by versions of one model shape.
        keep_paths: Complete export and sidecar paths still in use.
        partial_grace_seconds: Minimum age before a ``.part`` file can be
            treated as abandoned.

    Returns:
        Removed paths in deterministic filename order.
    """
    keep: set[Path] = set(keep_paths)
    now: float = time.time()
    removed: list[Path] = []
    for path in sorted(directory.iterdir()):
        if path in keep or not path.name.startswith(filename_prefix):
            continue
        if ".part" in path.name and now - path.stat().st_mtime < partial_grace_seconds:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        removed.append(path)
    return removed


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
        out_path: Final ONNX path, published atomically: the export writes into
            a pid-unique temp directory under its FINAL filename, then both the
            protobuf and any external-data sidecar (``<name>.data``, written by
            dynamo when weights exceed the 2 GB protobuf limit) move into
            place. A killed export can never leave a truncated file that later
            runs silently reuse, and the sidecar keeps a deterministic name the
            published graph references correctly.
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
    # bf16 also forces fp32 islands around transposed convs (see _Fp32Island);
    # derived from the dtype, like the opset — not a caller knob.
    inner: torch.nn.Module = _with_fp32_transposed_convs(model) if compute_dtype == torch.bfloat16 else model
    export_model: torch.nn.Module = _AutocastWrapper(inner, compute_dtype).eval() if compute_dtype is not None else model

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
    # Export into a pid-unique temp DIRECTORY under the final filename, so the
    # external-data sidecar dynamo writes for >2 GB weights is born with the
    # deterministic name the protobuf references (`<name>.data`), instead of a
    # permanent pid-suffixed temp name the caller would have to know about.
    tmp_dir: Path = out_path.with_name(f"{out_path.name}.part{os.getpid()}")
    tmp_dir.mkdir()
    tmp_path: Path = tmp_dir / out_path.name
    try:
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
        # Sidecar first: the protobuf must never be visible while the data it
        # references is missing.
        tmp_sidecar: Path = tmp_dir / f"{out_path.name}.data"
        if tmp_sidecar.exists():
            os.replace(tmp_sidecar, out_path.with_name(tmp_sidecar.name))
        os.replace(tmp_path, out_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
