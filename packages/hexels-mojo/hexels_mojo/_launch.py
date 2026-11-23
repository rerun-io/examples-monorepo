"""The one place that calls the Mojo extension: device check, runtime and tensor specs.

The kernels run on PyTorch's current CUDA stream, asynchronously, as the CUDA extensions do. The runtime
is lietorch-mojo's (`lietorch_mojo.backends._runtime`): the Mojo structs can be registered as Python
types by one extension only, so all extensions share one runtime.
"""

from collections.abc import Sequence
from functools import cache

import torch
from lietorch_mojo.backends import _current_raw_stream, _runtime
from torch import Tensor

from hexels_mojo import _hexels_mojo as _ext


@cache
def _dummy(device: torch.device) -> Tensor:
    """A non-null device address for empty tensors (Mojo pointers are never null); kernels read no element of them."""
    return torch.zeros(1, device=device)


def launch(op: str, scalars: tuple[int | float | bool, ...], tensors: Sequence[Tensor | None]) -> object:
    """Run Mojo op `op` on `tensors` (contiguous, on the device of the first non-empty one, which must be cuda:0 today) with
    the op's `scalars` (see mojo/_hexels_mojo.mojo); returns the op's result.

    `None` and empty tensors arrive as zero-row tensors."""
    device = next((t.device for t in tensors if t is not None and t.numel() > 0), torch.device("cuda", 0))
    specs: list[tuple[int, ...]] = []
    for t in tensors:
        if t is None or t.numel() == 0:
            specs.append((_dummy(device).data_ptr(), 0))
            continue
        if t.device.type != "cuda":
            raise RuntimeError(
                f"hexels_mojo.{op} needs CUDA tensors, got {t.device}: the kernels run on PyTorch's CUDA stream through"
                " lietorch_mojo's GPU runtime, which has no MPS / ROCm / XPU interop"
            )
        if t.device != device:
            raise ValueError(f"hexels_mojo.{op}: tensors on {device} and {t.device}")
        if (t.device.index or 0) != 0:
            raise RuntimeError(f"hexels_mojo.{op}: lietorch_mojo's GPU runtime is bound to cuda:0, got {t.device}")
        if not t.is_contiguous():
            raise ValueError(f"hexels_mojo.{op}: tensors must be contiguous")
        specs.append((t.data_ptr(), *t.shape))
    return getattr(_ext, op)(_runtime(True), (_current_raw_stream(device.index or 0), *scalars), tuple(specs))
