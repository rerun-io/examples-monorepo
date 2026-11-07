"""The one place that calls the Mojo extension: device dispatch, runtimes and tensor specs.

CPU tensors run the CPU loops on the Mojo thread pool. CUDA tensors run on PyTorch's current
stream, asynchronously, as the CUDA extension does. MPS tensors run the Metal kernels: the call
synchronises PyTorch's MPS queue first, passes Metal GPU addresses, and returns when the kernel is
done (`lietorch_mojo.metal`).

The runtimes are lietorch-mojo's (`lietorch_mojo.backends._runtime`): the Mojo structs can be
registered as Python types by one extension only, so both extensions share one runtime per device.
"""

from collections.abc import Sequence

import torch
from lietorch_mojo.backends import _current_raw_stream, _runtime
from lietorch_mojo.metal import gpu_addresses
from torch import Tensor

from dpvo.mojo_backend import _dpvo_mojo as _ext


def launch(op: str, scalars: tuple[int | float, ...], tensors: Sequence[Tensor]) -> None:
    """Run Mojo op `op` on `tensors` (contiguous, on one device) with the op's `scalars` (see _dpvo_mojo.mojo)."""
    device: torch.device = tensors[0].device
    for t in tensors:
        if t.device != device:
            raise ValueError(f"dpvo.mojo_backend.{op}: tensors on {device} and {t.device}")
        if not t.is_contiguous():
            raise ValueError(f"dpvo.mojo_backend.{op}: tensors must be contiguous")
    match device.type:
        case "cpu":
            specs = tuple((t.data_ptr(), *t.shape) for t in tensors)
            getattr(_ext, f"{op}_cpu")(_runtime(False), (0, *scalars), specs)
        case "cuda":
            if (device.index or 0) != 0:
                raise RuntimeError(f"dpvo.mojo_backend runs on cuda:0 only, got {device}")
            specs = tuple((t.data_ptr(), *t.shape) for t in tensors)
            getattr(_ext, f"{op}_gpu")(_runtime(True), (_current_raw_stream(0), *scalars), specs)
        case "mps":
            torch.mps.synchronize()
            specs = tuple((address, *t.shape) for address, t in zip(gpu_addresses(tensors), tensors, strict=True))
            getattr(_ext, f"{op}_gpu")(_runtime(True), (0, *scalars), specs)
        case _:
            raise RuntimeError(f"dpvo.mojo_backend supports CPU, CUDA and MPS tensors, got {device}")
