"""The one place that calls the Mojo extension: device checks, runtimes and tensor specs.

The kernels run on cuda:0 on PyTorch's current stream, asynchronously; the Gauss-Newton pose solve runs on the
CPU. The runtimes are lietorch-mojo's (`lietorch_mojo.backends._runtime`): the Mojo structs can be registered as
Python types by one extension only, so both extensions share one runtime per device.
"""

from collections.abc import Sequence

import torch
from lietorch_mojo.backends import _current_raw_stream, _runtime
from torch import Tensor

from mast3r_slam.mojo_backend import _mast3r_mojo as _ext  # pyrefly: ignore[missing-import]


def launch(op: str, scalars: tuple[int | float, ...], tensors: Sequence[Tensor]) -> object:
    """Run Mojo op `op` on `tensors` (contiguous, on one device) with the op's `scalars` (see _mast3r_mojo.mojo)."""
    device: torch.device = tensors[0].device
    for t in tensors:
        if t.device != device:
            raise ValueError(f"mast3r_slam.mojo_backend.{op}: tensors on {device} and {t.device}")
        if not t.is_contiguous():
            raise ValueError(f"mast3r_slam.mojo_backend.{op}: tensors must be contiguous")
    specs = tuple((t.data_ptr(), *t.shape) for t in tensors)
    match device.type:
        case "cuda":
            if (device.index or 0) != 0:
                raise RuntimeError(f"mast3r_slam.mojo_backend runs on cuda:0 only, got {device}")
            return getattr(_ext, f"{op}_gpu")(_runtime(True), (_current_raw_stream(0), *scalars), specs)
        case "cpu":
            return getattr(_ext, f"{op}_cpu")(_runtime(False), (0, *scalars), specs)
        case _:
            raise RuntimeError(f"mast3r_slam.mojo_backend supports CUDA (and CPU for the pose solve), got {device}")
