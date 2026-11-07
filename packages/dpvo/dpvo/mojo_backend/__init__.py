"""DPVO's correlation and bundle-adjustment kernels in Mojo: one source for CPU and CUDA.

`corr` and `ba` implement the `CorrKernels` / `BAKernels` protocols of `dpvo.kernels` with the
function names, argument order, shapes and dtypes of DPVO's former CUDA extensions (forward passes:
the training-only backward passes are not ported). The Mojo extension `_dpvo_mojo` is built by the `dpvo-mojo-build` task from
`mojo/*.mojo`, against lietorch-mojo's runtimes (`-I <lietorch-mojo>/lietorch_mojo/mojo`).

On CUDA the kernels run on PyTorch's current stream. Parity with the former CUDA extensions on real
DPVO inputs: `tests/test_mojo_backend.py` against `tests/fixtures/kernel_references.rrd`.

Importing the package registers it with `dpvo.kernels` for CPU and CUDA tensors; `dpvo.kernels` does
that import on its first lookup.
"""

from dpvo import kernels
from dpvo.mojo_backend import ba, corr


def register() -> None:
    """Serve `dpvo.kernels` lookups for CPU and CUDA tensors (idempotent)."""
    for device_type in ("cpu", "cuda"):
        kernels.register(device_type, corr=corr, ba=ba)


register()
