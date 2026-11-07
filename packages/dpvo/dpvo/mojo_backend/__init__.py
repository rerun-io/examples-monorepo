"""DPVO's correlation and bundle-adjustment kernels in Mojo: one source for CPU and CUDA.

`corr` and `ba` are drop-ins for the CUDA extensions `dpvo._cuda_corr` and `dpvo._cuda_ba` (the
`CorrKernels` / `BAKernels` protocols of `dpvo.kernels`): same function names, argument order,
shapes and dtypes. The Mojo extension `_dpvo_mojo` is built by the `dpvo-mojo-build` task from
`mojo/*.mojo`, against lietorch-mojo's runtimes (`-I <lietorch-mojo>/lietorch_mojo/mojo`).

On CUDA the kernels run on PyTorch's current stream like the extension. Parity with the CUDA
extension on real DPVO inputs: `tests/test_mojo_backend.py` against `tests/fixtures/kernel_references.rrd`.

Importing the package registers it with `dpvo.kernels` for CPU tensors; `dpvo.kernels` does that import on
its first CPU lookup. CUDA stays on the CUDA extension.
"""

from dpvo import kernels
from dpvo.mojo_backend import ba, corr


def register() -> None:
    """Serve `dpvo.kernels` lookups for CPU tensors (idempotent)."""
    kernels.register("cpu", corr=corr, ba=ba)


register()
