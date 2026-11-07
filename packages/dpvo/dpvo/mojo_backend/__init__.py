"""DPVO's correlation and bundle-adjustment kernels in Mojo: one source for CPU, CUDA and Apple GPUs (Metal).

`corr` and `ba` are drop-ins for the CUDA extensions `dpvo._cuda_corr` and `dpvo._cuda_ba` (the
`CorrKernels` / `BAKernels` protocols of `dpvo.kernels`): same function names, argument order,
shapes and dtypes. The Mojo extension `_dpvo_mojo` is built by the `dpvo-mojo-build` task from
`mojo/*.mojo`, against lietorch-mojo's runtimes (`-I <lietorch-mojo>/lietorch_mojo/mojo`).

On CUDA the kernels run on PyTorch's current stream like the extension; on MPS each call
synchronises and passes Metal GPU addresses (`lietorch_mojo.metal`). Parity with the CUDA
extension on real DPVO inputs: `tests/test_mojo_backend.py` against `tests/fixtures/kernel_references.rrd`.
"""
