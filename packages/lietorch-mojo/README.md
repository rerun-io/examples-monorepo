# lietorch-mojo

lietorch's Lie group API on Mojo kernels. `import lietorch_mojo as lietorch` is
a drop-in for upstream lietorch (VSLAM-LAB/lietorch @ 3e2695dd, the source of
the ai-demos `lietorch 1.0` build 2): same classes, methods, signatures, data
layout (`SE3` = `tx ty tz qx qy qz qw`) and tangent-space gradient conventions.

Status (step 1 of the lietorch → Mojo evaluation):

| Group | Forward | Backward | CPU | CUDA | float32 / float64 |
|-------|---------|----------|-----|------|-------------------|
| SE3   | yes     | yes      | yes | yes  | yes               |
| SO3, RxSO3, Sim3 | API shells; ops raise `NotImplementedError` | | | | |

`lietorch_mojo.backends` is a drop-in for the compiled `lietorch_backends`
module (`expm`, `expm_backward`, `logm`, …, `as_matrix`, `projector`, `Jinv`).

## Build

One command, from this directory, in an environment with `mojo` (for example
`mast3r-slam-dev`):

```bash
mojo build --emit shared-lib -I lietorch_mojo/mojo -o lietorch_mojo/_lietorch_mojo_backends.so lietorch_mojo/mojo/_lietorch_mojo_backends.mojo
```

From the repo root without activating the environment:

```bash
pixi run -e mast3r-slam-dev --frozen bash -c "cd packages/lietorch-mojo && mojo build --emit shared-lib -I lietorch_mojo/mojo -o lietorch_mojo/_lietorch_mojo_backends.so lietorch_mojo/mojo/_lietorch_mojo_backends.mojo"
```

The build compiles the GPU kernels ahead of time (ptxas runs during the build;
the `.so` embeds sm_120a cubins and no PTX, so there is no runtime JIT). The
extension creates its CUDA context on the first CUDA call, so CPU-only use
never touches the GPU.

## CUDA streams

Every kernel runs on PyTorch's current CUDA stream
(`torch.cuda.current_stream()`), asynchronously, and the backend never
synchronises: ops order with the PyTorch work around them as native PyTorch
ops do, so side streams and CUDA graph capture work. Upstream lietorch
always launches on the legacy default stream, so on a non-default stream its
results can be wrong. Parity tests against lietorch therefore run on the
default stream. `tests/test_streams.py` stress-tests the ordering.

Preliminary build times on pablo-dl-server (RTX 5090, 32 cores, other jobs
running): ~12–13.5 s with an empty Mojo cache (`MODULAR_HOME` fresh), ~1.6 s
when Mojo's compile cache already holds the same source, ~13 s after a
one-line code change (the cache is per module, so any code change rebuilds
everything).

## Test

There is no pixi feature for this package yet; run with `PYTHONPATH` set:

```bash
cd packages/lietorch-mojo
PYTHONPATH=$PWD pixi run -e mast3r-slam-dev --frozen python -m pytest -q            # unit lane: CPU, no lietorch
PYTHONPATH=$PWD pixi run -e mast3r-slam-dev --frozen python -m pytest -q -m integration   # CUDA + parity vs lietorch
```

`integration` tests need a CUDA GPU or the compiled upstream lietorch (the
parity reference, present in `mast3r-slam-dev` and `dpvo-dev`). Before the
extension is built every module that needs it is skipped. `tests/test_math.py`
builds its own test-only extension (`tests/mojo/math_probe.mojo`, ~4 s, cached
in `tests/mojo/.build/`), so the production extension has no test hooks.

## Numerics: where results differ from lietorch on purpose

The port keeps so3.h/se3.h's formulas and branches, with two exceptions. Both
are in zones where lietorch's own formula loses most of its digits:

- **float32 `SE3.calcQ`** (Jacobians: `Jinv`, `expm_backward`, `logm_backward`)
  uses its Taylor series for theta < 2. lietorch's closed forms cancel there,
  and lietorch's CPU and CUDA builds disagree with each other by up to ~0.2
  (strict row metric), so no single lietorch value exists to match.
- **float64 `SO3.left_jacobian`** evaluates (1 - cos θ)/θ² as
  sin²θ / ((1 + cos θ) θ²) for θ < 1. so3.h's form leaves ~1e-11 of error in
  `Exp`'s translation at θ = 1e-6, enough to fail float64
  `torch.autograd.gradcheck` at the identity (upstream lietorch fails it too).
  The change moves float64 results by at most lietorch's own error (≤ ~1e-10
  relative).

Separately, `SO3.act` (`cross` included) and the normalising quaternion
constructor spell out their multiply-adds as explicit FMAs. Left to the
compiler, FP contraction fused different pairs in different kernels, so the
same rotation rounded differently in `inv` and `mul` and float32
`X.inv() * X` kept ulp residuals in 29% of rows; now 0.6% (lietorch's CPU build:
0.9%).

## Layout

```
lietorch_mojo/
  __init__.py        beartype activation (PIXI_DEV_MODE) + exports
  groups.py          port of lietorch groups.py (LieGroup, SO3, RxSO3, SE3, Sim3, LieGroupParameter, cat, stack)
  group_ops.py       port of lietorch group_ops.py (autograd Functions)
  broadcasting.py    port of lietorch broadcasting.py
  backends.py        lietorch_backends-compatible functions; `_run` is the one Python launch site
  gradcheck.py       port of lietorch gradcheck.py (gradcheck returns (analytical, numerical) like upstream)
  mojo/
    lie_math.mojo    sin/cos/atan/sqrt that are accurate on CPU and NVIDIA GPU
    linalg.mojo      fixed-size register matrices (Eigen stand-in)
    lie_group.mojo   the LieGroup trait (dtype, N, K and the group primitives)
    so3.mojo         SO3[dtype], port of so3.h
    se3.mojo         SE3[dtype], port of se3.h
    group_ops.mojo   op codes, row widths, and one per-element function per op family (unary, unary backward, binary, binary backward)
    launch.mojo      TileTensor rows, GPU kernels, CPU loop, GPURuntime / CPURuntime
    _lietorch_mojo_backends.mojo   CPython entry (PyInit), runtime → comptime dispatch, tensor-count check
```

Every tensor crosses the boundary as a (batch, width) row-major `TileTensor`.
The extension owns the row widths (`row_widths`); `backends._run` checks each
input's shape against them before it launches, and the extension checks the
number of tensors, so a wrong shape raises `ValueError` instead of reading or
writing out of bounds. The CUDA and CPU contexts are Mojo objects
(`GPURuntime`, `CPURuntime`) held by Python and type-checked on every call.

Adding a group (SO3, RxSO3, Sim3) means one Mojo struct that implements
`LieGroup`, one dispatch line in `_lietorch_mojo_backends.mojo`, and its id in
`backends._IMPLEMENTED_DIMS`.
