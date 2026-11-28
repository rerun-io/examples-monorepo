# caspar-mojo

COLMAP's CASPAR GPU bundle adjustment (Levenberg-Marquardt + PCG on the reduced camera system, float32), ported
from about 700 Symforce-generated CUDA kernels to eleven generic Mojo GPU kernels. The compiler instantiates each kernel
per camera model and refined-parameter set; nothing is generated. numpy arrays in and out (`caspar_mojo.problem`),
no torch. The host loop (`caspar_mojo/mojo/solver.mojo`) is a 1:1 port of CASPAR's `GraphSolver::solve`: same
damping rule, same PCG stop, the same trial-by-trial trace (`BaResult.trace`).

Supported: PINHOLE, SIMPLE_RADIAL, OPENCV_FISHEYE, FULL_OPENCV (one model per solve); poses, points, focal-and-extra
and `sensor_from_rig` each free or fixed per node. FULL_OPENCV with free intrinsics (a 10x10 calibration block)
works here; CASPAR cannot do it. Not ported: the full-system PCG (`reduced_camera_system=False`), a refined
principal point, mixed camera models in one problem.

## Kernels

| Kernel (`caspar_mojo/mojo/`) | Compile-time parameters | Replaces in CASPAR |
|---|---|---|
| `linearize_kernel` (factor_kernels) | `Model`, `REFINE_INTRINSICS`, `REFINE_RIG` | `res_jac_first`, `res_jac` (camera blocks), every factor variant |
| `point_linearize_kernel` | `Model` | `res_jac` (point blocks) |
| `score_kernel` | `Model` | `score`, every factor variant |
| `point_gather_kernel` | `Model`, `REFINE_INTRINSICS`, `REFINE_RIG`, mode | `jtjnjtr_direct` pass 1 of `RcsMatvec` + `RcsPointSolve`; `RcsBackSubstitute` |
| `cross_points_kernel` | `Model`, `REFINE_INTRINSICS`, `REFINE_RIG`, `CAMERA_TERMS` | `jtjnjtr_direct` pass 2 of `RcsMatvec`, `RcsReduceRhs`'s product |
| `point_solve_kernel` (node_kernels) | none | `RcsPointSolve` in `RcsReduceRhs` |
| `pcg_precond_kernel` | `K` (calibration block size) | `normalize` (every node type), `update_r*`, `update_step*`, `alpha_*`, the rho dot |
| `pcg_direction_kernel` | none | `update_p`, `update_Mp`, `start_w*`, `BetaFromNumDenom` |
| `dot_kernel` | none | the p.w dots |
| `axpy_kernel` | none | the last `update_step` |
| `retract_kernel` | `Model` | `retract` (every node type), `pred_decrease_times_two` |

A camera model is one struct of the `CameraModel` trait (`camera_models.mojo`): the projection and its analytic
Jacobians (d pixel / d camera point, d pixel / d focal-and-extra). The kernels hold no model code. Fixed nodes are
runtime data (tangent offset -1). Three choices carry the speed, all generic:

- The matrix-free products recompute the Jacobians from the linearisation state instead of reading a per-observation
  cache: the state is a few hundred KB that stays in L2, a cache streams 18-50 floats per observation per pass.
- Everything that sums into a point (its gradient and block, the point half of the Schur product) runs one warp per
  point over a CSR copy of the observations sorted by point: no atomics, and the point solve fuses in.
- Everything that sums into a camera-side node runs over the observations in camera-then-frame order (the caller's
  order; COLMAP's adapter and the benchmark both give it), in tiles of 256 per block, 4 blocks per SM; a block
  pre-reduces each tile and carries the sum until the node changes, so the device atomics on one address (a camera's
  intrinsics are shared by every observation) fall by the tiles per block.

Per PCG iteration: 5 launches and one 16-byte read-back (CASPAR: about 40 launches and 2 read-backs).

## Build, test, benchmark

```bash
pixi run -e caspar-mojo-dev --frozen caspar-mojo-build      # caspar_mojo/_caspar_mojo.so for the local GPU
pixi run -e caspar-mojo-dev --frozen tests                  # unit lane (no GPU)
pixi run -e caspar-mojo-dev --frozen tests-integration      # GPU: every model and flag set against the reference
pixi run -e caspar-mojo --frozen python tools/benchmark.py  # solver timing on synthetic P1 and 100-frame problems
```

Mojo builds GPU kernels for the GPU of the host that builds, so build on the host that runs (RTX 5090 sm_120,
RTX 3060 sm_86, GB10 sm_121 all work with no flags). `caspar_mojo.reference` is the float64 numpy oracle (camera
models, LM + PCG, synthetic problems from a seed) that the integration tests compare against, trial by trial.
`tools/benchmark.py --profile` adds one solve per case with `CASPAR_MOJO_PROF=1`: a sync after every kernel and the
time per kernel printed at the end (never use it for timing). `tools/bench/` holds standalone probes:
`launch_overhead.mojo` (launch, sync and read-back cost) and `mojo_smoke_ext.mojo` + `coexist.py` (CASPAR and a Mojo
kernel in one process). The `colsfm` environments also carry this package, so
CASPAR (pycolmap) and caspar-mojo run in one process.

## Results

To be filled in.
