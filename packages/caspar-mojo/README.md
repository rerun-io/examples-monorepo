# caspar-mojo

COLMAP's CASPAR GPU bundle adjustment (Levenberg-Marquardt + PCG on the reduced camera system, float32), ported
from about 700 Symforce-generated CUDA kernels to ten generic Mojo GPU kernels. The compiler instantiates each kernel
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
| `linearize_kernel` (factor_kernels) | `Model`, `REFINE_INTRINSICS`, `REFINE_RIG` | `res_jac_first`, `res_jac`, every factor variant |
| `score_kernel` | `Model` | `score`, every factor variant |
| `cross_cameras_kernel` | `Model`, `REFINE_INTRINSICS`, `REFINE_RIG` | `jtjnjtr_direct` pass 1 of `RcsMatvec`, `RcsBackSubstitute`'s product |
| `cross_points_kernel` | `Model`, `REFINE_INTRINSICS`, `REFINE_RIG` | `jtjnjtr_direct` pass 2, `RcsReduceRhs`'s product |
| `point_solve_kernel` (node_kernels) | mode (negate / back-substitute) | `RcsPointSolve`, the point step of `RcsBackSubstitute` |
| `pcg_precond_kernel` | `K` (calibration block size) | `normalize` (every node type), `update_r*`, `update_step*`, `alpha_*`, the rho dot |
| `pcg_direction_kernel` | none | `update_p`, `update_Mp`, `start_w*`, `BetaFromNumDenom` |
| `dot_kernel` | none | the p.w dots |
| `axpy_kernel` | none | the last `update_step` |
| `retract_kernel` | `Model` | `retract` (every node type), `pred_decrease_times_two` |

A camera model is one struct of the `CameraModel` trait (`camera_models.mojo`): the projection and its analytic
Jacobians (d pixel / d camera point, d pixel / d focal-and-extra). The kernels hold no model code. Fixed nodes are
runtime data (tangent offset -1). The matrix-free products recompute the Jacobians from the linearisation state
rather than reading a per-observation cache. Scatter-adds pre-reduce over the block (observations sorted by camera,
then frame) before one device atomic. Per PCG iteration: 6 launches and one 16-byte read-back (CASPAR: about 40
launches and 2 read-backs).

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
time per kernel printed at the end (never use it for timing). The `colsfm` environments also carry this package, so
CASPAR (pycolmap) and caspar-mojo run in one process.

## Results

To be filled in.
