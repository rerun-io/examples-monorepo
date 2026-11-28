# caspar-mojo

COLMAP's CASPAR GPU bundle adjustment (Levenberg-Marquardt + PCG on the reduced camera system, float32), ported
from about 700 Symforce-generated CUDA kernels to twelve generic Mojo GPU kernels. The compiler instantiates each kernel
per camera model and refined-parameter set; nothing is generated. numpy arrays in and out (`caspar_mojo.problem`),
no torch. The host loop (`caspar_mojo/mojo/solver.mojo`) is a 1:1 port of CASPAR's `GraphSolver::solve`: same
damping rule, same PCG stop, the same trial-by-trial trace (`BaResult.trace`), with one stated difference:

- **Acceptance floor.** A trial is accepted iff `trial < best * solver_rel_decrease_min * (1 - COST_DECREASE_FLOOR)`,
  `COST_DECREASE_FLOOR = 1e-6`, in float64 (`solver.mojo`; the oracle `reference/solver.py` has the same line).
  CASPAR's float32 cost sum cannot resolve a smaller decrease (its noise is about 1e-6 relative); our float64 sum
  can, and polishing below it cost up to 2.7x the solve for no change in the result (clip 007, P1: mean reprojection
  error identical to 6 digits). 1e-6 is also Ceres's default function tolerance. Callers pass CASPAR's
  `solver_rel_decrease_min` (default 1.0) unchanged.

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
| `point_gather_kernel` | `Model`, `REFINE_INTRINSICS`, `REFINE_RIG`, `BACKSUBSTITUTE` | `jtjnjtr_direct` pass 1 of `RcsMatvec` + `RcsPointSolve`; `RcsBackSubstitute` |
| `cross_points_kernel` | `Model`, `REFINE_INTRINSICS`, `REFINE_RIG`, `CAMERA_TERMS` | `jtjnjtr_direct` pass 2 of `RcsMatvec`, `RcsReduceRhs`'s product |
| `point_solve_kernel` (node_kernels) | `K` (calibration block size: the point group follows it) | `RcsPointSolve` in `RcsReduceRhs` |
| `pcg_precond_kernel` | `K` (calibration block size) | `normalize` (every node type), `update_r*`, `update_step*`, `alpha_*`, the rho dot |
| `pcg_direction_kernel` | none | `update_p`, `update_Mp`, `start_w*`, `BetaFromNumDenom` |
| `dot_kernel` | none | the p.w dots |
| `axpy_kernel` | none | the last `update_step` |
| `retract_kernel` | `Model` | `retract` (every node type), `pred_decrease_times_two` |
| `gather_rows_kernel` (factor_kernels) | none | CASPAR's host-side `sort_indices` / `shared_indices` for the point-ordered copy |

The packed layout of the tangent vector and of the block diagonal (group order, block widths, offsets, the
triangle index), the damped block solve and the predicted-decrease term live in `blocks.mojo` (`Packed[K]`); the
host allocation, every kernel and the Python test hook (`packed_groups`) read offsets from it only. Score and
linearisation project through one helper (`observe[Model]`), so LM compares costs from the same arithmetic.

A camera model is one struct of the `CameraModel` trait (`camera_models.mojo`): the projection and its analytic
Jacobians (d pixel / d camera point, d pixel / d focal-and-extra). The kernels hold no model code. Fixed nodes are
runtime data (tangent offset -1). Three choices carry the speed, all generic:

- The matrix-free products recompute the Jacobians from the linearisation state instead of reading a per-observation
  cache: the state is a few hundred KB that stays in L2, a cache streams 18-50 floats per observation per pass.
- Everything that sums into a point (its gradient and block, the point half of the Schur product) runs one warp per
  point over a CSR copy of the observations sorted by point: no atomics, and the point solve fuses in.
- Everything that sums into a camera-side node runs over warp tiles: up to 32 consecutive observations of ONE image
  (camera, frame) each, eight per block tile, 6 blocks per SM, each block walking consecutive block tiles. Every warp
  shares its nodes by construction, whatever the image sizes (6,000 observations per image on P1, 368 on clip 007),
  so there is no uniformity test and no per-thread fallback; a block merges its warps' sums in order into running
  sums that reach memory once per node change, so the atomics on one address (a camera's intrinsics are shared by
  every observation) fall by the warps per block. A 256-observation tile per block instead made clip 007 9x slower:
  its images straddle tiles, and those tiles fell back to per-warp and per-thread atomics.
- **Observation order.** Any order is correct. `solve` (both doors) regroups observations by (camera, frame) with
  two stable counting sorts when they do not come sorted; sorted input (COLMAP's adapter, the benchmark) skips the
  copy. The point-ordered copy is a host row permutation (pinned) gathered on the device.

Per PCG iteration: 5 launches and one 16-byte read-back (CASPAR: about 40 launches and 2 read-backs).

## Build, test, benchmark

```bash
pixi run -e caspar-mojo-dev --frozen caspar-mojo-build      # caspar_mojo/_caspar_mojo.so for the local GPU
pixi run -e caspar-mojo-dev --frozen tests                  # unit lane (no GPU)
pixi run -e caspar-mojo-dev --frozen tests-integration      # GPU: every model and flag set against the reference
pixi run -e caspar-mojo-dev --frozen caspar-mojo-build-c    # caspar_mojo/libcaspar_mojo.so, the C ABI (include/caspar_mojo.h)
pixi run -e caspar-mojo --frozen python tools/benchmark.py  # solver timing on synthetic P1, 100-frame, clip007-shape problems
```

Mojo builds GPU kernels for the GPU of the host that builds, so build on the host that runs (RTX 5090 sm_120,
RTX 3060 sm_86, GB10 sm_121 all work with no flags). `caspar_mojo.reference` is the float64 numpy oracle (camera
models, LM + PCG, synthetic problems from a seed) that the integration tests compare against, trial by trial.
`CASPAR_MOJO_PROF=1` is the one profiling switch: a sync after every kernel and the time per kernel printed at the end
(never use it for timing); `tools/benchmark.py --profile` adds one such solve per case. `tools/bench/` holds standalone probes:
`launch_overhead.mojo` (launch, sync and read-back cost) and `mojo_smoke_ext.mojo` + `coexist.py` (CASPAR and a Mojo
kernel in one process). The `colsfm` environments also carry this package, so
CASPAR (pycolmap) and caspar-mojo run in one process.

## Results

Solver loop (`BaResult.solve_seconds`: first linearisation to the last trial, set-up and copies excluded), best of 5-7
warm solves, prod environment, no profiler, floor built in; Mojo LM trials / PCG iterations, ms per PCG iteration
all-in. Real models through `colsfm.ba_mojo.problem_from_reconstruction` (2026-11-27/28, heads 7003d3b4-df7bc39c7).
CASPAR: solver-only `dt_tot` of its verbose trace (baselines.md, colmap A/B; its trial and PCG counts in brackets).

| model | GPU | Mojo s | LM / PCG | ms/PCG | CASPAR s |
|---|---|---|---|---|---|
| p2 clip 007 (1.14 M obs) | RTX 3060 | 0.0135 | 6 / 13 | 1.04 | 0.066-0.078 (6-7 / 14) |
| p2 clip 007 | RTX 5090 | 0.0023 | 6 / 13 | 0.18 | 0.015-0.032 (9-18 / 19-43) |
| p2 clip 007 | GB10 | 0.0087 | 6 / 13 | 0.67 | 0.159 (10 / 22) |
| p2 perturbed (seed 0) | RTX 3060 | 0.278 | 18 / 561-598 | 0.50 | about 2.3 ms per PCG |
| p2 perturbed | RTX 5090 | 0.0485 | 18 / 572-657 | 0.085 | about 0.39 ms per PCG |
| p2 perturbed | GB10 | 0.180 | 18 / 556-628 | 0.32 | - |
| p1 (2.4 M obs) | RTX 3060 | 0.0836 | 21 / 97 | 0.86 | 0.518 (22 / 83) |
| p1 | RTX 5090 | 0.0164 | 21 / 96 | 0.17 | 0.085 (26 / 89) |
| p6 rig fixed (0.6 M obs) | RTX 3060 | 0.0386 | 25-29 / 168-194 | 0.23 | 0.360 (40 / 224) |
| p6 rig fixed | RTX 5090 | 0.0101 | 24-28 / 160-195 | 0.063 | - |
| p6 +rig | RTX 3060 | 0.0277 | 17 / 84-88 | 0.33 | 0.248 (20 / 111) |
| p6 +rig | RTX 5090 | 0.0147 | 17 / 84-88 | 0.17 | - |

The GB10 shares its memory with the CPU: its rows were measured at a host load of 8-9 and are noisier. Logs:
`/mnt/nas/ssd-8tb/scratch/caspar-mojo/results/port/`.
