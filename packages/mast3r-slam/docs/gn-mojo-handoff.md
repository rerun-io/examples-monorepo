# GN Mojo Handoff

## Goal

Replace the current GN backend in `mast3r-slam` with a true Mojo drop-in replacement for the CUDA/C++ path in:

- `mast3r_slam/backend/src/gn_kernels.cu`
- `mast3r_slam/backend/src/gn.cpp`

Hard requirements:

1. Numerical equivalence to the CUDA backend within explicit tolerances
2. Throughput at least 95% of CUDA on real workloads, not synthetic microbenchmarks
3. End-to-end signoff on real example runs:
   - `livingroom-tour.mp4` with base config
   - `example-base`

Signoff clarification:

- synthetic microbenchmarks are diagnostic only
- the hard 5% gate is on captured real GN fixtures and full example runs
- microbench regressions still matter for profiling, but they are not the release gate by themselves

## Current Status

The branch contains real GN Mojo work, but it does **not** meet the throughput requirement yet.

### What is implemented

- A separate Mojo GN module layout now exists:
  - `mast3r_slam/backend/mojo/gn_kernels.mojo`
  - `mast3r_slam/backend/mojo/gn.mojo`
  - `mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo`
- Python now routes GN through `mast3r_slam/gn_backends.py`
- There is a real Mojo implementation of `gauss_newton_rays_step`
- There is a real Mojo implementation of `pose_retr`
- There is a MAX custom-op prototype for `pose_retr`
- Step-level CUDA-vs-selected-backend tests and benchmarks exist

### What is still fallback or incomplete

- `gauss_newton_points_step` still uses the CUDA extension
- `gauss_newton_calib_step` still uses the CUDA extension
- `gauss_newton_points` and `gauss_newton_calib` public functions currently go through the Mojo host wrapper, but their per-step accumulation still calls CUDA
- `gauss_newton_rays` public function uses the Mojo host path plus Mojo `rays_step`, but is still slower than CUDA
- No final signoff benchmark has been run yet for:
  - `livingroom-tour.mp4`
  - `example-base`
  - real captured GN graph fixtures

## Important Files

### Primary implementation

- `pixi.toml`
- `packages/mast3r-slam/mast3r_slam/global_opt.py`
- `packages/mast3r-slam/mast3r_slam/gn_backends.py`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/matching.mojo`

### CUDA oracle / existing backend

- `packages/mast3r-slam/mast3r_slam/backend/include/gn.h`
- `packages/mast3r-slam/mast3r_slam/backend/src/gn.cpp`
- `packages/mast3r-slam/mast3r_slam/backend/src/gn_kernels.cu`

### MAX custom-op exploration

- `packages/mast3r-slam/mast3r_slam/max_ops.py`
- `packages/mast3r-slam/mast3r_slam/backend/max_ops/gn_ops/__init__.mojo`
- `packages/mast3r-slam/max-custom-ops/pixi.toml`

### Tests and benchmarks

- `packages/mast3r-slam/tests/test_gn_step_api.py`
- `packages/mast3r-slam/tests/test_max_custom_ops.py`
- `packages/mast3r-slam/tools/bench_gn_kernels.py`
- `packages/mast3r-slam/tools/bench_max_pose_retr.py`
- `packages/mast3r-slam/tools/bench_max_pose_retr_sweep.py`

### Legacy / intermediate files worth reading but not necessarily keeping

- `packages/mast3r-slam/mast3r_slam/backend/mojo/matching_kernels.mojo`
- `packages/mast3r-slam/mast3r_slam/mojo_gn.py`

## What Was Learned

### 1. The biggest remaining blocker is `gauss_newton_rays_step`, not the host solve

The current best measured numbers on the synthetic GN benchmark are:

| Surface | CUDA median | Mojo median | Ratio |
|---|---:|---:|---:|
| `rays_step` | `0.060 ms` | `0.095 ms` | `1.594x` |
| `calib_step` | `0.056 ms` | `0.056 ms` | `0.992x` |
| `rays_public` | `0.178 ms` | `0.315 ms` | `1.762x` |

Interpretation:

- `calib_step` is effectively parity today, but only because it is still using CUDA
- `rays_public` is too slow, but most of that gap is already explained by `rays_step`
- once `rays_step` is fixed, the public path may become competitive

### 2. The GPU dense-solve path was worse than a CPU-side block solve for these tiny GN systems

I tried two host-side strategies in `gn.mojo`:

- GPU dense assembly + GPU Cholesky in Torch
- CPU-side block assembly + CPU Cholesky from `Hs` and `gs`, returning `dx` to CUDA

Result:

- the GPU dense solve path was clearly worse
- the CPU-side block solve was materially better for the current small synthetic fixture

This matches the existing CUDA backend more closely than expected, because the current C++ path already transfers data to CPU/Eigen for the sparse solve.

### 3. MAX custom ops are technically viable, but small-op overhead is real

MAX `CustomOpLibrary` and launched GPU kernels work in this repo. The first tested surface was `pose_retr`.

Observed pose-retraction sweep:

| Work size | MAX / CUDA ratio |
|---|---:|
| `64` | `5.958x` |
| `256` | `5.258x` |
| `1024` | `3.691x` |
| `4096` | `1.953x` |
| `16384` | `0.672x` |

Interpretation:

- MAX overhead amortizes on large workloads
- MAX `pose_retr` alone is not a proof that GN will meet the requirement
- if MAX is revisited, it should be for a heavy accumulation surface like `rays_step`, not for tiny helper ops

### 4. Several obvious kernel tweaks did not close the gap

These were tried on `gauss_newton_rays_step` in `gn_kernels.mojo`:

- serial edge kernel
- block-parallel reduction
- `RAYS_THREADS = 64`
- `RAYS_THREADS = 256`
- caching pose data once per block
- computing relative Sim3 once per block in shared memory

Results:

- `64` threads was better than `256`
- shared pose and shared relative Sim3 did not materially change the benchmark
- the kernel is still about 1.6x slower than CUDA

This suggests the real issue is algorithmic/codegen quality in the current Mojo ray-step kernel, not just one missing cache or launch constant.

### 5. The current testing is good enough for iteration, but not enough for final signoff

Current correctness coverage:

- `test_gn_step_api.py` passes
- step-level parity for `rays_step` is checked
- current tolerance in tests:
  - `Hs`: `atol=1e-4`, `rtol=1e-4`
  - `gs`: `atol=1e-3`, `rtol=1e-4`

Missing for real signoff:

- captured real GN graph fixtures
- multi-iteration parity checks against CUDA
- final `Twc` pose tolerance report
- `livingroom-tour.mp4` end-to-end timing
- `example-base` timing report

## Likely Root Cause of Being Stuck

The branch no longer looks blocked on packaging or backend-selection architecture. Those pieces exist.

The actual stuck point is narrower:

- the current Mojo implementation of `gauss_newton_rays_step` does not compile down to something competitive with the CUDA kernel
- the current public GN path depends on that kernel
- without a faster `rays_step`, the hard 95% gate will not pass

## Revised Execution Plan

This is the recommended next plan after reviewing the current branch and re-checking the Modular docs.

### 1. Freeze scope around `gauss_newton_rays_step`

Do not widen the port until `gauss_newton_rays_step` is near parity on real fixtures.

Leave these on CUDA for now:

- `gauss_newton_points_step`
- `gauss_newton_calib_step`

Do not spend more time right now on:

- `pose_retr`
- backend-selector cleanup
- packaging refactors
- host-path rewrites that do not directly move `rays_step`

The branch evidence already says `rays_step` is the critical path.

### 2. Build the signoff harness around real fixtures

Before more tuning, add captured GN fixtures at the `_backends.gauss_newton_*` call boundary.

Required captured tensors:

- `Twc`
- `Xs`
- `Cs`
- `ii`
- `jj`
- `idx_ii2jj`
- `valid_match`
- `Q`
- plus `K`, `height`, `width` for calib

Required checks:

- multi-iteration CUDA-vs-selected-backend parity
- final `Twc` translation / quaternion / scale error report
- iteration-count report
- real-fixture benchmark rows for `gauss_newton_rays`

Signoff rule:

- the hard `<= 1.05x` gate is on captured real fixtures and real example runs
- synthetic step and kernel microbenches remain diagnostic only

### 3. Profile like a kernel project

Use `nsys` first for end-to-end time attribution, then Nsight Compute and/or `kbench` on the hot kernel.

The next agent should record for `gauss_newton_rays_step` versus CUDA:

- registers per thread
- spills / local memory
- achieved occupancy
- shared-memory usage
- memory transactions
- barrier / reduction cost
- kernel duration

Doc notes verified against current Modular docs:

- block reductions in `gpu.primitives.block` are explicitly documented as easier to use correctly and often more efficient than manual `barrier()` plus shared-memory reductions
- MAX custom ops are documented under `max.experimental.torch.CustomOpLibrary`
- `DeviceContext` is documented as a GPU execution stream abstraction

### 4. Rewrite `gauss_newton_rays_step` to match CUDA more literally

The current Mojo kernel still looks too helper-heavy in the hot loop.

Rewrite priorities:

- fewer helper boundaries inside the inner loop
- scalar locals or fixed local arrays instead of excess `InlineArray` traffic
- one relative Sim3 computation per block
- accumulation order closer to the CUDA kernel
- replace manual shared-memory tree reductions with Mojo block or warp reduction primitives where possible

After the rewrite, retune thread/block size instead of assuming `RAYS_THREADS = 64` remains optimal.

### 5. Treat stream / wrapper changes as a secondary experiment

Do not lead with another architecture rewrite unless profiling proves the integration boundary is the main cost.

If profiling later shows real boundary overhead between the shared-lib Mojo path and PyTorch/CUDA ordering, then the right follow-up is to:

- align stream usage
- or move only the hot surface into a proper MAX custom op

Do not keep adding ad hoc sync behavior without proof it helps.

### 6. Use MAX as Plan B, but only for the hot surface

If the CUDA-shaped Mojo rewrite still leaves `gauss_newton_rays_step` materially off target, pivot only that surface to MAX.

Suggested pivot threshold:

- if `gauss_newton_rays_step` remains worse than about `1.15x` CUDA on the captured real fixture, stop iterating on the shared-lib path and try MAX for that one accumulation op

Do not broaden MAX to the whole GN stack unless the single-op experiment proves it is the right direction.

### 7. Only after `rays_step` is near parity should the rest move

Once `rays_step` is close enough:

- rerun the public `gauss_newton_rays` benchmark on the captured real fixture
- rerun full pipeline benchmarks on `livingroom-tour.mp4` base config
- rerun `example-base`

Only then decide whether `points_step` and `calib_step` are worth porting.

Keep the current file split:

- `gn_kernels.mojo`
- `gn.mojo`
- `matching.mojo`
- `mast3r_slam_mojo_backends.mojo`

## Reproduction Commands

### Build

```bash
pixi run -e mast3r-slam-dev _build-mojo-kernels
```

### GN tests

```bash
cd packages/mast3r-slam
pixi run -e mast3r-slam-dev pytest tests/test_gn_step_api.py -q
```

### GN microbench

```bash
cd packages/mast3r-slam
pixi run -e mast3r-slam-dev python tools/bench_gn_kernels.py
```

### MAX custom-op tests

```bash
cd packages/mast3r-slam
pixi run --manifest-path max-custom-ops/pixi.toml test-max-custom-ops
```

### MAX pose-retr microbench

```bash
cd packages/mast3r-slam
pixi run --manifest-path max-custom-ops/pixi.toml bench-max-pose-retr
pixi run --manifest-path max-custom-ops/pixi.toml python tools/bench_max_pose_retr_sweep.py
```

### Full pipeline signoff command that still needs to be run

`example-base` uses `normal-apt-tour.mp4`, not `livingroom-tour.mp4`.

```bash
pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
  --dataset data/livingroom-tour.mp4 --img-size 512 --config config/base.yaml
```

## Notes For The Next Agent

- The branch contains both shared-lib Mojo work and MAX custom-op experiments
- The MAX work is exploratory, not production-ready
- `matching_kernels.mojo` still contains old combined code and should be treated as legacy
- The cleaned split files are the intended direction
- The current branch is useful as a record of what was tried, even where it failed
