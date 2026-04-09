# Isolated MAX GN Kernel Plan

This note captures the clean path for building a true isolated MAX custom-op implementation of the GN rays kernel, while keeping the current idiomatic Mojo shared-library backend as the active production path.

## Current repo state

- Active GN backend path should remain the idiomatic Mojo shared-library implementation exposed by `mast3r_slam_mojo_backends`.
- The experimental MAX custom-op work should not be on the active dispatch path until it is correct and benchmarked.
- The `mast3r-slam-max` Pixi feature now installs `modular` directly through Pixi from the MAX nightly channel, which is the right environment model for MAX and Mojo together.

## Environment

The relevant Pixi setup is:

- feature: `mast3r-slam-max`
- channels:
  - `https://prefix.dev/ai-demos`
  - `https://conda.modular.com/max-nightly/`
  - `conda-forge`
- dependency:
  - `modular`

This is the intended equivalent of the Modular quickstart guidance, but adapted to an existing monorepo instead of `pixi init`.

## What upstream MAX kernels show

The most relevant upstream references are:

- MAX custom-op overview:
  - https://docs.modular.com/max/develop/custom-ops/
- Build custom ops:
  - https://docs.modular.com/max/develop/build-custom-ops/
- PyTorch custom kernels:
  - https://docs.modular.com/max/develop/custom-kernels-pytorch/
- MAX GPU profiling:
  - https://docs.modular.com/max/gpu-system-profiling/
- Upstream example package:
  - `max/examples/custom_ops`
- Upstream production kernel patterns:
  - `max/kernels/src/state_space/*`

The main design patterns from those examples are:

1. Keep the package isolated.
   Use a dedicated Python wrapper plus a dedicated `kernels/` Mojo package. Do not mix the isolated MAX kernel work into the shared-lib Mojo backend files.

2. Keep the op boundary thin.
   A registered op should only do shape checks, tensor conversion, dispatch, and launch. The math should live in a separate lower-level module.

3. Prefer production launch patterns over tutorial shortcuts.
   The examples use simple `foreach` in places, but the production `max/kernels` code favors explicit compiled/queued GPU launches. That is the right direction for GN rays.

4. Keep `InputTensor`/`OutputTensor` only at the boundary.
   Convert once, then run the real kernel math using the lower-level abstraction the math wants. Do not spread tensor-boundary indexing logic through the full algorithm.

5. Treat multi-output as normal.
   Returning `hs` and `gs` as separate outputs is idiomatic. We should not contort the design just to avoid multiple outputs.

6. Benchmark in layers.
   First a direct kernel/op benchmark, then PyTorch wrapper benchmarking, then `nsys` profiling. Do not tune using only end-to-end SLAM runs.

## What likely went wrong in the current experiment

The failed experiment mixed too many concerns inside one op:

- large `foreach` closures doing full GN math directly against `InputTensor` indexing
- debug probes and production logic in the same file
- attempts to reuse shared-lib launch ideas too literally
- fallback logic in Python that obscured whether the MAX path was truly isolated

That made it hard to distinguish:

- interface correctness
- MAX compiler/runtime issues
- kernel math bugs
- dispatch and synchronization mistakes

## Clean design for our repo

Build the isolated MAX path as a separate vertical slice under `packages/mast3r-slam/mast3r_slam/backend/max_gn_rays/` or a similarly explicit location.

Recommended files:

- `packages/mast3r-slam/mast3r_slam/max_gn_rays.py`
  - Python wrapper for `CustomOpLibrary`
  - no fallback to shared-lib Mojo inside the strict path
- `packages/mast3r-slam/mast3r_slam/backend/max_gn_rays/kernels/__init__.mojo`
  - registers ops
- `packages/mast3r-slam/mast3r_slam/backend/max_gn_rays/kernels/gn_rays_ops.mojo`
  - thin op boundary
- `packages/mast3r-slam/mast3r_slam/backend/max_gn_rays/kernels/gn_rays_gpu.mojo`
  - real GPU math kernels
- `packages/mast3r-slam/mast3r_slam/backend/max_gn_rays/benchmarks.mojo`
  - direct op microbenchmarks

## Implementation order

1. Start with a true step-only MAX op.
   Implement only the equivalent of `gauss_newton_rays_step`, producing partial `hs` and `gs`.

2. Keep outputs explicit.
   Use two outputs:
   - `hs_partial`
   - `gs_partial`

3. Mirror the validated Mojo/CUDA math structure.
   Reuse the same decomposition:
   - relative pose setup
   - transformed point
   - residuals
   - Jacobian terms
   - Hessian/gradient accumulation

4. Move full math out of the boundary file.
   `gn_rays_ops.mojo` should dispatch into functions from `gn_rays_gpu.mojo`, not embed the whole algorithm inline.

5. Avoid a giant `foreach` closure for production.
   If `foreach` is used at all, use it only for tiny validation probes. The real kernel should use the launch style that matches the production `max/kernels` examples more closely.

6. Only after the step op is correct, rebuild the public GN loop in Python.
   The Python side can still solve the dense system and apply `pose_retr` until the MAX step is proven.

## Validation plan

Validation should be done in this order:

1. `check-max-python`
   Verify `max.experimental.torch` and the wrapper import cleanly.

2. direct step parity
   Compare MAX step `hs` and `gs` against the idiomatic Mojo/CUDA oracle on representative synthetic rays inputs.

3. public GN parity
   Rebuild the Python GN loop around the isolated step op and compare final `Twc`.

4. synthetic throughput
   Use `packages/mast3r-slam/tools/bench_gn_mojo_vs_cuda.py` as the primary tuning surface.

5. profiling
   Use `nsys` and then deeper MAX/kernel profiling only after the op is numerically correct.

## Success criteria

The isolated MAX kernel is ready only when all of these are true:

- no fallback to shared-lib Mojo in the strict MAX path
- numerical agreement with CUDA/Mojo within the current GN tolerance
- throughput on representative synthetic rays is within the target window
- the active repo path can still stay on idiomatic Mojo until MAX is ready

## Recommendation for now

Keep the repo on the idiomatic Mojo backend today.

Use the MAX environment and this plan to build the isolated custom-op implementation as a separate clean effort, not as a partially active hybrid path.
