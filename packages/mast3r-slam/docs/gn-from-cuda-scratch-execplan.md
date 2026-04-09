# Rebuild the GN Mojo Backend From Scratch Using Only the Original CUDA/C++ Backend as Reference

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained under the rules in `.agents/PLANS.md`.

## Purpose / Big Picture

After this work, a contributor should be able to implement the GN backend in Mojo from scratch without consulting any prior Mojo GN code. The only implementation reference assumed to exist is the original CUDA/C++ backend. The result should be one clean, idiomatic Mojo GN backend with no parallel “current” versus “idiomatic” split, wired into `mast3r-slam` as the default backend, while preserving CUDA as the correctness oracle and fallback.

The user-visible proof is straightforward. The package should build a single Mojo shared library that exports the GN public functions and step functions, `mast3r_slam` should call those exports by default, the GN smoke tests should pass, and the real rays fixtures for `normal-apt` and `livingroom` should show the Mojo public path within 10% of the CUDA public path while preserving the already established tiny pose/update drift envelope.

## Progress

- [x] (2026-04-10 07:45 CDT) Wrote a standalone scratch-build ExecPlan that assumes only the original CUDA/C++ GN backend is available as implementation reference.
- [ ] Inventory the untouched CUDA/C++ backend in detail and copy the required behavior into this document as implementation requirements.
- [ ] Create a single Mojo GN backend layout with one canonical public path and no duplicate Mojo variants.
- [ ] Implement the low-level Mojo kernels for pose retraction and ray accumulation first.
- [ ] Implement the Mojo public GN orchestration around the ray step path and verify it against the real rays fixtures.
- [ ] Decide whether points and calib step kernels must also be ported immediately or may remain CUDA-backed temporarily behind the same public Mojo orchestration.
- [ ] Promote the single Mojo GN implementation to the default backend once tests and real-fixture signoff pass.

## Surprises & Discoveries

- Observation: the release gate for this GN work must be based on real captured rays fixtures, not synthetic microbenchmarks.
  Evidence: earlier work showed synthetic `rays_step` timing often favored CUDA more strongly than the practical public path on the real checked-in rays fixtures.

- Observation: the checked-in `normal-apt` and `livingroom` signoff runs are uncalibrated by default.
  Evidence: `packages/mast3r-slam/config/base.yaml` sets `use_calib: False`, so the live signoff path exercises `solve_GN_rays`, not `solve_GN_calib`.

- Observation: the original C++/CUDA backend mixes two responsibilities in one implementation unit.
  Evidence: `packages/mast3r-slam/mast3r_slam/backend/src/gn_kernels.cu` contains both GPU kernels such as `ray_align_kernel` and host-side sparse solve logic using Eigen through the `SparseBlock` class.

- Observation: the public API shape in C++ is simple even though the implementation is not.
  Evidence: `packages/mast3r-slam/mast3r_slam/backend/src/gn.cpp` exposes six GN entrypoints plus `pose_retr`, and those entrypoints are exactly the boundary the Mojo backend must match.

## Decision Log

- Decision: this spec assumes the current Mojo GN code does not exist and must not be used as implementation reference.
  Rationale: the user explicitly wants a spec that another agent can follow even if all current Mojo work is deleted.
  Date/Author: 2026-04-10 / Codex

- Decision: the scratch implementation target is one canonical Mojo GN backend, not multiple Mojo variants.
  Rationale: the user wants the idiomatic implementation only. The earlier Mojo paths are useful only as lessons, not as code to preserve.
  Date/Author: 2026-04-10 / Codex

- Decision: CUDA remains in scope as the oracle and fallback.
  Rationale: parity and performance signoff still require a trusted baseline, and the original backend is the only guaranteed implementation reference for this spec.
  Date/Author: 2026-04-10 / Codex

- Decision: the order of implementation is public orchestration plus rays first, then points and calib.
  Rationale: the real fixture corpus and the checked-in example runs currently validate the rays path. That makes rays the only surface with a realistic end-to-end acceptance gate in the current repository.
  Date/Author: 2026-04-10 / Codex

## Outcomes & Retrospective

This document deliberately does not describe a migration from the current Mojo branch state. It describes how to reconstruct the GN Mojo backend from the original CUDA/C++ backend alone. That is its main value. It removes hidden assumptions such as “copy the current `gn.mojo`” or “reuse the earlier kernel split.” A future contributor should be able to follow this document after deleting the existing Mojo GN implementation and still land in the right design space.

The practical limitation remains the same one discovered earlier: the current in-repo real signoff path is strongest for rays. This spec therefore makes rays the mandatory first milestone and treats points and calib as follow-on surfaces unless a real calibrated signoff path is added.

## Context and Orientation

The GN backend in this repository is the Gauss-Newton optimizer used by `mast3r-slam` to refine pose tensors. In plain language, it computes residuals and Jacobians for matched points between keyframes, assembles a linear system, solves for a pose update, and applies that update on the Sim3 manifold. “Sim3” means the pose state has translation, rotation, and one scalar scale parameter. In this repository, each pose is stored as eight numbers: translation `t[3]`, quaternion `q[4]`, and scale `s[1]`. The update vector has seven numbers: translation increment `tau[3]`, rotation increment `omega[3]`, and log-scale increment `sigma[1]`.

The untouched reference implementation lives here:

- `packages/mast3r-slam/mast3r_slam/backend/include/gn.h`
- `packages/mast3r-slam/mast3r_slam/backend/src/gn.cpp`
- `packages/mast3r-slam/mast3r_slam/backend/src/gn_kernels.cu`

Read those files first. `gn.h` defines the public function signatures. `gn.cpp` defines the Python extension boundary and confirms which functions are public. `gn_kernels.cu` contains all of the math helpers, the CUDA kernels, and the host-side linear solve logic.

The Mojo implementation that this spec asks for should be created fresh in these repository-relative paths:

- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo`
- `packages/mast3r-slam/mast3r_slam/gn_backends.py`

The benchmark and test paths that must remain valid are:

- `packages/mast3r-slam/tests/test_gn_step_api.py`
- `packages/mast3r-slam/tests/test_gn_fixture_utils.py`
- `packages/mast3r-slam/tools/bench_gn_real_fixtures.py`
- `packages/mast3r-slam/tools/bench_gn_kernels.py`

The live application caller is:

- `packages/mast3r-slam/mast3r_slam/global_opt.py`

The checked-in real fixture corpus that matters today is:

- `packages/mast3r-slam/artifacts/gn-fixtures/verify-normal-apt/rays-000.pt`
- `packages/mast3r-slam/artifacts/gn-fixtures/verify-livingroom/rays-000.pt`

In this spec, “step function” means the per-iteration accumulation function that returns Hessian blocks and gradient blocks but does not apply the full GN loop. “Public GN function” means the function that runs multiple iterations, solves for `dx`, retracts the poses in place, and returns the final update tensor.

## Required Public Interfaces

The Mojo backend must expose exactly these public Python-callable functions, because they are the ones exported by the original extension in `gn.cpp`:

- `gauss_newton_points`
- `gauss_newton_points_step`
- `gauss_newton_rays`
- `gauss_newton_rays_step`
- `gauss_newton_calib`
- `gauss_newton_calib_step`
- `pose_retr`

The function signatures must match the original extension boundary in `gn.h` and `gn.cpp`.

The points public function takes:

    Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
    sigma_point, C_thresh, Q_thresh, max_iter, delta_thresh

The rays public function takes:

    Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
    sigma_ray, sigma_dist, C_thresh, Q_thresh, max_iter, delta_thresh

The calib public function takes:

    Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q,
    height, width, pixel_border, z_eps,
    sigma_pixel, sigma_depth, C_thresh, Q_thresh, max_iter, delta_thresh

The step functions are the same minus `max_iter` and `delta_thresh`.

The public GN functions must mutate `Twc` in place through pose retraction and return a tuple whose first element is the final `dx` tensor. That behavior is what the existing tests and callers expect.

The step functions must return a pair of tensors `(Hs, gs)` where:

- `Hs` has shape `(4, num_edges, 7, 7)`
- `gs` has shape `(2, num_edges, 7)`

Those shapes are enforced by `packages/mast3r-slam/tests/test_gn_step_api.py`.

## What the Original CUDA Backend Actually Does

The scratch Mojo implementation should copy behavior, not syntax, from the original backend.

The original helper math in `gn_kernels.cu` includes:

- `huber`
- quaternion composition and inversion
- `actSO3`
- `actSim3`
- `relSim3`
- `apply_Sim3_adj_inv`
- `expSO3`
- `expSim3`
- `retrSim3`

The original low-level kernels are:

- `pose_retr_kernel`
- `point_align_kernel`
- `ray_align_kernel`
- `calib_proj_kernel`

The original host-side GN implementation also contains:

- `SparseBlock`
- `get_unique_kf_idx`
- `create_inds`
- sparse matrix assembly from the `Hs` blocks and `gs` vectors
- an Eigen `SimplicialLLT` solve
- the public multi-iteration GN loop

The clean Mojo rewrite should preserve the public behavior but not blindly copy the old organization. In particular, the Mojo backend should separate low-level kernels from public orchestration. That means:

- `gn_kernels.mojo` should contain the math helpers and low-level kernels.
- `gn.mojo` should contain the public GN orchestration, indexing setup, solve path, and exported wrappers.
- `mast3r_slam_mojo_backends.mojo` should only export the Mojo functions to Python.

## Plan of Work

The first milestone is to re-create the public boundary and the ray path. Define the Mojo shared-library exports and the Python backend selector first, because the rest of the repository only needs those names to exist. Create `gn_backends.py` so that it prefers the Mojo shared library when it is importable and falls back to the CUDA extension otherwise. Make this boundary mirror the C++ function names exactly. Do not introduce “current” versus “idiomatic” dual naming. This scratch implementation should have only one Mojo GN path.

Then implement the low-level pose retraction and the rays step kernel in `gn_kernels.mojo`. The pose retraction path is small and easy to verify because the original `pose_retr_kernel` works on one tensor pair and does not depend on the rest of the GN system. The rays step kernel is the mandatory first accumulation kernel because the live signoff fixtures exercise the rays path and the real benchmark harness already exists for it.

Once the rays step kernel exists, implement the public GN orchestration in `gn.mojo`. This file must do the same logical work as the original C++ host code: compute unique keyframe indices, create the optimizer-space index remapping, run the step kernel each iteration, assemble the linear system, solve for `dx`, and retract the poses in place. The implementation does not need to use the same internal data structure names as `SparseBlock`, but it must preserve the same observable behavior.

The scratch implementation should start with a CPU-side solve path for the public GN loop. That is the safer starting point because the original backend already moves into CPU/Eigen for the solve, and earlier experiments showed that a GPU dense-solve rewrite was worse for these tiny systems. Use the original C++ behavior as the conceptual guide: build the block system from `Hs` and `gs`, solve in double precision, and cast the resulting `dx` back to the output device dtype.

After the public rays path is correct, wire it into the live caller through `gn_backends.py` and rerun the real rays benchmarks. Only once the real rays public path meets the target should you widen scope to points and calib. When you do widen scope, port the points and calib public orchestration in the same style first. If the real repository still lacks a calibrated real-fixture path, it is acceptable to keep the points and calib step surfaces CUDA-backed temporarily while still making the public Mojo orchestration canonical.

When points and calib are addressed, do not create extra Mojo variants. If you need a temporary fallback, keep it at the Python backend selector level by calling the CUDA extension when the Mojo symbol is missing.

## Required Implementation Details

The following behavior must be preserved because it is visible in the original code and in the existing tests.

The pose state is laid out as eight floats per pose, with translation first, quaternion next, and scale last. The update state is seven floats per pose update. The first `pin` poses are fixed. In the current caller logic this is always `1`, and the public GN loop should keep that behavior unless the caller interface changes explicitly.

The rays kernel computes four residual rows per matched point: three direction residuals and one distance residual. The kernel builds the 14-column stacked Jacobian row made from the `i` and `j` pose blocks, accumulates the upper-triangular Hessian entries, and accumulates the two 7-vectors for the right-hand side. The Mojo code does not need to preserve the exact local variable names, but it must preserve the same accumulation order closely enough that the parity tolerances pass.

The public GN loop must stop either when `max_iter` is reached or when `norm(dx) < delta_thresh`. The output `dx` tensor must be returned even when the loop terminates early.

The calib path has one important indexing detail that caused a bug in earlier work: in the public calib argument list, `ii` and `jj` are at positions `4` and `5` because `K` is inserted before them. Do not assume all public GN entrypoints place `ii` and `jj` at the same argument indices.

The calib step call also requires forwarding `Q_thresh`. The original `gauss_newton_calib_step` takes seventeen arguments total, not sixteen.

## Concrete Steps

Run all commands from the repository root unless otherwise stated.

Read the original reference files first:

    sed -n '1,220p' packages/mast3r-slam/mast3r_slam/backend/include/gn.h
    sed -n '1,260p' packages/mast3r-slam/mast3r_slam/backend/src/gn.cpp
    sed -n '1,220p' packages/mast3r-slam/mast3r_slam/backend/src/gn_kernels.cu
    sed -n '220,620p' packages/mast3r-slam/mast3r_slam/backend/src/gn_kernels.cu

Create or replace the scratch Mojo implementation files:

    packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo
    packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo
    packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo
    packages/mast3r-slam/mast3r_slam/gn_backends.py

Build the shared library after each meaningful Mojo change:

    pixi run -e mast3r-slam-dev _build-mojo-kernels

Run the GN smoke suite after each public boundary change:

    pixi run -e mast3r-slam-dev pytest \
      packages/mast3r-slam/tests/test_gn_fixture_utils.py \
      packages/mast3r-slam/tests/test_gn_step_api.py -q

Benchmark the real rays fixtures from the package directory once the public rays path exists:

    cd packages/mast3r-slam
    pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py \
      artifacts/gn-fixtures/verify-normal-apt/rays-000.pt \
      --warmup 5 --runs 20

    cd packages/mast3r-slam
    pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py \
      artifacts/gn-fixtures/verify-livingroom/rays-000.pt \
      --warmup 5 --runs 20

Run the synthetic GN benchmark as supporting evidence only:

    pixi run -e mast3r-slam-dev python packages/mast3r-slam/tools/bench_gn_kernels.py

The expected smoke-test success line is:

    13 passed

The accepted real rays public benchmark targets from earlier work are:

    normal-apt current≈43.789 ms idiomatic≈43.931 ms ratio≈1.003x
    livingroom current≈43.798 ms idiomatic≈43.713 ms ratio≈0.998x

The scratch implementation does not need to reproduce those exact decimal values on the first try, but the final public rays Mojo path must be within `1.10x` of the CUDA public path on those real fixtures and must preserve the tiny drift envelope already established by the repository tests and benchmarks.

## Validation and Acceptance

This scratch-build plan is complete only when all of the following are true.

The Mojo shared library builds cleanly.

The GN smoke suite passes.

The public Mojo rays path is the default selected path in `gn_backends.py` unless CUDA is explicitly forced.

The real rays fixture benchmark reports the Mojo public path within 10% of CUDA on both:

- `artifacts/gn-fixtures/verify-normal-apt/rays-000.pt`
- `artifacts/gn-fixtures/verify-livingroom/rays-000.pt`

The real rays fixture benchmark still reports a tiny parity envelope relative to CUDA, approximately:

- translation max abs `~1e-7`
- quaternion max abs `~1e-8`
- scale max abs `~1e-8`
- `dx` max abs `~1e-7`

The public Python call sites in `global_opt.py` must not require call-signature changes.

If points and calib are not fully ported at the step level yet, that must be explicit in the resulting code and comments, and the public Mojo orchestration must still remain the canonical path to keep.

## Idempotence and Recovery

The build, test, and benchmark commands above are safe to rerun. The real fixture benchmarks are read-only. If a scratch implementation step breaks the shared library build, do not keep layering new wrappers on top. Revert only the last Mojo-file edits, rebuild, and then reintroduce the change in smaller pieces.

If a contributor gets stuck on points or calib after the public rays path is working, they should not block release of the canonical rays Mojo path. They may temporarily keep points and calib on the CUDA step functions through the shared public Mojo orchestration, provided that this status is documented clearly in code comments and in the benchmark notes.

## Artifacts and Notes

The single most important implementation truth is this: the new agent should not try to recreate every historical Mojo experiment. It should recreate one clean GN backend from the original C++/CUDA contract.

The single most important technical truth is this: the public rays path is the real signoff path in the current repository. If the scratch implementation gets that path correct, fast, and default-selected, the rest of the work becomes incremental instead of existential.

The single most important coding truth is this: keep one Mojo implementation. Any dual-path setup is temporary scaffolding, not the desired end state.

Revision note: this ExecPlan was written specifically to replace the earlier canonical artifact with a true from-scratch implementation spec. It assumes the current Mojo GN code is unavailable and that the only implementation reference is the original untouched CUDA/C++ backend.
