# Port the Full GN Mojo Backend to an Idiomatic Implementation Within 10% on Real Fixtures

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained under the rules in `.agents/PLANS.md`.

## Purpose / Big Picture

After this change, the Mojo GN backend in `mast3r-slam` should no longer depend on the current “pragmatic” Mojo implementation for production ray, point, or calibration behavior. Instead, the repository should have an idiomatic Mojo implementation of the full GN surface, selected through the existing `MAST3R_SLAM_FORCE_MOJO_RAYS=idiomatic` mechanism for the rays path and equivalent direct idiomatic entrypoints for the remaining GN surfaces. The user-visible proof is that real captured GN fixtures from both normal-apt and livingroom report the idiomatic public path within 10% of the current Mojo path, while matching the existing pose/update accuracy envelope.

This plan is broader than the earlier idiomatic experiment. It is not limited to `gauss_newton_rays_step`. It covers the full GN stack that currently matters in this repository: the step kernels, the public GN wrappers, the host-side solve/orchestration path in Mojo, and the Python-visible export surface. The CUDA backend remains the correctness oracle and fallback while the idiomatic Mojo port is brought up.

## Progress

- [x] (2026-04-09 17:31 CDT) Earlier work established that only the rays path had an explicit idiomatic Mojo variant, while points and calib still used the current Mojo or CUDA-backed implementations.
- [x] (2026-04-10 00:15 CDT) Earlier work also established that the selectable idiomatic rays path can be brought within 5% of the current Mojo path on real fixtures by reusing the current Mojo implementation behind the idiomatic export names.
- [ ] Replace that temporary export-level convergence with real idiomatic Mojo implementations for the full GN surface: rays, points, calib, and the public GN orchestration path.
- [ ] Extend the benchmark and fixture coverage so idiomatic-vs-current comparisons exist for all relevant GN public surfaces, not only rays.
- [ ] Validate the idiomatic GN implementation on real captured normal-apt and livingroom fixtures, plus the bounded example runs, against the relaxed `<= 1.10x` target.
- [ ] Decide whether the idiomatic path is ready to become the default Mojo implementation, or whether specific surfaces must stay on the current implementation longer.

## Surprises & Discoveries

- Observation: two obvious local rewrites inside the standalone idiomatic ray kernel did not close the performance gap.
  Evidence: `packages/mast3r-slam/docs/gn-rays-idiomatic-optimization-execplan.md` records failed attempts based on row-buffer reuse and helper-boundary rewrites, both of which preserved accuracy but remained slower on the real `rays-000` fixture.

- Observation: the export boundary itself contributes measurable overhead differences, even when two paths ultimately execute the same underlying Mojo logic.
  Evidence: the earlier plan recorded that binding the idiomatic Python-visible names directly to the current Mojo functions reduced the normal-apt idiomatic/current public ratio to `1.031x` and the livingroom ratio to `1.003x`.

- Observation: the repository currently has reliable real-fixture capture and benchmark tooling for rays, but not the same depth of checked-in fixture corpus or benchmark reporting for points and calib.
  Evidence: `packages/mast3r-slam/tools/bench_gn_real_fixtures.py` already compares CUDA, current Mojo, and idiomatic Mojo for rays when the corresponding exports exist, while `packages/mast3r-slam/tools/bench_gn_kernels.py` still reports only the selected backend for `calib_step` and has no points benchmark row.

- Observation: the current Mojo public GN path in `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo` already contains a substantial amount of host-side orchestration, including index preparation, dense block assembly on the Python side, Cholesky solve, and pose retraction dispatch.
  Evidence: `gn.mojo` defines `get_unique_kf_idx`, `create_inds`, `solve_step_system`, `gauss_newton_impl`, and the public/exported GN wrapper functions.

## Decision Log

- Decision: this plan uses the current Mojo implementation as the baseline for performance comparison and the CUDA backend as the correctness oracle.
  Rationale: the current Mojo path is already validated on real fixtures and example runs, while CUDA remains the external reference implementation for numerical parity.
  Date/Author: 2026-04-09 / Codex

- Decision: the performance target for this broader idiomatic migration is relaxed to `<= 1.10x` the current Mojo implementation on real fixtures.
  Rationale: the user explicitly relaxed the target for this broader rewrite. A 10% window is tight enough to force serious measurement, but wide enough to make a full idiomatic migration plausible.
  Date/Author: 2026-04-09 / Codex

- Decision: the full idiomatic migration should proceed surface by surface in this order: rays public path, calib step/public path, points step/public path, then common helper cleanup.
  Rationale: rays already has the most measurement infrastructure and is the hot path the repository has tuned most heavily. Calib and points can then reuse the proven integration patterns and benchmark updates.
  Date/Author: 2026-04-09 / Codex

- Decision: the earlier export-level convergence for the idiomatic rays path is an acceptable temporary bridge, but not the final state for this plan.
  Rationale: the goal here is a real idiomatic implementation of the full GN surface, not just an alias of the current Mojo path. The bridge remains useful while the real idiomatic surfaces are brought up incrementally.
  Date/Author: 2026-04-09 / Codex

## Outcomes & Retrospective

This section should be updated after each milestone. At completion it must summarize which GN surfaces are truly idiomatic Mojo, which still depend on the current Mojo implementation, whether the 10% performance target was met on both normal-apt and livingroom, and whether the default export selection should move to the idiomatic implementation.

## Context and Orientation

The GN backend in this repository spans multiple layers. A newcomer needs to know the following files first.

`packages/mast3r-slam/mast3r_slam/backend/src/gn_kernels.cu` is the CUDA oracle. It contains the original GPU kernels and the extension-backed GN behavior that the repository historically depended on.

`packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo` contains the current validated Mojo GN kernels. This is the implementation that currently wins on real fixtures and bounded example runs.

`packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels_idiomatic.mojo` contains the experimental idiomatic ray-step kernel. It is useful as a design reference, but by itself it has not matched the current Mojo implementation on real fixtures.

`packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo` contains the Mojo-side GN orchestration and Python-visible wrappers. This file is where the current and idiomatic step kernels get bound into the full public `gauss_newton_*` behavior.

`packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo` exports the Mojo functions to Python under the module name `mast3r_slam_mojo_backends`.

`packages/mast3r-slam/mast3r_slam/gn_backends.py` selects among CUDA, current Mojo, and idiomatic Mojo from Python. This is the file that honors `MAST3R_SLAM_FORCE_MOJO_RAYS` and the CUDA force flag.

`packages/mast3r-slam/tools/bench_gn_real_fixtures.py` is the real-fixture benchmark tool. It runs public and step-level comparisons and prints CUDA, current Mojo, and idiomatic Mojo rows when those exports exist. Its output is the main acceptance evidence for this plan.

`packages/mast3r-slam/artifacts/gn-fixtures/verify-normal-apt/rays-000.pt` and `packages/mast3r-slam/artifacts/gn-fixtures/verify-livingroom/rays-000.pt` are the current checked-in real rays fixtures. Equivalent real fixtures must be added for the other GN surfaces as part of this plan.

In this repository, a “real fixture” means a `.pt` file captured at the `_backends.gauss_newton_*` boundary during a real SLAM run. It contains the actual tensors and scalar parameters used by the GN call, which makes it suitable for reproducible accuracy and throughput comparisons.

An “idiomatic Mojo implementation” in this plan means a Mojo implementation whose code structure reflects the repository’s current best understanding of clear Modular-style organization, rather than merely exposing the current Mojo implementation under a second export name.

## Plan of Work

The first milestone is to reintroduce a real idiomatic rays implementation behind the idiomatic export names without breaking the current performance-validated path. Start in `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo` and `packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo`. Keep the current export names and selector behavior intact, but restore the idiomatic rays exports so they no longer alias the current implementation. Only do this once there is enough benchmark coverage to compare the real idiomatic rays path against the current Mojo path on both normal-apt and livingroom fixtures.

The second milestone is to make `bench_gn_real_fixtures.py` and `bench_gn_kernels.py` fully surface-aware. Add explicit rows or modes for points and calib so the repository can measure current Mojo versus idiomatic Mojo on those surfaces too. Add any missing fixture capture support in the live code if the current capture utilities do not already record the necessary inputs for points or calib.

The third milestone is to port the calib path idiomatically. Create idiomatic step and public wrappers for `gauss_newton_calib_step` and `gauss_newton_calib`. Reuse the current validated host-side GN orchestration patterns in `gn.mojo` where practical, but organize the code so the idiomatic calib path is a real Mojo implementation, not a thin alias of the current wrapper.

The fourth milestone is to port the points path idiomatically. Create idiomatic step and public wrappers for `gauss_newton_points_step` and `gauss_newton_points`, again keeping CUDA as the oracle and fallback.

The fifth milestone is to consolidate the public GN orchestration. Once idiomatic rays, calib, and points all exist and pass their real-fixture checks, split `gn.mojo` into clearly named current-versus-idiomatic helpers or refactor it into smaller shared pieces so the idiomatic public path is easy to audit. At the end of this milestone, the idiomatic path should be a complete, real Mojo implementation of the GN surface rather than a set of aliases.

At each milestone, rebuild the Mojo shared library, rerun the GN tests, rerun the real-fixture benchmark on both normal-apt and livingroom, and record the exact output in this file. If any idiomatic surface misses the 10% target, keep it opt-in and continue the migration with the remaining surfaces rather than deleting working code.

## Concrete Steps

Run all commands from the repository root `/var/tmp/vibe-kanban/worktrees/4658-mast3r-slam-mojo/examples-monorepo` unless otherwise stated.

Rebuild the Mojo shared library after every meaningful Mojo change:

    pixi run -e mast3r-slam-dev _build-mojo-kernels

Run the existing GN correctness tests:

    pixi run -e mast3r-slam-dev pytest \
      packages/mast3r-slam/tests/test_gn_fixture_utils.py \
      packages/mast3r-slam/tests/test_gn_step_api.py -q

Capture fresh real fixtures from the package directory so relative paths resolve correctly:

    cd packages/mast3r-slam
    rm -rf artifacts/gn-fixtures/verify-normal-apt
    mkdir -p artifacts/gn-fixtures/verify-normal-apt
    MAST3R_SLAM_GN_CAPTURE_DIR=artifacts/gn-fixtures/verify-normal-apt \
    MAST3R_SLAM_GN_CAPTURE_LIMIT=1 \
    pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
      --dataset data/normal-apt-tour.mp4 \
      --img-size 512 \
      --config config/base.yaml \
      --max-frames 60 \
      --no-viz

    cd packages/mast3r-slam
    rm -rf artifacts/gn-fixtures/verify-livingroom
    mkdir -p artifacts/gn-fixtures/verify-livingroom
    MAST3R_SLAM_GN_CAPTURE_DIR=artifacts/gn-fixtures/verify-livingroom \
    MAST3R_SLAM_GN_CAPTURE_LIMIT=1 \
    pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
      --dataset data/livingroom-tour.mp4 \
      --img-size 512 \
      --config config/base.yaml \
      --max-frames 60 \
      --no-viz

Benchmark one real fixture at a time from the package directory:

    cd packages/mast3r-slam
    pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py \
      artifacts/gn-fixtures/verify-normal-apt/rays-000.pt \
      --warmup 5 --runs 20

    cd packages/mast3r-slam
    pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py \
      artifacts/gn-fixtures/verify-livingroom/rays-000.pt \
      --warmup 5 --runs 20

Run the synthetic benchmark for supporting evidence:

    MAST3R_SLAM_FORCE_MOJO_RAYS=current \
      pixi run -e mast3r-slam-dev python packages/mast3r-slam/tools/bench_gn_kernels.py

    MAST3R_SLAM_FORCE_MOJO_RAYS=idiomatic \
      pixi run -e mast3r-slam-dev python packages/mast3r-slam/tools/bench_gn_kernels.py

As this plan progresses, add the corresponding real-fixture capture and benchmark commands for calib and points once those surfaces gain explicit idiomatic implementations.

## Validation and Acceptance

This plan is complete only when all of the following are true.

The Mojo shared library builds successfully after the idiomatic changes.

The GN tests pass.

The idiomatic public rays path is at most `1.10x` the current Mojo public rays path on the normal-apt real fixture.

The idiomatic public rays path is at most `1.10x` the current Mojo public rays path on the livingroom real fixture.

Equivalent real-fixture comparisons exist for calib and points once those surfaces are ported idiomatically, and those comparisons also meet the `1.10x` bound.

The idiomatic public paths preserve the existing tiny pose/update drift envelope relative to CUDA on the real fixtures.

If a surface fails the 10% target, it may remain opt-in temporarily, but this file must record the exact measured gap and the next step chosen to close it.

## Idempotence and Recovery

The build and benchmark commands in this plan are safe to rerun. The fixture capture commands remove and recreate their target directories before writing a new `rays-000.pt`, so repeated runs replace the old artifacts deterministically. If an idiomatic migration step hurts performance or accuracy, restore only the touched Mojo files and rerun the same validation commands. Keep the current validated Mojo implementation and the CUDA backend intact until the idiomatic migration is fully signed off.

## Artifacts and Notes

The most important artifacts for this plan are the real benchmark transcripts for normal-apt and livingroom, plus any added calib and points fixtures. Each transcript should report CUDA, current Mojo, and idiomatic Mojo rows, including the step and public timings and the pose/update drift metrics.

The earlier plans remain relevant reference material:

`packages/mast3r-slam/docs/gn-rays-idiomatic-rewrite-execplan.md` explains how the experimental idiomatic ray kernel was added beside the current Mojo path.

`packages/mast3r-slam/docs/gn-rays-idiomatic-optimization-execplan.md` records the failed local optimization attempts and the later export-level convergence used to satisfy the 5% real-fixture goal.

This plan intentionally builds on those results instead of repeating them.

## Interfaces and Dependencies

At the end of this plan, the following Python-visible interfaces must still exist in `mast3r_slam_mojo_backends`:

    gauss_newton_rays_step(...)
    gauss_newton_rays_step_idiomatic(...)
    gauss_newton_rays_impl(...)
    gauss_newton_rays_impl_idiomatic(...)
    gauss_newton_points_impl(...)
    gauss_newton_calib_impl(...)

If points and calib gain explicit idiomatic variants, they should follow the same naming pattern:

    gauss_newton_points_impl_idiomatic(...)
    gauss_newton_calib_impl_idiomatic(...)

The selector logic in `packages/mast3r-slam/mast3r_slam/gn_backends.py` must remain deterministic and externally controllable. Environment-driven selection is acceptable, but implicit heuristics are not.

Revision note: created after the repository had already met a 5% real-fixture goal for the idiomatic rays selector by aliasing it to the current Mojo implementation. This new plan raises the scope to a real idiomatic migration of the full GN surface, while relaxing the acceptance threshold to 10%.
