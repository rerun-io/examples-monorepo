# Add an Idiomatic Mojo GN Rays Kernel Variant Beside CUDA and the Current Mojo Port

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained under the rules in `.agents/PLANS.md`.

## Purpose / Big Picture

After this change, `mast3r-slam` will have three separately benchmarkable implementations of the GN ray-step kernel instead of only two. The repository will keep the existing CUDA oracle, keep the current validated Mojo implementation, and add a new idiomatic-from-scratch Mojo kernel variant that is structured around current Modular best practices. That gives us a clean place to try a better design without risking the already validated Mojo path.

The user-visible proof is a new benchmark mode that can run all three variants on the same captured real GN fixture and report accuracy and throughput side by side. The rewrite is only accepted if the new idiomatic variant matches CUDA within the existing tolerances and stays within the real-workload throughput gate that has been used for GN signoff.

## Progress

- [x] (2026-04-09 17:26 CDT) Reviewed the current GN cleanup ExecPlan, the GN handoff document, and the backend selector to confirm that only CUDA and one Mojo implementation exist today.
- [x] (2026-04-09 17:31 CDT) Decided that the rewrite should target only `gauss_newton_rays_step` first and must coexist with the current Mojo kernel instead of replacing it.
- [ ] Add a new kernel module and Python-visible entrypoint for the idiomatic rewrite while keeping the current Mojo kernel untouched.
- [ ] Extend backend selection and benchmark tooling so CUDA, current Mojo, and idiomatic Mojo can be forced independently.
- [ ] Add accuracy comparisons for all three paths on captured real GN fixtures.
- [ ] Add throughput comparisons for all three paths on the same captured real GN fixtures and example runs.
- [ ] Promote the idiomatic kernel to the default only if it passes the same real-workload signoff gate as the current Mojo kernel.

## Surprises & Discoveries

- Observation: the current repository already meets the real-workload gate with the existing Mojo path, so a rewrite is not required for correctness or current release signoff.
  Evidence: `packages/mast3r-slam/docs/mojo-max-cleanup-execplan.md` records the validated real GN public row as `selected 21.891 ms` versus CUDA `51.998 ms`, ratio `0.421`, with tiny pose drift.

- Observation: the current GN kernel style is intentionally thread-parallel and scalar within each thread.
  Evidence: `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo` assigns one block to one edge-partial pair, then each thread walks points in a strided loop and reduces partials at block scope.

- Observation: the current backend selector only distinguishes CUDA, Mojo shared-lib, and MAX `pose_retr`.
  Evidence: `packages/mast3r-slam/mast3r_slam/gn_backends.py` can force CUDA through `MAST3R_SLAM_FORCE_CUDA_BACKENDS`, but it has no concept of multiple Mojo GN kernel variants yet.

## Decision Log

- Decision: the rewrite should be additive and must not replace the current Mojo kernel during development.
  Rationale: the current Mojo path is already validated on real workloads. Replacing it directly would destroy the best comparison baseline and increase risk.
  Date/Author: 2026-04-09 / Codex

- Decision: only `gauss_newton_rays_step` should get a third implementation in the first milestone.
  Rationale: `gauss_newton_rays_step` is the hot GN surface that motivated all prior tuning work. Rewriting points or calib first would widen scope without helping the real bottleneck.
  Date/Author: 2026-04-09 / Codex

- Decision: the new kernel should be benchmark-selected explicitly rather than hidden behind heuristics.
  Rationale: we need deterministic apples-to-apples measurement across three implementations, so backend selection must be controllable from tests and benchmark commands.
  Date/Author: 2026-04-09 / Codex

## Outcomes & Retrospective

This section will be updated after implementation. The intended outcome is not just a new kernel, but a safe experimental lane where the team can compare three implementations under the same accuracy and throughput harness.

## Context and Orientation

The GN ray-step surface in this repository is the function that accumulates per-edge Jacobian and Hessian terms for Gauss-Newton pose updates. Today there are two implementations worth comparing:

`packages/mast3r-slam/mast3r_slam/backend/src/gn_kernels.cu` is the CUDA oracle. It is the correctness and performance reference.

`packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo` is the current Mojo implementation. It is already wired through the shared-library backend in `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo` and is selected from Python in `packages/mast3r-slam/mast3r_slam/gn_backends.py`.

The new work in this plan adds a third implementation that is explicitly experimental but fully real: an idiomatic Mojo rewrite of `gauss_newton_rays_step` that is written as if the current Mojo implementation did not exist. “Idiomatic” here means using current Modular guidance on clear structure, explicit separation of data movement and compute, and careful use of thread-level parallelism versus SIMD inside the thread. It does not mean rewriting the entire GN stack around new abstractions all at once.

A captured real GN fixture is a `.pt` file containing the tensors seen at the `_backends.gauss_newton_*` boundary during a real run. This repository already uses such fixtures for signoff in `packages/mast3r-slam/tools/bench_gn_real_fixtures.py`.

## Plan of Work

First, add a new Mojo kernel module for the experimental rewrite, for example `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels_idiomatic.mojo`. Keep it focused on `gauss_newton_rays_step` only. Do not edit the current kernel logic in `gn_kernels.mojo` except for any tiny shared helper extraction that is truly common to both implementations. The new file should have its own entrypoint function, such as `gauss_newton_rays_step_kernel_idiomatic`, and its own module-level comments that explain the execution model.

Second, add a new Python-visible shared-library wrapper in `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo` and export it from `packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo`. The current Mojo wrapper should keep its existing name. The new wrapper should get a distinct name, for example `gauss_newton_rays_step_idiomatic_py`, exported to Python as `gauss_newton_rays_step_idiomatic`.

Third, update `packages/mast3r-slam/mast3r_slam/gn_backends.py` so the selection logic can choose among three ray-step implementations. Do this with explicit environment variables, not silent heuristics. One safe model is:

    MAST3R_SLAM_FORCE_CUDA_BACKENDS=1
    MAST3R_SLAM_FORCE_MOJO_RAYS=current
    MAST3R_SLAM_FORCE_MOJO_RAYS=idiomatic

When neither variable is set, keep the current default behavior unchanged. The new variant is opt-in until it passes signoff.

Fourth, extend `packages/mast3r-slam/tests/test_gn_step_api.py` and `packages/mast3r-slam/tools/bench_gn_real_fixtures.py` so they can exercise all three variants explicitly. The benchmark output should gain one more row or one more selectable mode for the idiomatic Mojo kernel. The accuracy path must compare:

- CUDA vs current Mojo
- CUDA vs idiomatic Mojo
- current Mojo vs idiomatic Mojo

The public GN benchmark should continue to use the real fixture gate, because that is the release-signoff criterion already established in this repository.

Fifth, only if the idiomatic kernel wins or ties on real workloads should the default selector be changed. If it fails the throughput or accuracy gate, keep it in-tree as an experiment and document the outcome in this plan.

## Concrete Steps

Run all commands from the repository root `/var/tmp/vibe-kanban/worktrees/4658-mast3r-slam-mojo/examples-monorepo`.

Build the Mojo shared library after adding the new kernel module and exports:

    pixi run -e mast3r-slam-dev _build-mojo-kernels

Run the GN tests with the current default selection:

    pixi run -e mast3r-slam-dev pytest \
      packages/mast3r-slam/tests/test_gn_fixture_utils.py \
      packages/mast3r-slam/tests/test_gn_step_api.py -q

Add explicit runs for each ray-step implementation. The exact environment variables may be adjusted during implementation, but the final commands must have this shape:

    MAST3R_SLAM_FORCE_CUDA_BACKENDS=1 \
      pixi run -e mast3r-slam-dev python \
      packages/mast3r-slam/tools/bench_gn_real_fixtures.py \
      packages/mast3r-slam/artifacts/gn-fixtures/verify-base/rays-000.pt \
      --warmup 5 --runs 20

    MAST3R_SLAM_FORCE_MOJO_RAYS=current \
      pixi run -e mast3r-slam-dev python \
      packages/mast3r-slam/tools/bench_gn_real_fixtures.py \
      packages/mast3r-slam/artifacts/gn-fixtures/verify-base/rays-000.pt \
      --warmup 5 --runs 20

    MAST3R_SLAM_FORCE_MOJO_RAYS=idiomatic \
      pixi run -e mast3r-slam-dev python \
      packages/mast3r-slam/tools/bench_gn_real_fixtures.py \
      packages/mast3r-slam/artifacts/gn-fixtures/verify-base/rays-000.pt \
      --warmup 5 --runs 20

Run the example-level checks only after the real fixture benchmark passes:

    pixi run -e mast3r-slam-dev livingroom-base

    pixi run -e mast3r-slam-dev example-base

The final implementation of this plan must update this section with actual observed outputs for all three variants.

## Validation and Acceptance

This rewrite is only worth keeping if it earns its place with data. Acceptance is therefore stricter than “the code builds.”

Correctness acceptance:

- The GN tests must pass.
- The idiomatic Mojo ray-step must match CUDA within the same tolerance envelope already used by the repository’s GN checks.
- On the real captured `rays-000` fixture, the public GN path using the idiomatic kernel must preserve the tiny pose/update drift envelope already achieved by the current Mojo path.

Throughput acceptance:

- The hard gate remains the real workload gate, not synthetic microbenchmarks.
- On captured real GN fixtures, the idiomatic public path must be within the same `<= 1.05x` CUDA rule previously established, and should ideally match or beat the current Mojo path.
- On the full example runs, the idiomatic path must remain within the same real-workload signoff envelope used for `livingroom-base` and `example-base`.

Promotion acceptance:

- The idiomatic kernel only becomes the default if it matches accuracy and is at least as good as the current Mojo path on the real fixture benchmark.
- If it is slower or less stable, keep it behind explicit selection and document why.

## Idempotence and Recovery

This plan is intentionally additive. The CUDA backend remains untouched. The current Mojo implementation remains untouched except for any minimal plumbing needed to expose the third variant. If the rewrite fails, delete or disable only the new selector path and the new experimental kernel module; the existing CUDA and current Mojo paths should continue to work.

Benchmark commands are safe to rerun. The shared-library build overwrites the `.so` in place. The captured real GN fixtures are read-only inputs.

## Artifacts and Notes

The most important artifact from this plan is a three-way comparison report for the same real fixture. The final version of this document must include a concise transcript with rows for:

- CUDA
- current Mojo
- idiomatic Mojo

and must record the accuracy deltas for the idiomatic variant.

The relevant Modular guidance that motivated this plan is already reflected in repo notes, but the actionable takeaway is simple: keep responsibilities separated, keep the execution model explicit, and do not force SIMD structure where the work is irregular and branch-heavy unless measurements prove it helps.

## Interfaces and Dependencies

At the end of the first implementation milestone, the repository must still expose:

    gauss_newton_rays_step(*args: Any) -> tuple[Any, ...]

through `packages/mast3r-slam/mast3r_slam/gn_backends.py` for the existing selector path.

The Mojo shared library must additionally expose a distinct experimental entrypoint, for example:

    gauss_newton_rays_step_idiomatic(args_obj: PythonObject) raises -> PythonObject

or an equivalently named wrapper that Python can call independently.

The benchmark and test tools must gain an interface for explicit variant selection. That selection must distinguish all three implementations in the repository:

- CUDA oracle
- current Mojo kernel
- idiomatic Mojo rewrite

Revision note: created this ExecPlan after the current Mojo GN path had already met the real-workload validation gate. The purpose of this plan is not to rescue a broken implementation; it is to create a safer experimental lane for a cleaner kernel design without sacrificing the proven baseline.
