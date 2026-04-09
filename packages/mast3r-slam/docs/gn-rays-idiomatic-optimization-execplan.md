# Optimize the Idiomatic GN Rays Kernel Until It Is Close to the Current Mojo Baseline

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained under the rules in `.agents/PLANS.md`.

## Purpose / Big Picture

After this change, the experimental idiomatic Mojo GN ray-step kernel will still be a separate selectable implementation, but it should be measurably closer to the current production Mojo kernel instead of lagging it by a clear margin. The user-visible proof is the real-fixture benchmark report from `packages/mast3r-slam/tools/bench_gn_real_fixtures.py`, which must continue to show CUDA, current Mojo, and idiomatic Mojo side by side while the idiomatic row moves closer to the current Mojo row without losing accuracy.

This plan does not attempt a broad GN rewrite. It focuses only on the hot kernel in `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels_idiomatic.mojo`, because that is the only place where the experimental path differs materially from the current validated Mojo implementation.

## Progress

- [x] (2026-04-09 22:56 CDT) Re-read `.agents/PLANS.md`, `mojo-syntax`, and `mojo-gpu-fundamentals` before editing Mojo code.
- [x] (2026-04-09 23:01 CDT) Compared `gn_kernels_idiomatic.mojo` against `gn_kernels.mojo` and identified the clearest avoidable hot-path cost: a fresh `InlineArray[Float32, 14]` is constructed inside `accumulate_pair_row()` four times per point.
- [x] (2026-04-09 23:08 CDT) Implemented the reusable row-scratch experiment in the idiomatic kernel without changing selector behavior or public wrappers.
- [x] (2026-04-09 23:10 CDT) Rebuilt the Mojo shared library and reran the GN correctness tests; the build succeeded and `11 passed`.
- [x] (2026-04-09 23:11 CDT) Re-benchmarked CUDA, current Mojo, and idiomatic Mojo on the real `rays-000` fixture and found that the scratch-row experiment made the idiomatic kernel slower, not faster.
- [x] (2026-04-09 23:14 CDT) Reverted the kernel edit to avoid regressing the experimental variant and kept this ExecPlan as the record of what was tried and why it was discarded.
- [x] (2026-04-09 23:24 CDT) Re-opened the idiomatic and current kernels and confirmed that the next credible gap is the top-level hot-loop helper boundary itself, not just the per-row temporary buffer.
- [x] (2026-04-09 23:28 CDT) Replaced the top-level `accumulate_pair_row()` helper with a kernel-local compile-time closure and reran the full validation loop.
- [x] (2026-04-09 23:29 CDT) Confirmed that the helper-boundary rewrite also kept accuracy but still ran slower than the current Mojo kernel on `rays-000`.
- [x] (2026-04-09 23:31 CDT) Reverted the helper-boundary rewrite so the repository stays at the best measured idiomatic implementation.
- [x] (2026-04-09 23:31 CDT) Concluded that the idiomatic kernel should remain a readability-first experiment until a profiler-guided or larger structural rewrite is attempted.

## Surprises & Discoveries

- Observation: the idiomatic kernel is not algorithmically different from the current Mojo kernel on the hot ray-step path; most of the gap appears to come from how the work is organized inside the kernel body.
  Evidence: both kernels use the same block geometry, the same shared relative-pose setup, the same scalar per-thread accumulation strategy, and the same block reduction shape. The idiomatic kernel introduces one additional `InlineArray[Float32, 14]` allocation inside `accumulate_pair_row()` for every residual row.

- Observation: removing the per-row `InlineArray[Float32, 14]` construction did not improve the idiomatic kernel on the real fixture.
  Evidence:
    before the experiment, `mojo-idiomatic public` on `rays-000` was `24.239 ms`;
    after the experiment, the same row measured `25.489 ms`;
    `mojo-current public` remained faster at `23.646 ms`.
    Accuracy stayed unchanged at `dtrans 2.98e-07`, `dquat 1.68e-08`, `dscale 5.96e-08`, `ddx 1.75e-07`.

- Observation: after ruling out the local row-buffer allocation, the biggest remaining code-shape difference is that the idiomatic kernel keeps the hot row accumulation as a top-level helper with a very wide argument list, while the current kernel uses a nested compile-time closure and a thread-local scratch buffer.
  Evidence: `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels_idiomatic.mojo` still routes each residual row through `accumulate_pair_row(...)`, while `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo` defines `@parameter def accumulate_row(...)` inside the kernel loop and reuses `jxv`.

- Observation: changing the idiomatic kernel to use the same nested compile-time closure shape as the current kernel did not close the gap either.
  Evidence:
    after that rewrite, `mojo-current public` measured `24.167 ms` and `mojo-idiomatic public` measured `25.612 ms` on `rays-000`;
    accuracy again stayed unchanged at `dtrans 2.98e-07`, `dquat 1.68e-08`, `dscale 5.96e-08`, `ddx 1.75e-07`.

## Decision Log

- Decision: optimize the idiomatic kernel with small, measurable hot-path edits instead of rewriting its structure again.
  Rationale: the idiomatic variant already exists to be more readable than the current kernel. Large structural rewrites would make it harder to isolate which changes help or hurt.
  Date/Author: 2026-04-09 / Codex

- Decision: the first optimization target is the per-row `InlineArray[Float32, 14]` allocation inside `accumulate_pair_row()`.
  Rationale: it is repeated four times per point, is absent from the current kernel, and can be removed without abandoning the clearer helper-based organization.
  Date/Author: 2026-04-09 / Codex

- Decision: revert the scratch-row optimization after measurement and keep the pre-existing idiomatic kernel code.
  Rationale: the change preserved correctness but made the real-fixture timing worse, so keeping it would violate the “no regression” constraint for this experiment.
  Date/Author: 2026-04-09 / Codex

- Decision: the next optimization pass should target the hot-loop helper boundary itself while preserving the idiomatic file/module structure outside the kernel body.
  Rationale: the row-buffer allocation alone did not explain the gap, and the nested compile-time closure shape is now the clearest remaining code-generation difference from the current faster kernel.
  Date/Author: 2026-04-09 / Codex

- Decision: revert the helper-boundary rewrite after measurement as well.
  Rationale: it also failed the “improve the idiomatic path on the real fixture” requirement, so it does not belong in the kept code.
  Date/Author: 2026-04-09 / Codex

## Outcomes & Retrospective

This optimization pass produced a negative result, which is still useful. The clearest low-risk hot-path change did not close the gap between the idiomatic kernel and the current Mojo kernel. The repository therefore remains in the best known state: the original idiomatic kernel stays available for readability and comparison, and the current Mojo kernel remains the performance baseline.

The main lesson from the completed passes is that neither of the two obvious local explanations for the gap were sufficient. Removing the per-row buffer allocation did not help, and matching the current kernel’s nested compile-time row-accumulation shape also did not help. The idiomatic kernel should therefore stay as an experimental readability reference unless a future pass is driven by profiler evidence or a broader redesign.

## Context and Orientation

The file `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo` contains the current production Mojo GN ray-step kernel. The file `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels_idiomatic.mojo` contains an experimental rewrite with the same execution model but cleaner structure. Both are exported through `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo` and selected from Python by `packages/mast3r-slam/mast3r_slam/gn_backends.py`.

The real signoff tool is `packages/mast3r-slam/tools/bench_gn_real_fixtures.py`. It reads a captured GN fixture from `packages/mast3r-slam/artifacts/gn-fixtures/...` and reports CUDA, current Mojo, and idiomatic Mojo timings plus the pose/update drift metrics. That tool is the acceptance source for this plan.

“Within striking distance” in this plan means the idiomatic Mojo public timing should move materially closer to the current Mojo timing on the real fixture while preserving the same tiny drift envelope already observed for both Mojo paths. This plan does not redefine the main release gate; it only improves the additive experiment.

## Plan of Work

Edit `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels_idiomatic.mojo` again, but keep the scope narrow. The module-level helpers for shared pose load and relative-pose setup can stay as they are. The hot row accumulation helper should move from a top-level function into a kernel-local compile-time closure so the compiler sees a shape closer to the current fast kernel while the surrounding module still reads more clearly than `gn_kernels.mojo`.

Reuse a thread-local 14-wide scratch array in that closure. Do not change selector semantics, public wrappers, or benchmark interfaces. The point of this plan is still to improve the existing idiomatic variant, not to broaden the experiment.

After the code change, rebuild with the existing Mojo task, rerun the GN tests, and rerun the real-fixture benchmark. Record the exact observed rows for CUDA, current Mojo, and idiomatic Mojo in this file.

## Concrete Steps

Run all commands from the repository root `/var/tmp/vibe-kanban/worktrees/4658-mast3r-slam-mojo/examples-monorepo`.

Build the shared library:

    pixi run -e mast3r-slam-dev _build-mojo-kernels

Run the GN tests:

    pixi run -e mast3r-slam-dev pytest \
      packages/mast3r-slam/tests/test_gn_fixture_utils.py \
      packages/mast3r-slam/tests/test_gn_step_api.py -q

Benchmark the real captured fixture:

    pixi run -e mast3r-slam-dev python \
      packages/mast3r-slam/tools/bench_gn_real_fixtures.py \
      packages/mast3r-slam/artifacts/gn-fixtures/verify-base/rays-000.pt \
      --warmup 5 --runs 20

Observed output from the attempted optimization:

    pixi run -e mast3r-slam-dev _build-mojo-kernels
    ...
    mojo build completed successfully

    pixi run -e mast3r-slam-dev pytest \
      packages/mast3r-slam/tests/test_gn_fixture_utils.py \
      packages/mast3r-slam/tests/test_gn_step_api.py -q
    ...........
    11 passed in 0.91s

    pixi run -e mast3r-slam-dev python \
      packages/mast3r-slam/tools/bench_gn_real_fixtures.py \
      packages/mast3r-slam/artifacts/gn-fixtures/verify-base/rays-000.pt \
      --warmup 5 --runs 20
    fixture                  backend        scope    kind        cuda ms     run ms    ratio     dtrans      dquat     dscale        ddx
    rays-000                 cuda           step     rays          5.127      5.130    1.001        nan        nan        nan   0.00e+00
    rays-000                 cuda           public   rays         54.030     52.130    0.965        nan        nan        nan        nan
    rays-000                 mojo-current   step     rays          5.127      2.111    0.412        nan        nan        nan   1.64e+04
    rays-000                 mojo-current   public   rays         54.030     23.646    0.438   2.98e-07   1.68e-08   5.96e-08   1.75e-07
    rays-000                 mojo-idiomatic step     rays          5.127      2.380    0.464        nan        nan        nan   1.64e+04
    rays-000                 mojo-idiomatic public   rays         54.030     25.489    0.472   2.98e-07   1.68e-08   5.96e-08   1.75e-07

Because the idiomatic row got slower, the first code change was reverted after this measurement.

Observed output from the helper-boundary rewrite attempt:

    pixi run -e mast3r-slam-dev _build-mojo-kernels
    ...
    mojo build completed successfully

    pixi run -e mast3r-slam-dev pytest \
      packages/mast3r-slam/tests/test_gn_fixture_utils.py \
      packages/mast3r-slam/tests/test_gn_step_api.py -q
    ...........
    11 passed in 1.08s

    pixi run -e mast3r-slam-dev python \
      packages/mast3r-slam/tools/bench_gn_real_fixtures.py \
      packages/mast3r-slam/artifacts/gn-fixtures/verify-base/rays-000.pt \
      --warmup 5 --runs 20
    fixture                  backend        scope    kind        cuda ms     run ms    ratio     dtrans      dquat     dscale        ddx
    rays-000                 cuda           step     rays          5.153      5.153    1.000        nan        nan        nan   0.00e+00
    rays-000                 cuda           public   rays         53.102     52.353    0.986        nan        nan        nan        nan
    rays-000                 mojo-current   step     rays          5.153      2.161    0.419        nan        nan        nan   1.64e+04
    rays-000                 mojo-current   public   rays         53.102     24.167    0.455   2.98e-07   1.68e-08   5.96e-08   1.75e-07
    rays-000                 mojo-idiomatic step     rays          5.153      2.448    0.475        nan        nan        nan   1.64e+04
    rays-000                 mojo-idiomatic public   rays         53.102     25.612    0.482   2.98e-07   1.68e-08   5.96e-08   1.75e-07

Because the idiomatic row was still slower, that code change was also reverted.

## Validation and Acceptance

Acceptance for this plan is narrower than the overall GN signoff. The code is acceptable if all of the following remain true:

The shared library builds successfully.

The GN tests pass unchanged.

The idiomatic ray-step still matches the existing tiny drift envelope on the real `rays-000` fixture.

The idiomatic public timing on the same fixture improves relative to the previous recorded `24.239 ms` result and moves closer to the current Mojo timing.

If the timing does not improve or accuracy regresses, revert or refine the optimization rather than broadening scope.

The first attempt failed that timing condition and was reverted. The helper-boundary rewrite also failed it and was reverted.

## Idempotence and Recovery

This plan is safe to rerun. The build step overwrites the Mojo shared library in place, the tests are read-only, and the benchmark reads a captured fixture without mutating it. If an optimization hurts performance or accuracy, restore only the idiomatic kernel file and rerun the same validation commands.

## Artifacts and Notes

The key artifact is the before/after benchmark delta for the idiomatic row on `rays-000`.

Before this pass:

    mojo-current public:   22.327 ms
    mojo-idiomatic public: 24.239 ms

During the attempted optimization:

    mojo-current public:   23.646 ms
    mojo-idiomatic public: 25.489 ms

That is enough evidence to say both attempted optimization directions were not the right ones.

## Interfaces and Dependencies

No public Python interface changes are expected from this plan. The existing idiomatic export path must remain available through:

    gauss_newton_rays_step_idiomatic(...)

inside the Mojo shared library and through the explicit selector in `packages/mast3r-slam/mast3r_slam/gn_backends.py`.

Revision note: created to optimize the already-working experimental idiomatic GN ray-step kernel without changing the release-default current Mojo path.

Revision note: updated after the first targeted optimization attempt. The reusable row-scratch experiment preserved correctness but made the idiomatic kernel slower on the real fixture, so the kernel edit was reverted and this file now records that negative result.

Revision note: updated again after re-inspecting the kernels. The next optimization target is the hot-loop helper boundary, because the first local scratch-buffer change did not improve throughput.

Revision note: updated after the second optimization attempt. Matching the current kernel’s nested compile-time helper shape also failed to improve the idiomatic kernel, so that code was reverted and the plan now records both negative results.
