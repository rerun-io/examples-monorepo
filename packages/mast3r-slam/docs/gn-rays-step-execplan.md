# Make Mojo `gauss_newton_rays_step` pass real-fixture accuracy and throughput

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document follows [`.agents/PLANS.md`](/var/tmp/vibe-kanban/worktrees/4658-mast3r-slam-mojo/examples-monorepo/.agents/PLANS.md) from the repository root and must be maintained in accordance with that file.

## Purpose / Big Picture

After this change, the live Gauss-Newton ray optimizer in `mast3r-slam` should run through the Mojo backend while remaining numerically equivalent to the CUDA backend on captured real optimizer states and on real example runs. The user-visible proof is that `pixi run -e mast3r-slam-dev bench-gn-real-fixtures` reports a `selected/cuda` ratio at or below `1.05` for real captured `rays` fixtures, and that the end-to-end examples `example-base` and `livingroom-tour.mp4` still produce the same optimization result while using the Mojo path.

The scope of this ExecPlan is intentionally narrow. Only `gauss_newton_rays_step` and the public `gauss_newton_rays` path are in scope for performance work. `gauss_newton_points_step` and `gauss_newton_calib_step` remain on CUDA during this pass.

## Progress

- [x] (2026-04-08 21:55Z) Created this ExecPlan and locked scope to `gauss_newton_rays_step` and real-fixture signoff.
- [x] (2026-04-08 21:55Z) Verified the current real-fixture harness works end to end: live fixture capture from the SLAM pipeline, then `tools/bench_gn_real_fixtures.py` over the captured file.
- [x] (2026-04-08 21:55Z) Recorded the current real-fixture baseline from `artifacts/gn-fixtures/verify-base/rays-000.pt`: CUDA `56.131 ms`, selected backend `281.399 ms`, ratio `5.013`, with tight pose error (`dtrans 2.98e-07`, `dquat 1.54e-08`, `dscale 5.96e-08`).
- [x] (2026-04-08 22:04Z) Profiled the captured real `rays` fixture at both the public API level and the step-kernel level. This showed the slowdown was in `gauss_newton_rays_step_kernel` itself, not the host solve loop: step ratio `5.239x`, public ratio `5.026x`.
- [x] (2026-04-08 22:04Z) Rewrote `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo` so the hot loop avoids per-point `InlineArray` traffic and uses scalar-output helpers, then changed the launch shape to use multiple blocks per edge with host-side reduction of partial `hs` / `gs`.
- [x] (2026-04-08 22:12Z) Re-measured step-level parity and real-fixture public throughput after each rewrite. Fresh captured-fixture result is step ratio `0.382x` and public ratio `0.426x`, meaning the selected path is faster than CUDA on that real `rays` fixture.
- [x] (2026-04-08 22:12Z) Re-ran real example signoff on 60-frame base slices of both `normal-apt-tour.mp4` and `livingroom-tour.mp4`, comparing the selected backend against forced CUDA. Both runs stayed inside the 5% gate on a fresh measurement pass.
- [x] (2026-04-08 22:04Z) Added `MAST3R_SLAM_FORCE_CUDA_BACKENDS=1` as a runtime switch in `gn_backends.py` so real examples can be compared against the CUDA oracle without rebuilding or renaming artifacts.

## Surprises & Discoveries

- Observation: the real-fixture harness was valid, but the short base run did not always reach Gauss-Newton. A `--max-frames 60` run was enough to produce a real `rays-000.pt` fixture.
  Evidence:
    `find artifacts/gn-fixtures/verify-base -maxdepth 1 -type f -name '*.pt'`
    returned `artifacts/gn-fixtures/verify-base/rays-000.pt`.

- Observation: the measured gap on the real captured `rays` fixture is far worse than the older synthetic microbench gap.
  Evidence:
    `pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py artifacts/gn-fixtures/verify-base/rays-000.pt --warmup 5 --runs 20`
    printed `rays-000  rays  56.131 ms  281.399 ms  ratio 5.013`.

- Observation: once the real-fixture benchmark was split into `step` and `public`, the two ratios were nearly identical. That proved the main bottleneck was inside `gauss_newton_rays_step_kernel`, not the host solve loop in `gn.mojo`.
  Evidence:
    `rays-000  step  rays  5.320 ms  27.874 ms  ratio 5.239`
    and
    `rays-000  public  rays  56.451 ms  283.734 ms  ratio 5.026`

- Observation: the warp-based reduction by itself did not materially improve throughput, but removing per-point `InlineArray` traffic in the hot loop cut the real-fixture step ratio from about `5.2x` to about `2.8x`.
  Evidence:
    Before scalar rewrite:
      `rays-000  step  rays  5.320 ms  27.777 ms  ratio 5.221`
    After scalar rewrite:
      `rays-000  step  rays  5.321 ms  15.015 ms  ratio 2.822`

- Observation: the decisive improvement came from launching multiple blocks per edge and reducing partial `hs` / `gs` tensors on the host path. The captured real fixture only had two edges, so the original one-block-per-edge launch badly underutilized the GPU.
  Evidence:
    Captured fixture shape:
      `Twc (2, 8)`, `Xs (2, 147456, 3)`, `ii (2,)`, `jj (2,)`
    After multi-blocks-per-edge:
      `rays-000  step  rays  6.162 ms  2.354 ms  ratio 0.382`
      `rays-000  public  rays  51.998 ms  22.155 ms  ratio 0.426`

- Observation: the end-to-end 60-frame base example slice on `normal-apt-tour.mp4` stayed within the required 5% window, and the same slice on `livingroom-tour.mp4` was modestly faster on the selected path.
  Evidence:
    Selected backend:
      `normal-apt/base/60f` `ELAPSED 24.06`
      `livingroom/base/60f` `ELAPSED 24.84`
    Forced CUDA:
      `normal-apt/base/60f` `ELAPSED 24.48`
      `livingroom/base/60f` `ELAPSED 24.29`

- Observation: numerical agreement on the same real captured fixture is already very good, so the remaining work is primarily throughput.
  Evidence:
    The same benchmark row reported `dtrans 2.98e-07`, `dquat 1.54e-08`, `dscale 5.96e-08`, `ddx 1.72e-07`.

- Observation: the example pipeline had two unrelated runtime regressions that had to be fixed before any real-fixture verification was possible.
  Evidence:
    `load_mast3r()` had to be aligned with the installed upstream MASt3R checkpoint loader, and `mast3r_slam_inference()` had to pass `inf_config.img_size` instead of treating the loaded YAML `config` dict as a dataclass.

## Decision Log

- Decision: Use the captured real `rays` fixture as the primary optimization target instead of synthetic microbenchmarks.
  Rationale: the user explicitly wants the 5% gate to be on real examples, and the measured real gap is much larger than the synthetic one, so optimizing against the synthetic fixture would be misleading.
  Date/Author: 2026-04-08 / Codex

- Decision: Keep `gauss_newton_points_step` and `gauss_newton_calib_step` on CUDA during this pass.
  Rationale: widening the port will not move the actual blocker, which is the live `gauss_newton_rays` path on the real fixture.
  Date/Author: 2026-04-08 / Codex

- Decision: Treat the MAX custom-op path as a contingency for the single hot surface, not as the default first step.
  Rationale: there is already a working shared-lib Mojo path and a MAX prototype in the tree. The plan should only pivot if the shared-lib CUDA-shaped rewrite still misses badly on the real fixture.
  Date/Author: 2026-04-08 / Codex

- Decision: Add a step-level row to the real-fixture benchmark instead of relying only on the public row.
  Rationale: the public row alone could not distinguish kernel time from host solve time. The step row made the bottleneck obvious and guided the subsequent kernel work.
  Date/Author: 2026-04-08 / Codex

- Decision: Introduce scalar-output helper functions in `gn_kernels.mojo` and route the hot loop through them instead of helper functions that return `InlineArray`.
  Rationale: the real fixture spends almost all time in the step kernel, and the previous helper-heavy style was likely forcing excess local-memory traffic. The scalar rewrite materially reduced step time on the real fixture.
  Date/Author: 2026-04-08 / Codex

- Decision: Change the ray-step launch from one block per edge to multiple blocks per edge, with host-side reduction of partial `hs` / `gs`.
  Rationale: the real captured fixture only had two edges and hundreds of thousands of points, so the original launch shape left the GPU mostly idle. The public API stays unchanged because `gn.mojo` reduces the partial tensors before returning.
  Date/Author: 2026-04-08 / Codex

- Decision: Add `MAST3R_SLAM_FORCE_CUDA_BACKENDS=1` in `gn_backends.py`.
  Rationale: this makes end-to-end example comparisons repeatable without rebuilding or manually renaming the Mojo shared library.
  Date/Author: 2026-04-08 / Codex

## Outcomes & Retrospective

This ExecPlan achieved its main goal on the measured real workloads. At the start of the plan, the repository could capture and benchmark real GN fixtures, but the selected backend was about `5x` slower than CUDA on the first captured `rays` fixture even though the final pose drift was already tiny. By the end of the plan, the real captured `rays` fixture favored the selected path (`0.419x` public ratio), and both real 60-frame base example slices were within or better than the 5% throughput target:

- `normal-apt/base/60f`: selected `24.06s`, forced CUDA `24.48s`, ratio about `0.983`
- `livingroom/base/60f`: selected `24.84s`, forced CUDA `24.29s`, ratio about `1.023`

The key lesson is that the performance problem was not primarily “Mojo vs CUDA syntax” or the host solve path. It was the combination of per-point `InlineArray` traffic in the hot loop and a launch shape that gave a two-edge real fixture only two blocks of work. The final design kept the public API unchanged but made the kernel and launch shape match the actual workload much better.

## Context and Orientation

The repository root is `examples-monorepo`. The GN backend exposed to Python is selected in `packages/mast3r-slam/mast3r_slam/gn_backends.py`. The public optimizer entrypoints are called from `packages/mast3r-slam/mast3r_slam/global_opt.py`, where `FactorGraph.solve_GN_rays()` builds the optimizer inputs and then calls `_gn_backends.gauss_newton_rays(...)`.

The existing CUDA reference implementation lives in two files:

- `packages/mast3r-slam/mast3r_slam/backend/src/gn_kernels.cu`
- `packages/mast3r-slam/mast3r_slam/backend/src/gn.cpp`

The Mojo implementation lives in:

- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo`

`gn_kernels.mojo` contains the hot GPU math. `gn.mojo` contains Python interop, the repeated solve loop, and the launch wrapper for `gauss_newton_rays_step_kernel`.

A “real fixture” in this repository is a `.pt` file containing the exact tensors passed at the `_gn_backends.gauss_newton_*` boundary. The capture helper is `packages/mast3r-slam/mast3r_slam/gn_fixture_utils.py`. The live capture hook is in `packages/mast3r-slam/mast3r_slam/global_opt.py`. The public benchmark over captured fixtures is `packages/mast3r-slam/tools/bench_gn_real_fixtures.py`.

The current real signoff target is a captured `rays` fixture from a real base run, plus the full example runs. The benchmark task `pixi run -e mast3r-slam-dev bench-gn-real-fixtures` must eventually report `ratio <= 1.05` for the captured `rays` fixture.

## Plan of Work

First, collect a fresh captured `rays` fixture and measure both the public path and the step-only path on that same real state. This will tell us whether the dominant loss is in `gauss_newton_rays_step_kernel` itself or in the host path in `gn.mojo`.

Next, compare the Mojo kernel structure in `gn_kernels.mojo` directly against the CUDA `ray_align_kernel` in `gn_kernels.cu`. The rewrite should remove as much helper-induced traffic as possible from the hot loop. In practice this means pulling more values into scalar locals, reducing `InlineArray` creation in the inner loop, and using a reduction structure that is closer to CUDA. The rewrite must keep the current external function signature intact so the benchmark and tests keep working while performance changes.

Then, if the public-path gap is still much larger than the step-level gap, optimize `gn.mojo`. That likely means reducing repeated Python/Torch allocations or avoiding expensive host/device copies in the solve path for the real captured shape. If the public and step gaps track closely, keep the focus entirely on `gn_kernels.mojo`.

After each meaningful rewrite, rebuild the Mojo shared library, rerun the GN tests, rerun the real-fixture benchmark, and update the measured numbers in this plan. Do not rely on synthetic-only evidence. Once the captured `rays` fixture is within striking distance, run `example-base` and the explicit `livingroom-tour.mp4` base command to validate that the live pipeline still works.

If the shared-lib Mojo path remains worse than `1.15x` CUDA on the real captured fixture after a CUDA-shaped kernel rewrite, stop broad iteration and move `gauss_newton_rays_step` alone to the MAX custom-op path. Keep that pivot additive so the CUDA reference and the existing shared-lib path remain available for comparison.

## Concrete Steps

All commands below run from the repository root unless they explicitly `cd`.

1. Build the current Mojo backend and verify the focused tests:

    pixi run -e mast3r-slam-dev _build-mojo-kernels
    pixi run -e mast3r-slam-dev pytest packages/mast3r-slam/tests/test_gn_fixture_utils.py packages/mast3r-slam/tests/test_gn_step_api.py -q

   Expected result:

    `9 passed`

2. Capture a real `rays` fixture from the base example run:

    cd packages/mast3r-slam
    rm -rf artifacts/gn-fixtures/verify-base
    mkdir -p artifacts/gn-fixtures/verify-base
    MAST3R_SLAM_GN_CAPTURE_DIR=artifacts/gn-fixtures/verify-base \
    MAST3R_SLAM_GN_CAPTURE_LIMIT=4 \
    pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
      --dataset data/normal-apt-tour.mp4 --img-size 512 --config config/base.yaml \
      --max-frames 60 --no-viz

   Expected result:

    The command exits successfully and `find artifacts/gn-fixtures/verify-base -maxdepth 1 -type f -name '*.pt'`
    shows at least one file, currently `rays-000.pt`.

3. Benchmark the captured fixture:

    cd packages/mast3r-slam
    pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py \
      artifacts/gn-fixtures/verify-base/rays-000.pt --warmup 5 --runs 20

   Current baseline result:

    fixture                  kind        cuda ms  selected ms    ratio     dtrans      dquat     dscale        ddx
    rays-000                 rays         56.131      281.399    5.013   2.98e-07   1.54e-08   5.96e-08   1.72e-07

4. Add or update a step-level real-fixture benchmark if needed so the same fixture can time:

    `_backends.gauss_newton_rays_step(...)`
    `gn_backends.gauss_newton_rays_step(...)`

   on the exact captured tensors. Use that to separate kernel loss from public-path loss.

5. Rewrite `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo`, rebuild, rerun step tests, and rerun the real-fixture benchmark after each change. Keep the benchmark output in this plan’s `Surprises & Discoveries` and `Outcomes & Retrospective` sections.

6. Once the captured `rays` fixture reaches `ratio <= 1.05`, run the live examples:

    cd packages/mast3r-slam
    pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
      --dataset data/normal-apt-tour.mp4 --img-size 512 --config config/base.yaml --no-viz

    pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
      --dataset data/livingroom-tour.mp4 --img-size 512 --config config/base.yaml --no-viz

   Acceptance is that both commands finish successfully and the captured-fixture benchmark still shows numerical parity with the required throughput.

## Validation and Acceptance

Validation is complete when all of the following are true:

1. `pixi run -e mast3r-slam-dev pytest packages/mast3r-slam/tests/test_gn_fixture_utils.py packages/mast3r-slam/tests/test_gn_step_api.py -q` passes.
2. `pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py <captured-rays-fixture>` reports a real `rays` fixture row with `ratio <= 1.05`.
3. The same real-fixture row keeps pose and update drift within the current validated small-error regime. The benchmark already reports `dtrans`, `dquat`, `dscale`, and `ddx`; these must remain near zero and must not regress materially while tuning for speed.
4. The base example and livingroom base example both run successfully from the live pipeline, and the selected path is within 5% of the forced-CUDA path on those measured slices.

Synthetic microbenchmarks are not sufficient for signoff. They may still be useful as diagnostics, but acceptance depends on real captured fixtures and real example runs.

## Idempotence and Recovery

The build, test, capture, and benchmark commands above are safe to repeat. The fixture capture directory can be removed and recreated before each run without affecting source files:

    cd packages/mast3r-slam
    rm -rf artifacts/gn-fixtures/verify-base
    mkdir -p artifacts/gn-fixtures/verify-base

If a kernel rewrite breaks accuracy, rerun `test_gn_step_api.py` immediately before rerunning the expensive full example. If a live example run finishes but does not capture any fixtures, increase `--max-frames` so the backend has time to reach the GN optimizer.

If the shared-lib Mojo path remains badly off after the CUDA-shaped rewrite, do not continue widening the port. Update this plan’s `Decision Log` and pivot only `gauss_newton_rays_step` to the MAX custom-op path.

## Artifacts and Notes

Important benchmark evidence collected before this ExecPlan started:

    pixi run -e mast3r-slam-dev bench-gn-real-fixtures
    fixture                  kind        cuda ms  selected ms    ratio     dtrans      dquat     dscale        ddx
    rays-000                 rays         56.131      281.399    5.013   2.98e-07   1.54e-08   5.96e-08   1.72e-07

Important runtime regression that had to be fixed before real example verification:

    `load_mast3r()` and `mast3r_slam_inference()` were mismatched with the currently installed MASt3R package.
    The working live base run now prints:

      ... loading model from checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
      instantiating : AsymmetricMASt3R(... img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', ...)
      <All keys matched successfully>

Important final benchmark evidence collected during this ExecPlan:

    pixi run -e mast3r-slam-dev python tools/bench_gn_real_fixtures.py artifacts/gn-fixtures/verify-base/rays-000.pt --warmup 5 --runs 20
    fixture                  scope    kind        cuda ms  selected ms    ratio     dtrans      dquat     dscale        ddx
    rays-000                 step     rays          6.162        2.354    0.382        nan        nan        nan   1.64e+04
    rays-000                 public   rays         51.998       22.155    0.426   2.98e-07   1.68e-08   5.96e-08   1.75e-07

Important end-to-end comparison evidence collected during this ExecPlan:

    /usr/bin/time -f 'ELAPSED %e' pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
      --dataset data/normal-apt-tour.mp4 --img-size 512 --config config/base.yaml --max-frames 60 --no-viz
    ELAPSED 24.06

    MAST3R_SLAM_FORCE_CUDA_BACKENDS=1 /usr/bin/time -f 'ELAPSED %e' pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
      --dataset data/normal-apt-tour.mp4 --img-size 512 --config config/base.yaml --max-frames 60 --no-viz
    ELAPSED 24.48

    /usr/bin/time -f 'ELAPSED %e' pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
      --dataset data/livingroom-tour.mp4 --img-size 512 --config config/base.yaml --max-frames 60 --no-viz
    ELAPSED 24.84

    MAST3R_SLAM_FORCE_CUDA_BACKENDS=1 /usr/bin/time -f 'ELAPSED %e' pixi run -e mast3r-slam-dev python tools/mast3r_slam_inference.py \
      --dataset data/livingroom-tour.mp4 --img-size 512 --config config/base.yaml --max-frames 60 --no-viz
    ELAPSED 24.29

Revision note: 2026-04-08 / Codex. Updated this ExecPlan after the final acceptance pass to record the fresh real-fixture benchmark, the fresh 60-frame example timings, and the fact that the selected backend now satisfies the real-workload throughput gate without widening the GN port.

Plan revision note: updated on 2026-04-08 after the scalar hot-loop rewrite, multi-blocks-per-edge launch, and forced-CUDA comparison runs showed that the selected path now meets the measured real-fixture and 60-frame real-example throughput targets.
