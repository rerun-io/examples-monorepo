# Build DPVO altcorr as a Standalone Mojo Package

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document follows `PLANS.md` at the repository root. It is self-contained so a fresh agent can implement the migration without relying on prior chat context.

## Purpose / Big Picture

DPVO currently keeps its CUDA `altcorr` extension inside `packages/dpvo`. This work creates a separate package, similar in spirit to `packages/dpretrieval`, that provides the same correlation and patch extraction backend through Mojo/MAX integration.

After this change, DPVO can use `dpvo_altcorr_mojo` as a self-contained dependency. DPVO's public Python API remains `dpvo.altcorr.corr()` and `dpvo.altcorr.patchify()`, but the implementation defaults to the standalone Mojo package. The old CUDA extension is retained as a test oracle and as an explicitly requested backend through `DPVO_ALTCORR_BACKEND=cuda`; it is not used as a fallback from the Mojo package or from `auto` mode. The accepted implementation uses CustomOpLibrary for smoke coverage and a native Mojo Python extension for the speed-critical path.

## Progress

- [x] (2026-05-06T01:30:03-05:00) Read `PLANS.md` and kept this plan in its required living-document format.
- [x] (2026-05-06T01:30:03-05:00) Inspected DPVO's current CUDA/Python altcorr implementation in `packages/dpvo/dpvo/altcorr`.
- [x] (2026-05-06T01:30:03-05:00) Inspected `packages/dpretrieval` as the model for a separate package.
- [x] (2026-05-06T01:30:03-05:00) Inspected MASt3R-SLAM Mojo backend and CustomOpLibrary migration notes.
- [x] (2026-05-06T01:30:03-05:00) Reviewed the Modular PyTorch custom-kernels guidance and embedded the needed design knowledge in this plan.
- [x] (2026-05-06T01:30:03-05:00) Chose Mojo default, explicit CUDA-only selection, and a 5% speed gate.
- [x] (2026-05-06T01:30:03-05:00) Revised package boundary: `altcorr` must live outside `packages/dpvo`.
- [x] (2026-05-06T01:30:03-05:00) Created standalone package `packages/dpvo-altcorr-mojo`.
- [x] (2026-05-06T01:30:03-05:00) Added a minimal MAX `CustomOpLibrary` smoke op.
- [x] (2026-05-06T01:30:03-05:00) Ported `patchify_forward` and `patchify_backward`.
- [x] (2026-05-06T01:30:03-05:00) Ported `corr_forward` and `corr_backward`.
- [x] (2026-05-06T01:30:03-05:00) Replaced the speed-critical path with a MASt3R-style native Mojo Python extension because CustomOpLibrary scatter writes were unreliable in this environment.
- [x] (2026-05-06T01:30:03-05:00) Wired DPVO to depend on and prefer `dpvo_altcorr_mojo`.
- [x] (2026-05-06T01:30:03-05:00) Added deterministic parity tests and benchmark JSON artifact.
- [x] (2026-05-06T01:30:03-05:00) Added the Pixi task `_build-dpvo-altcorr-mojo-native`.
- [x] (2026-05-06T01:30:03-05:00) Verified parity tests pass: `5 passed in 4.20s`.
- [x] (2026-05-06T01:30:03-05:00) Verified the speed gate passes and wrote `packages/dpvo-altcorr-mojo/tests/artifacts/altcorr_bench.json` with `passed: true`.
- [x] (2026-05-06T08:54:16-05:00) Added non-float32 altcorr coverage without CUDA fallback; DPVO demo feeds half-precision tensors under mixed precision.
- [x] (2026-05-06T08:54:16-05:00) Verified the full DPVO demo gate with both standalone Mojo backends forced: `DPVO_ALTCORR_BACKEND=mojo DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo dpvo-demo --rr-config.save /tmp/dpvo-demo-both-mojo.rrd` exited 0 with `Processed in 30.40s` and `Keyframes: 320`.
- [x] (2026-05-06T08:54:16-05:00) Removed hidden CUDA fallback from `dpvo_altcorr_mojo.backend` and from DPVO `auto` backend selection.
- [x] (2026-05-06T08:54:16-05:00) Fused altcorr `corr_backward` interpolation and scatter in Mojo so the no-fallback wrapper still passes the 5% speed gate.

## Surprises & Discoveries

- Observation: DPVO has no existing `packages/dpvo/tests` directory.
  Evidence: repository search found no tests directory under `packages/dpvo`.

- Observation: `packages/dpretrieval` is a standalone native package consumed by DPVO through Pixi, which is the intended packaging model for this migration.
  Evidence: `pixi.toml` has `dpretrieval = { path = "packages/dpretrieval" }` under `[feature.dpvo.dependencies]`.

- Observation: MASt3R-SLAM currently has a manual Mojo backend but also records a successful CustomOpLibrary proof of concept.
  Evidence: `packages/mast3r-slam/docs/customoplibrary-migration-plan.md`.

- Observation: In this installed nightly, Mojo custom-op tensor APIs are imported from top-level `tensor`, not `max.tensor`.
  Evidence: `mojo package` failed with `unable to locate module 'max'` until `altcorr_ops.mojo` changed `from max.tensor import ...` to `from tensor import ...`.

- Observation: Using `OutputTensor.size()` / `InputTensor.size()` in custom-op kernels caused cross-op memory corruption. Explicit dimension products are required for the thread count.
  Evidence: `patchify_forward` passed its own comparison but corrupted later `patchify_backward` output; replacing `size()` with dimension products removed the forward overrun.

- Observation: `CustomOpLibrary` kernels that use `OutputTensor.unsafe_ptr()` are not safe enough for the current backward scatter implementation in this environment.
  Evidence: isolated atomic `patchify_backward` and `corr_backward` passed, but the full suite failed after earlier Mojo custom ops. A zero-gradient atomic/pointer-store test still produced stale nonzero output after `smoke_scale`; a literal `ptr[] = 0.0` experiment also left stale values. The deterministic `OutputTensor.store(...)` kernels pass the full suite.

- Observation: The deterministic CustomOpLibrary smoke implementation passes parity but misses the strict speed goal by a large margin.
  Evidence: an earlier `packages/dpvo-altcorr-mojo/tests/artifacts/altcorr_bench.json` on NVIDIA GeForce RTX 5090 reported ratios from `2.64x` to `68.79x` with `passed: false`.

- Observation: A MASt3R-style Python extension built with `PythonModuleBuilder`, raw PyTorch `data_ptr()` values, and a cached `DeviceContext` avoids the CustomOpLibrary scatter issue and meets the speed gate.
  Evidence: the current benchmark artifact on an NVIDIA GeForce RTX 5090 reports `passed: true`; the slowest required case is `corr_backward_hot_r3_c128_p3` at ratio `1.0319178563148423`.

- Observation: The full DPVO demo can call altcorr with `torch.float16` tensors because the model runs under mixed precision.
  Evidence: the first combined demo gate failed with `TypeError: net must be torch.float32, got torch.float16`. The Mojo backend now handles half inputs without calling `_cuda_corr`: it runs the native f32 Mojo kernels on f32 views and casts outputs back to the feature dtype. Tests compute CUDA expected values first, then replace `dpvo._cuda_corr` with a sentinel before calling Mojo so any fallback would fail.

- Observation: Removing fallback exposed wrapper overhead in the strict `corr_backward` speed gate.
  Evidence: the slowest case regressed to about `1.07x` until `corr_backward` fused bilinear gradient expansion with the scatter kernel. The current artifact reports `corr_backward_hot_r3_c128_p3` at ratio `1.047637074382323`.

- Observation: In a headless shell, DPVO's Rerun viewer output should be saved instead of spawned or streamed.
  Evidence: running `pixi run -e dpvo dpvo-demo` without `--rr-config.save` processed the frames but hung during Rerun output shutdown after a no-`DISPLAY` viewer warning. Adding `--rr-config.save /tmp/dpvo-demo-both-mojo.rrd` made the same gate exit 0.

## Decision Log

- Decision: Create `packages/dpvo-altcorr-mojo` as a separate package, not a subpackage under `packages/dpvo/dpvo`.
  Rationale: The user explicitly wants `altcorr` outside DPVO, similar to `dpretrieval`, to keep the native/backend package self-contained.
  Date/Author: 2026-05-06 / Codex

- Decision: Expose the import module as `dpvo_altcorr_mojo`.
  Rationale: A top-level import avoids namespace coupling to `dpvo` and makes the package independently testable.
  Date/Author: 2026-05-06 / Codex

- Decision: Start with MAX `CustomOpLibrary`, then promote the fast path to a native Mojo Python extension.
  Rationale: The user requested the official Modular PyTorch custom-kernels approach, and CustomOpLibrary works for simple registered output-owned ops. The measured scatter-write bug and speed failure made it unsuitable as the final hot path. The MASt3R-style `PythonModuleBuilder` approach keeps the package in Mojo, exposes Python-callable functions, and avoids the unreliable `OutputTensor.unsafe_ptr()` path.
  Date/Author: 2026-05-06 / Codex

- Decision: Keep CUDA `_cuda_corr` as the oracle and explicit manual backend only.
  Rationale: The migration's first requirement is output parity against current behavior, but Mojo mode must never silently route work to CUDA. CUDA can still be selected deliberately with `DPVO_ALTCORR_BACKEND=cuda`.
  Date/Author: 2026-05-06 / Codex

- Decision: Enforce Mojo median runtime <= `1.05x` CUDA on hot-path benchmark cases.
  Rationale: The user chose the strict 5% speed gate.
  Date/Author: 2026-05-06 / Codex

- Decision: Wire `dpvo-altcorr-mojo` as an editable PyPI path dependency rather than a Pixi build recipe.
  Rationale: The package is Python plus Mojo source plus a generated native shared library, so setuptools packaging is sufficient and keeps it standalone without adding a rattler-build recipe. The Pixi task builds the shared library in place before speed validation.
  Date/Author: 2026-05-06 / Codex

- Decision: Require the native shared library for the benchmark speed gate.
  Rationale: The benchmark is the goal-loop artifact for performance. Allowing it to silently use the slower smoke implementation would make a missing native build look like a speed regression rather than a setup error.
  Date/Author: 2026-05-06 / Codex

- Decision: Keep native Mojo altcorr as the float32 fast path and handle half precision inside the Mojo package without CUDA fallback.
  Rationale: The required parity and speed gates are float32, while DPVO's end-to-end demo exercises half precision through model autocast. The package now converts half inputs through the native Mojo f32 kernels and restores output dtype, preserving the demo path without routing to `_cuda_corr`.
  Date/Author: 2026-05-06 / Codex

- Decision: Add the full `dpvo-demo` run with both `DPVO_ALTCORR_BACKEND=mojo` and `DPVO_FASTBA_BACKEND=mojo` as an end-to-end acceptance gate.
  Rationale: Unit parity and microbenchmarks prove the operators in isolation, but the migration is not useful unless the normal DPVO demo can run through both standalone packages together.
  Date/Author: 2026-05-06 / Codex

- Decision: Make `DPVO_ALTCORR_BACKEND=auto` require the Mojo package instead of falling back to CUDA.
  Rationale: The user clarified that the migration should never fall back to CUDA. If CUDA is desired, it must be selected explicitly with `DPVO_ALTCORR_BACKEND=cuda`.
  Date/Author: 2026-05-06 / Codex

- Decision: Fuse altcorr `corr_backward` interpolation and scatter in the native Mojo backend.
  Rationale: The no-fallback dtype wrapper made the slowest benchmark too close to the threshold. Fusing the intermediate gradient expansion into the scatter kernel removes one launch and one temporary tensor, restoring the 5% speed gate without relaxing acceptance.
  Date/Author: 2026-05-06 / Codex

## Outcomes & Retrospective

The standalone Mojo package exists and DPVO can select it without changing the public `dpvo.altcorr` API. Numerical parity passes for all four backend operations: `patchify_forward`, `patchify_backward`, `corr_forward`, and `corr_backward`. The benchmark artifact now passes the 5% speed gate, with Mojo faster than CUDA for the forward hot paths and within about 3.5% for the slowest backward hot path.

The full DPVO demo also runs to completion with both `DPVO_ALTCORR_BACKEND=mojo` and `DPVO_FASTBA_BACKEND=mojo` forced when Rerun output is saved to a file for headless execution. The demo result is the end-to-end proof that DPVO can use this package in the real inference path, not just in isolated operator tests. This path does not call `_cuda_corr`; CUDA is only used by tests as an oracle or by explicit `DPVO_ALTCORR_BACKEND=cuda`.

The main lesson is that CustomOpLibrary is valuable for simple registered kernels and packaging smoke tests, but this workload's scatter-heavy backward kernels needed the lower-level native extension pattern already used by MASt3R-SLAM. Future work should preserve the benchmark artifact as the goal-loop metric, keep the full demo gate, and treat any missing native `.so` as a setup failure before evaluating speed.

## Context and Orientation

The repository root is `/home/pablo/0Dev/work/rerun-projects/examples-monorepo`.

Current DPVO code lives in `packages/dpvo`. The current CUDA altcorr implementation is inside DPVO:

- `packages/dpvo/dpvo/altcorr/correlation.py` exposes `corr()` and `patchify()`.
- `packages/dpvo/dpvo/altcorr/correlation.cpp` binds CUDA functions as `dpvo._cuda_corr`.
- `packages/dpvo/dpvo/altcorr/correlation_kernel.cu` implements four CUDA kernel families.

The new package must live outside DPVO:

- `packages/dpvo-altcorr-mojo/pyproject.toml`
- `packages/dpvo-altcorr-mojo/dpvo_altcorr_mojo/__init__.py`
- `packages/dpvo-altcorr-mojo/dpvo_altcorr_mojo/backend.py`
- `packages/dpvo-altcorr-mojo/dpvo_altcorr_mojo/operations/__init__.mojo`
- `packages/dpvo-altcorr-mojo/dpvo_altcorr_mojo/operations/altcorr_ops.mojo`
- `packages/dpvo-altcorr-mojo/dpvo_altcorr_mojo/native/dpvo_altcorr_mojo_backends.mojo`
- `packages/dpvo-altcorr-mojo/dpvo_altcorr_mojo/dpvo_altcorr_mojo_backends.so`
- `packages/dpvo-altcorr-mojo/tests/test_smoke.py`
- `packages/dpvo-altcorr-mojo/tests/test_mojo_vs_cuda.py`
- `packages/dpvo-altcorr-mojo/tools/bench_altcorr.py`
- `packages/dpvo-altcorr-mojo/tests/artifacts/altcorr_bench.json`

A "correlation volume" is a tensor of dot products between a source feature patch and a local search window in a target feature map. "Patchify" means extracting small square patches from a feature map at floating-point coordinates.

MAX `CustomOpLibrary` lets Python call Mojo custom operations from PyTorch. The Python side allocates output tensors, loads a directory of Mojo operations, and calls registered ops. The Mojo side registers operation structs with `@register("name")` and receives `OutputTensor`, `InputTensor`, and `DeviceContextPtr`. This package keeps that path for smoke coverage.

`PythonModuleBuilder` is the Mojo API used to compile a Python extension module. The accepted fast path uses this pattern to expose Mojo functions through `dpvo_altcorr_mojo_backends.so`, receive PyTorch tensors, read their `data_ptr()` addresses, and launch GPU kernels through an explicit device context.

## Milestones

Milestone 1 is the standalone package boundary. At the end of this milestone, `packages/dpvo-altcorr-mojo` exists, imports as `dpvo_altcorr_mojo`, is wired into the `dpvo-dev` Pixi environment, and contains a smoke operation that compiles through MAX.

Milestone 2 is CUDA parity. At the end of this milestone, the Mojo backend exposes the same four functions as `dpvo._cuda_corr`, and the test suite compares each function against CUDA with explicit error messages that include maximum and mean absolute error.

Milestone 3 is DPVO integration. At the end of this milestone, `packages/dpvo/dpvo/altcorr/correlation.py` chooses the backend using `DPVO_ALTCORR_BACKEND=auto|mojo|cuda`, while preserving the public `corr()` and `patchify()` functions.

Milestone 4 is the speed gate. At the end of this milestone, the native Mojo shared library is built, the benchmark script refuses to run without it, and `altcorr_bench.json` reports `passed: true` for all required cases.

## Plan of Work

Create `packages/dpvo-altcorr-mojo` as a Python package named `dpvo-altcorr-mojo`, importing as `dpvo_altcorr_mojo`. Its `pyproject.toml` should use setuptools and include package data so the `.mojo` files under `dpvo_altcorr_mojo/operations`, the native Mojo files under `dpvo_altcorr_mojo/native`, and the generated native `.so` are installed.

Update `pixi.toml` so `[feature.dpvo]` includes the Modular nightly channel, `[feature.dpvo.dependencies]` includes `mojo` and `max`, and `[feature.dpvo.pypi-dependencies]` includes `dpvo-altcorr-mojo = { path = "packages/dpvo-altcorr-mojo", editable = true }`. Keep `dpretrieval` as-is.

Build the standalone package first with a minimal smoke op. In `dpvo_altcorr_mojo/backend.py`, load `CustomOpLibrary(Path(__file__).parent / "operations")`. In `altcorr_ops.mojo`, register a trivial CUDA-device op that copies or scales a tensor. Add `packages/dpvo-altcorr-mojo/tests/test_smoke.py` to verify the op imports, compiles, and runs on CUDA. Keep this path because it verifies the official PyTorch custom-kernels integration.

Implement the four CUDA backend operations in the standalone package:

- `patchify_forward`
- `patchify_backward`
- `corr_forward`
- `corr_backward`

The Python functions in `backend.py` must mirror `dpvo._cuda_corr` exactly:

    forward(fmap1, fmap2, coords, ii, jj, radius) -> tuple[Tensor]
    backward(fmap1, fmap2, coords, ii, jj, grad, radius) -> tuple[Tensor, Tensor]
    patchify_forward(net, coords, radius) -> tuple[Tensor]
    patchify_backward(net, coords, grad, radius) -> tuple[Tensor]

For `corr_forward`, compute the final bilinearly interpolated `[B, M, 2R + 1, 2R + 1, H_patch, W_patch]` tensor directly in Mojo. Do not reproduce the CUDA implementation's temporary `[B, M, 2R + 2, 2R + 2, H_patch, W_patch]` unless parity debugging requires it.

For the accepted fast path, implement these operations in `dpvo_altcorr_mojo/native/dpvo_altcorr_mojo_backends.mojo` and expose them through `PythonModuleBuilder("dpvo_altcorr_mojo_backends")`. The native module should allocate PyTorch outputs, convert tensor `data_ptr()` values to Mojo pointers, and launch GPU kernels through a cached device context. Backward operations should use atomic scatter accumulation because multiple logical work items can write the same gradient location.

Keep the CustomOpLibrary implementations as smoke coverage, but do not use `OutputTensor.unsafe_ptr()` scatter kernels as the final path unless a full-suite test that calls forward ops before backward ops proves the stale-write issue is gone.

After the standalone package passes direct tests, update `packages/dpvo/dpvo/altcorr/correlation.py` to select a backend. Default `DPVO_ALTCORR_BACKEND=auto` and explicit `DPVO_ALTCORR_BACKEND=mojo` should require `dpvo_altcorr_mojo.backend` and raise a clear error if unavailable. `DPVO_ALTCORR_BACKEND=cuda` should force the old extension. No mode should silently fall back to CUDA.

## Concrete Steps

Run commands from the repository root.

Verify CustomOpLibrary is available after Pixi edits:

    pixi run -e dpvo-dev python -c "from max.experimental.torch import CustomOpLibrary; print('custom op ok')"

Expected output:

    custom op ok

Build the native Mojo extension used by the speed gate:

    pixi run -e dpvo-dev _build-dpvo-altcorr-mojo-native

Expected result:

    packages/dpvo-altcorr-mojo/dpvo_altcorr_mojo/dpvo_altcorr_mojo_backends.so exists

Run the standalone smoke and parity tests:

    pixi run -e dpvo-dev python -m pytest packages/dpvo-altcorr-mojo/tests/test_smoke.py packages/dpvo-altcorr-mojo/tests/test_mojo_vs_cuda.py -q

Expected output:

    5 passed

Run the benchmark gate:

    pixi run -e dpvo-dev python packages/dpvo-altcorr-mojo/tools/bench_altcorr.py --json packages/dpvo-altcorr-mojo/tests/artifacts/altcorr_bench.json --max-ratio 1.05 --warmup 10 --runs 100

Expected behavior:

    The command exits 0 only if every required hot-path Mojo/CUDA median ratio is <= 1.05.

Verify the standalone import:

    pixi run -e dpvo-dev python -c "import dpvo_altcorr_mojo; print(dpvo_altcorr_mojo.__name__)"

Expected output:

    dpvo_altcorr_mojo

Verify DPVO can force each backend at import time:

    DPVO_ALTCORR_BACKEND=mojo pixi run -e dpvo-dev python -c "from dpvo import altcorr; print('mojo import ok')"
    DPVO_ALTCORR_BACKEND=cuda pixi run -e dpvo-dev python -c "from dpvo import altcorr; print('cuda import ok')"

Expected output:

    mojo import ok
    cuda import ok

Run the end-to-end DPVO demo gate with both standalone Mojo backends forced. In headless environments, save Rerun output so the process does not wait on a viewer connection:

    DPVO_ALTCORR_BACKEND=mojo DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo dpvo-demo --rr-config.save /tmp/dpvo-demo-both-mojo.rrd

Expected output includes a clean process exit and a short completion transcript:

    Total time: 30.40s
    Terminating...
    Done!
    Processed in 30.40s
    Keyframes: 320

## Validation and Acceptance

The standalone package is accepted only when it can be imported independently:

    pixi run -e dpvo-dev python -c "import dpvo_altcorr_mojo; print(dpvo_altcorr_mojo.__name__)"

Expected output:

    dpvo_altcorr_mojo

Parity tests must compare Mojo against `dpvo._cuda_corr` for `patchify_forward`, `patchify_backward`, `corr_forward`, and `corr_backward`.

Use float32 first. Forward tolerances should be `atol=1e-5, rtol=1e-5`. Backward tolerances may be `atol=2e-4, rtol=2e-4` because atomic accumulation order can differ. Assertion messages must include max and mean absolute error.

Benchmark JSON must include:

    {
      "gpu_name": "...",
      "max_ratio": 1.05,
      "passed": true,
      "cases": [
        {
          "name": "corr_forward_hot_r3_c128_p3",
          "cuda_ms": 0.123,
          "mojo_ms": 0.126,
          "ratio": 1.024,
          "passed": true
        }
      ]
    }

Required benchmark cases:

- `patchify_forward_r0_c128`
- `patchify_forward_r1_c128`
- `patchify_backward_r1_c128`
- `corr_forward_hot_r3_c128_p3`
- `corr_backward_hot_r3_c128_p3`

The current benchmark evidence is:

    PASS patchify_forward_r0_c128: cuda=0.0267ms mojo=0.0166ms ratio=0.622
    PASS patchify_forward_r1_c128: cuda=0.0267ms mojo=0.0207ms ratio=0.775
    PASS patchify_backward_r1_c128: cuda=0.0246ms mojo=0.0240ms ratio=0.975
    PASS corr_forward_hot_r3_c128_p3: cuda=0.0890ms mojo=0.0608ms ratio=0.683
    PASS corr_backward_hot_r3_c128_p3: cuda=0.6758ms mojo=0.7080ms ratio=1.048

The goal-loop metric is the conjunction of the parity tests and the speed JSON. A future agent should consider the goal regressed if tests fail, if the benchmark command exits nonzero, if any required case is missing, or if any case has `ratio > 1.05`.

The end-to-end DPVO gate must also pass before accepting the migration:

    DPVO_ALTCORR_BACKEND=mojo DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo dpvo-demo --rr-config.save /tmp/dpvo-demo-both-mojo.rrd

This command proves that DPVO can run the example video while both new standalone packages are selected. In this run, success was:

    Total time: 30.40s
    Done!
    Processed in 30.40s
    Keyframes: 320

## Idempotence and Recovery

Do not delete the CUDA implementation. It remains the oracle for tests and the explicitly selected backend for `DPVO_ALTCORR_BACKEND=cuda`.

All validation commands are safe to rerun. Rebuilding `_build-dpvo-altcorr-mojo-native` overwrites the generated shared library in place.

If the benchmark exits with `Native Mojo backend is required for the speed gate`, rebuild the native extension:

    pixi run -e dpvo-dev _build-dpvo-altcorr-mojo-native

If CustomOpLibrary uses stale Mojo code, clear only the Modular custom op cache:

    rm -f /tmp/.modular_1000/mojo_pkg/*.mojopkg

If DPVO fails with Mojo, force CUDA:

    DPVO_ALTCORR_BACKEND=cuda

If `dpvo-demo` is run in a shell without a graphical display, include `--rr-config.save /tmp/dpvo-demo-both-mojo.rrd`. Omitting this flag can still process frames but hang during Rerun viewer output shutdown.

If the 5% speed gate fails, keep parity tests passing, record the benchmark output in `Surprises & Discoveries`, and optimize before relaxing any implementation detail. Do not relax the 5% gate.

## Artifacts and Notes

The current benchmark JSON is stored at `packages/dpvo-altcorr-mojo/tests/artifacts/altcorr_bench.json` and reports:

    {
      "gpu_name": "NVIDIA GeForce RTX 5090",
      "max_ratio": 1.05,
      "passed": true,
      "cases": [
        {"name": "patchify_forward_r0_c128", "ratio": 0.6218487242285231, "passed": true},
        {"name": "patchify_forward_r1_c128", "ratio": 0.7748502616660873, "passed": true},
        {"name": "patchify_backward_r1_c128", "ratio": 0.9752603807447852, "passed": true},
        {"name": "corr_forward_hot_r3_c128_p3", "ratio": 0.6826024540617691, "passed": true},
        {"name": "corr_backward_hot_r3_c128_p3", "ratio": 1.047637074382323, "passed": true}
      ]
    }

The official Modular PyTorch custom-kernels guidance shaped the CustomOpLibrary package layout: Python loads a directory of Mojo registered ops, allocates PyTorch tensors, and calls those ops. This plan does not depend on that external page because the repository now contains the working implementation and the required commands. The MASt3R-SLAM backend shaped the native extension fast path: expose Mojo functions through `PythonModuleBuilder`, load the `.so` from Python, and operate on PyTorch tensor pointers directly.

The current full-demo gate transcript is:

    DPVO_ALTCORR_BACKEND=mojo DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo dpvo-demo --rr-config.save /tmp/dpvo-demo-both-mojo.rrd
    Processing Frames: 100%|█████████▉| 320/321 [00:27<00:00, 11.74it/s]
    Total time: 30.40s
    Terminating...
    Done!
    Processed in 30.40s
    Keyframes: 320

## Interfaces and Dependencies

`packages/dpvo-altcorr-mojo/pyproject.toml` must define a standalone Python package. It must include `operations/*.mojo`, `native/*.mojo`, and `*.so` as package data.

`pixi.toml` must make DPVO depend on the standalone package by editable path, similar to `dpretrieval`, and must provide the `_build-dpvo-altcorr-mojo-native` task.

Use `max.experimental.torch.CustomOpLibrary` for smoke registered operations. Use the native `PythonModuleBuilder` shared library for the speed-critical path.

Use current Mojo syntax: `def`, `comptime`, no `fn`, no CUDA syntax, no `__global__`, no `threadIdx`, no `kernel<<<...>>>`.

`packages/dpvo-altcorr-mojo/dpvo_altcorr_mojo/backend.py` must continue to provide:

    forward(fmap1, fmap2, coords, ii, jj, radius) -> tuple[Tensor]
    backward(fmap1, fmap2, coords, ii, jj, grad, radius) -> tuple[Tensor, Tensor]
    patchify_forward(net, coords, radius) -> tuple[Tensor]
    patchify_backward(net, coords, grad, radius) -> tuple[Tensor]

`packages/dpvo/dpvo/altcorr/correlation.py` must continue to support:

    DPVO_ALTCORR_BACKEND=auto
    DPVO_ALTCORR_BACKEND=mojo
    DPVO_ALTCORR_BACKEND=cuda

The public DPVO API remains unchanged:

    from dpvo import altcorr
    patches = altcorr.patchify(net, coords, radius, mode="bilinear")
    corr = altcorr.corr(fmap1, fmap2, coords, ii, jj, radius=3, dropout=1.0)

## Revision Note

This revision supersedes the earlier mid-migration plan that treated CustomOpLibrary as the final backend and recorded a failing speed gate. The implemented package now uses CustomOpLibrary for smoke coverage and a native Mojo Python extension for the fast path because that is the version that passes both the CUDA parity suite and the required 5% benchmark gate.

The 2026-05-06 revision adds the full DPVO demo gate with both standalone Mojo backends forced and records the half-precision no-fallback support discovered by that gate. This matters because isolated float32 operator tests did not exercise the mixed-precision inference path used by `dpvo-demo`.
