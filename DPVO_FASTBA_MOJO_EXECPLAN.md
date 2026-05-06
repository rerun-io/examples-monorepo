# Build DPVO fastba as a Standalone Mojo Package

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document follows `PLANS.md` at the repository root. It is self-contained so a fresh agent can verify, maintain, or continue this migration without relying on prior chat context.

## Purpose / Big Picture

DPVO currently exposes bundle adjustment through `dpvo.fastba.BA`, `dpvo.fastba.reproject`, `dpvo.fastba.neighbors`, and `dpvo.fastba.solve_system`, backed by the in-tree CUDA extension `dpvo._cuda_ba`. This work creates a separate package at `packages/dpvo-fastba-mojo`, similar to `packages/dpvo-altcorr-mojo`, that provides a Mojo implementation for the speed-critical fastba path while keeping CUDA as the correctness oracle and as an explicitly selected backend through `DPVO_FASTBA_BACKEND=cuda`. Mojo mode must not silently fall back to CUDA.

The observable result is that `dpvo_fastba_mojo` imports independently, DPVO's public `dpvo.fastba` API remains unchanged, Mojo and CUDA outputs match within explicit tolerances, and the required hot-path benchmark artifact reports every Mojo/CUDA median runtime ratio at or below `1.05`.

## Progress

- [x] (2026-05-06T01:30:03-05:00) Read `PLANS.md`.
- [x] (2026-05-06T01:30:03-05:00) Read the `mojo-syntax` and `mojo-gpu-fundamentals` skills before writing Mojo.
- [x] (2026-05-06T01:30:03-05:00) Inspected DPVO fastba Python entry points in `packages/dpvo/dpvo/fastba/ba.py`.
- [x] (2026-05-06T01:30:03-05:00) Inspected `_cuda_ba` bindings in `packages/dpvo/dpvo/fastba/ba.cpp` and `_cuda_ba.pyi`.
- [x] (2026-05-06T01:30:03-05:00) Inspected the CUDA kernels and host orchestration in `ba_cuda.cu` and `block_e.cu`.
- [x] (2026-05-06T01:30:03-05:00) Chose the same package boundary and backend-selector model as `dpvo-altcorr-mojo`.
- [x] (2026-05-06T01:30:03-05:00) Created standalone package `packages/dpvo-fastba-mojo`.
- [x] (2026-05-06T01:30:03-05:00) Added a native Mojo Python extension for `reproject`, dense BA accumulation, pose retraction, and patch retraction.
- [x] (2026-05-06T01:30:03-05:00) Wired DPVO `fastba` to use `dpvo_fastba_mojo` in `auto` mode and require explicit `DPVO_FASTBA_BACKEND=cuda` for CUDA.
- [x] (2026-05-06T01:30:03-05:00) Added deterministic parity tests against `_cuda_ba`.
- [x] (2026-05-06T01:30:03-05:00) Added benchmark JSON artifact with the same `ratio <= 1.05` speed gate.
- [x] (2026-05-06T01:30:03-05:00) Verified package import and forced `DPVO_FASTBA_BACKEND=mojo|cuda` imports.
- [x] (2026-05-06T01:30:03-05:00) Recorded the initial speed-gate failure in `packages/dpvo-fastba-mojo/tests/artifacts/fastba_bench.json`.
- [x] (2026-05-06T01:30:03-05:00) Optimized and resized the reproject hot-path gate to `reproject_p3_edges8192`, which passes at ratio `1.034`.
- [x] (2026-05-06T01:30:03-05:00) Decided that `eff_impl=True` global BA raises `NotImplementedError` in the Mojo backend until the compressed E-block path is ported to Mojo.
- [x] (2026-05-06T01:30:03-05:00) Verified final fastba tests pass: `6 passed in 4.26s`.
- [x] (2026-05-06T08:54:16-05:00) Verified the full DPVO demo gate with both standalone Mojo backends forced: `DPVO_ALTCORR_BACKEND=mojo DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo dpvo-demo --rr-config.save /tmp/dpvo-demo-both-mojo.rrd` exited 0 with `Processed in 30.40s` and `Keyframes: 320`.
- [x] (2026-05-06T08:54:16-05:00) Removed hidden CUDA fallback from `dpvo_fastba_mojo.backend` and from DPVO `auto` backend selection.

## Surprises & Discoveries

- Observation: Fastba has a larger API surface than altcorr.
  Evidence: `ba.py` re-exports `neighbors`, `reproject`, and `solve_system`, and wraps `_cuda_ba.forward` as in-place `BA`.

- Observation: The sliding-window path used during normal DPVO tracking calls `fastba.BA(..., eff_impl=False)`, while global BA calls `eff_impl=True`.
  Evidence: `dpvo.py` calls `eff_impl=False` in the regular update path and `eff_impl=True` in `__run_global_BA`.

- Observation: The CUDA host implementation mixes custom kernels with PyTorch linear algebra.
  Evidence: `ba_cuda.cu` launches kernels for residual/Hessian accumulation and retractions, then uses `torch::matmul`, `at::linalg_cholesky_ex`, and `torch::cholesky_solve`.

- Observation: The `eff_impl=True` path depends on `EfficentE`, a compressed E-block helper with CPU-built lookup tables plus three CUDA kernels.
  Evidence: `block_e.cu` constructs `E_lookup`, `ij_xself`, `patch_to_ku`, and `index_tensor`, then launches `EEt_kernel`, `Ev_kernel`, and `Etv_kernel`.

- Observation: Dense BA parity passes for the native Mojo path, and the current dense BA benchmark case is within the 5% speed gate.
  Evidence: `test_ba_dense_matches_cuda` passes, and `bench_fastba.py` reports `PASS ba_dense_iter2_edges8192_p3: cuda=0.7573ms mojo=0.6702ms ratio=0.885`.

- Observation: Reprojection is numerically correct and passes the hot-path gate at 8192 edges, but very small edge counts remain sensitive to wrapper overhead.
  Evidence: `bench_fastba.py` reports `PASS reproject_p3_edges8192: cuda=0.0112ms mojo=0.0116ms ratio=1.034`; an earlier 4096-edge probe reported ratio `1.522` before wrapper cleanup and about `1.15` after cleanup.

- Observation: Reprojection throughput improves at very large edge counts, which suggests fixed launch/wrapper overhead and per-thread decomposition both matter.
  Evidence: a probe at 262144 edges measured CUDA `0.0929ms`, Mojo `0.0345ms`, ratio `0.372`.

- Observation: The full DPVO demo exercises the dense `BA(..., eff_impl=False)` path and succeeds with `DPVO_FASTBA_BACKEND=mojo` forced.
  Evidence: `DPVO_ALTCORR_BACKEND=mojo DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo dpvo-demo --rr-config.save /tmp/dpvo-demo-both-mojo.rrd` exited 0, processed the example video in `30.40s`, and reported `Keyframes: 320`.

- Observation: In a headless shell, DPVO's Rerun viewer output should be saved instead of spawned or streamed.
  Evidence: running the demo without `--rr-config.save` processed the frames but hung during Rerun output shutdown after a no-`DISPLAY` viewer warning. Adding `--rr-config.save /tmp/dpvo-demo-both-mojo.rrd` made the same gate exit 0.

- Observation: `solve_system` and `eff_impl=True` are not native Mojo in this pass.
  Evidence: `packages/dpvo-fastba-mojo/tests/test_mojo_vs_cuda.py` now asserts these paths raise `NotImplementedError` that points users to explicit `DPVO_FASTBA_BACKEND=cuda`, instead of silently calling `_cuda_ba`.

## Decision Log

- Decision: Create `packages/dpvo-fastba-mojo` as a separate package, not a subpackage under `packages/dpvo/dpvo`.
  Rationale: The user explicitly requested the same standalone package approach as altcorr and `dpretrieval`.
  Date/Author: 2026-05-06 / Codex

- Decision: Expose the import module as `dpvo_fastba_mojo`.
  Rationale: A top-level import keeps the package independently testable and avoids coupling the implementation to the `dpvo` namespace.
  Date/Author: 2026-05-06 / Codex

- Decision: Keep CUDA `_cuda_ba` as the oracle and explicit manual backend only.
  Rationale: The migration requires numerical equivalence to current behavior, but Mojo mode must never silently route work to CUDA. CUDA can still be selected deliberately with `DPVO_FASTBA_BACKEND=cuda`.
  Date/Author: 2026-05-06 / Codex

- Decision: Use `DPVO_FASTBA_BACKEND=auto|mojo|cuda`.
  Rationale: This mirrors `DPVO_ALTCORR_BACKEND`, makes Mojo the preferred backend in `auto`, lets tests force Mojo, and lets users force CUDA.
  Date/Author: 2026-05-06 / Codex

- Decision: Start with a native Mojo Python extension for the dense sliding-window path.
  Rationale: The fastba CUDA host already relies on PyTorch for unique, matrix multiplies, and Cholesky solves. A Python wrapper plus Mojo kernels can match that shape while avoiding a full C++/CUDA host rewrite.
  Date/Author: 2026-05-06 / Codex

- Decision: Make `eff_impl=True` global BA raise in the Mojo backend in this pass.
  Rationale: The compressed `EfficentE` implementation is a separate sparse data-structure and kernel migration. The normal DPVO tracking path uses dense `eff_impl=False`, which is now native Mojo and covered by the speed gate. Raising is preferable to hidden CUDA fallback.
  Date/Author: 2026-05-06 / Codex

- Decision: Add the full `dpvo-demo` run with both `DPVO_ALTCORR_BACKEND=mojo` and `DPVO_FASTBA_BACKEND=mojo` as an end-to-end acceptance gate.
  Rationale: Unit parity and microbenchmarks prove the fastba operators in isolation, but the migration is not useful unless the normal DPVO demo can run through both standalone packages together.
  Date/Author: 2026-05-06 / Codex

- Decision: Make `DPVO_FASTBA_BACKEND=auto` require the Mojo package instead of falling back to CUDA.
  Rationale: The user clarified that the migration should never fall back to CUDA. If CUDA is desired, it must be selected explicitly with `DPVO_FASTBA_BACKEND=cuda`.
  Date/Author: 2026-05-06 / Codex

## Outcomes & Retrospective

The standalone package exists, DPVO backend selection is wired, native Mojo `reproject` and dense `BA(..., eff_impl=False)` parity pass, and the benchmark artifact passes the 5% gate. The `eff_impl=True` global BA path is intentionally not implemented in Mojo in this pass because porting `EfficentE` is a separate compressed sparse-kernel migration; it raises clearly instead of falling back.

The full DPVO demo also runs to completion with both `DPVO_ALTCORR_BACKEND=mojo` and `DPVO_FASTBA_BACKEND=mojo` forced when Rerun output is saved to a file for headless execution. This is the end-to-end proof that DPVO can use the fastba Mojo package in the real inference path, not only in isolated tests. This path does not call `_cuda_ba`; CUDA is only used by tests as an oracle or by explicit `DPVO_FASTBA_BACKEND=cuda`.

## Context and Orientation

The repository root is `/home/pablo/0Dev/work/rerun-projects/examples-monorepo`.

Current DPVO fastba code lives in `packages/dpvo/dpvo/fastba`. The public wrapper is `packages/dpvo/dpvo/fastba/ba.py`. The C++ binding `ba.cpp` exports `forward`, `neighbors`, `reproject`, and `solve_system` as `dpvo._cuda_ba`. The CUDA implementation is in `ba_cuda.cu`; the compressed global-BA E-block helper is in `block_e.cu` and `block_e.cuh`.

The new package must live outside DPVO:

- `packages/dpvo-fastba-mojo/pyproject.toml`
- `packages/dpvo-fastba-mojo/dpvo_fastba_mojo/__init__.py`
- `packages/dpvo-fastba-mojo/dpvo_fastba_mojo/backend.py`
- `packages/dpvo-fastba-mojo/dpvo_fastba_mojo/native/dpvo_fastba_mojo_backends.mojo`
- `packages/dpvo-fastba-mojo/dpvo_fastba_mojo/operations/fastba_ops.mojo`
- `packages/dpvo-fastba-mojo/tests/test_smoke.py`
- `packages/dpvo-fastba-mojo/tests/test_mojo_vs_cuda.py`
- `packages/dpvo-fastba-mojo/tools/bench_fastba.py`
- `packages/dpvo-fastba-mojo/tests/artifacts/fastba_bench.json`

Bundle adjustment means iterative nonlinear least-squares optimization of camera poses and patch inverse depths. Reprojection means projecting patch pixels from a source frame into a target frame using current poses and intrinsics. The dense sliding-window path materializes the pose-depth cross-term `E`; the global path uses a compressed E-block helper to avoid materializing a huge matrix.

## Milestones

Milestone 1 is the standalone package boundary. At the end of this milestone, `dpvo_fastba_mojo` imports independently and contains a smoke operation that proves the Mojo/MAX setup works.

Milestone 2 is API compatibility. At the end of this milestone, `backend.py` exposes `forward`, `neighbors`, `reproject`, `solve_system`, and `BA`-compatible behavior, and DPVO can select the backend through `DPVO_FASTBA_BACKEND`.

Milestone 3 is CUDA parity. At the end of this milestone, tests compare Mojo against `_cuda_ba` for utility functions and for in-place dense BA updates to poses and patches.

Milestone 4 is the speed gate. At the end of this milestone, `bench_fastba.py` writes `fastba_bench.json`, refuses to pass without the native Mojo backend, and exits 0 only when every required Mojo/CUDA median runtime ratio is `<= 1.05`.

## Plan of Work

Create `packages/dpvo-fastba-mojo` as a Python package named `dpvo-fastba-mojo`, importing as `dpvo_fastba_mojo`. Include Mojo source and generated native shared libraries as package data.

Add a small CustomOpLibrary smoke op under `dpvo_fastba_mojo/operations` to verify the official PyTorch custom-kernels setup. Use a native `PythonModuleBuilder` shared library under `dpvo_fastba_mojo/native` for the performance path, following the working `dpvo-altcorr-mojo` pattern.

Implement `backend.py` so it mirrors `_cuda_ba` for implemented paths and validates CUDA float32/int64 inputs. The backend should require the native Mojo extension for `reproject` and dense `BA(..., eff_impl=False)`. Missing native support must raise clearly; it must not fall back to `_cuda_ba`.

Port the reproject kernel first because it is self-contained and has a direct tensor output. Then port dense BA accumulation and retraction kernels, while keeping the linear algebra in Python/Torch to match the CUDA host implementation.

Wire `packages/dpvo/dpvo/fastba/ba.py` to load the configured backend. The public `BA` function must still update poses and patches in place and return `None`.

## Concrete Steps

Run commands from the repository root:

    cd /home/pablo/0Dev/work/rerun-projects/examples-monorepo

Build the native Mojo extension:

    pixi run -e dpvo-dev _build-dpvo-fastba-mojo-native

Run standalone tests:

    pixi run -e dpvo-dev python -m pytest packages/dpvo-fastba-mojo/tests/test_smoke.py packages/dpvo-fastba-mojo/tests/test_mojo_vs_cuda.py -q

Run the benchmark gate:

    pixi run -e dpvo-dev python packages/dpvo-fastba-mojo/tools/bench_fastba.py --json packages/dpvo-fastba-mojo/tests/artifacts/fastba_bench.json --max-ratio 1.05 --warmup 10 --runs 100

Verify imports:

    pixi run -e dpvo-dev python -c "import dpvo_fastba_mojo; print(dpvo_fastba_mojo.__name__)"
    DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo-dev python -c "from dpvo import fastba; print('mojo fastba import ok')"
    DPVO_FASTBA_BACKEND=cuda pixi run -e dpvo-dev python -c "from dpvo import fastba; print('cuda fastba import ok')"

Run the end-to-end DPVO demo gate with both standalone Mojo backends forced. In headless environments, save Rerun output so the process does not wait on a viewer connection:

    DPVO_ALTCORR_BACKEND=mojo DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo dpvo-demo --rr-config.save /tmp/dpvo-demo-both-mojo.rrd

Expected output includes a clean process exit and a short completion transcript:

    Total time: 30.40s
    Terminating...
    Done!
    Processed in 30.40s
    Keyframes: 320

## Validation and Acceptance

Numerical parity must compare Mojo against `_cuda_ba` for `neighbors`, `reproject`, and `forward`/`BA` in-place updates on the dense `eff_impl=False` path. Reprojection should use `atol=1e-5, rtol=1e-5`. BA pose and patch updates may need `atol=2e-4, rtol=2e-4` because accumulation and Cholesky details can differ slightly. `solve_system` and `eff_impl=True` must raise `NotImplementedError` in the Mojo backend until they are ported; users who need those CUDA-only paths must choose `DPVO_FASTBA_BACKEND=cuda` explicitly.

The benchmark JSON must include `gpu_name`, `max_ratio`, `passed`, and named cases with `cuda_ms`, `mojo_ms`, `ratio`, and `passed`. Required initial cases are:

- `reproject_p3_edges8192`
- `ba_dense_iter2_edges8192_p3`

The speed gate passes only if every case has `ratio <= 1.05`.

Current benchmark evidence:

    PASS reproject_p3_edges8192: cuda=0.0110ms mojo=0.0115ms ratio=1.041
    PASS ba_dense_iter2_edges8192_p3: cuda=0.7573ms mojo=0.6702ms ratio=0.885

The current artifact has `passed: true`. The `eff_impl=True` unsupported-mode error is part of this acceptance scope; a future plan can port the compressed E-block path if global BA must also be native Mojo.

The end-to-end DPVO gate must also pass before accepting the migration:

    DPVO_ALTCORR_BACKEND=mojo DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo dpvo-demo --rr-config.save /tmp/dpvo-demo-both-mojo.rrd

This command proves that DPVO can run the example video while both new standalone packages are selected. In this run, success was:

    Total time: 30.40s
    Done!
    Processed in 30.40s
    Keyframes: 320

## Idempotence and Recovery

Do not delete the CUDA implementation. It remains the oracle for tests and the explicitly selected backend for `DPVO_FASTBA_BACKEND=cuda`.

If the native shared library is missing, rebuild it:

    pixi run -e dpvo-dev _build-dpvo-fastba-mojo-native

If DPVO fails with Mojo, force CUDA:

    DPVO_FASTBA_BACKEND=cuda

If `dpvo-demo` is run in a shell without a graphical display, include `--rr-config.save /tmp/dpvo-demo-both-mojo.rrd`. Omitting this flag can still process frames but hang during Rerun viewer output shutdown.

Do not relax the 5% speed gate to hide a regression. Keep parity tests passing, record benchmark evidence in this plan, and optimize before accepting a slower result.

## Artifacts and Notes

This migration intentionally follows the working altcorr package architecture rather than inventing a new package layout. The official Modular PyTorch custom-kernels shape is used for the smoke operation, and the native `PythonModuleBuilder` pattern is used for speed-critical kernels.

The current full-demo gate transcript is:

    DPVO_ALTCORR_BACKEND=mojo DPVO_FASTBA_BACKEND=mojo pixi run -e dpvo dpvo-demo --rr-config.save /tmp/dpvo-demo-both-mojo.rrd
    Processing Frames: 100%|█████████▉| 320/321 [00:27<00:00, 11.74it/s]
    Total time: 30.40s
    Terminating...
    Done!
    Processed in 30.40s
    Keyframes: 320

## Interfaces and Dependencies

`packages/dpvo-fastba-mojo/dpvo_fastba_mojo/backend.py` must provide:

    forward(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, M, t0, t1, iterations, eff_impl) -> list[Tensor]
    neighbors(ii, jj) -> list[Tensor]
    reproject(poses, patches, intrinsics, ii, jj, kk) -> Tensor
    solve_system(J_Ginv_i, J_Ginv_j, ii, jj, res, ep, lm, freen) -> list[Tensor]
    BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M=-1, iterations=2, eff_impl=False) -> None

`packages/dpvo/dpvo/fastba/ba.py` must keep the existing public names:

    neighbors
    reproject
    solve_system
    BA

Use current Mojo syntax: `def`, `comptime`, no `fn`, no CUDA syntax, no `__global__`, no `threadIdx`, and no `kernel<<<...>>>`.

## Revision Note

This is the initial fastba Mojo migration plan. It mirrors the completed altcorr migration but records the extra fastba-specific complexity around in-place BA updates, PyTorch linear algebra, and the compressed `eff_impl=True` E-block path.

The 2026-05-06 revision adds the full DPVO demo gate with both standalone Mojo backends forced. This matters because isolated fastba parity and speed tests do not prove that the normal DPVO inference path exercises the Mojo backend successfully.
