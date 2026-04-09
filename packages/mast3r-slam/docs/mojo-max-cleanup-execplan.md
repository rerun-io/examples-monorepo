# Clean Up Mojo and MAX Interop Boundaries

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained under the rules in `.agents/PLANS.md`.

## Purpose / Big Picture

After this change, the `mast3r-slam` Mojo and MAX backends should be easier to maintain without changing runtime behavior. The unsafe Python-to-Mojo pointer conversions will be isolated behind small, documented helper functions instead of being repeated throughout the hot paths, the MAX custom-op package will have clearer comments about why it uses `InputTensor` / `OutputTensor` and launched kernels, and the repository will explicitly document whether the nested `packages/mast3r-slam/max-custom-ops/pixi.toml` is still necessary.

The user-visible proof is unchanged behavior: the Mojo shared library still builds, the GN parity tests still pass, and the real GN fixture benchmark still stays within the previously verified accuracy and throughput envelope.

## Progress

- [x] (2026-04-09 15:10 CDT) Reviewed `.agents/PLANS.md` and the Mojo skills required for syntax, GPU programming, and Python interop.
- [x] (2026-04-09 15:18 CDT) Audited the current shared-lib Mojo backend and MAX custom-op package to find repeated `UnsafePointer` boundaries and context-caching patterns.
- [x] (2026-04-09 15:22 CDT) Verified that the root `mast3r-slam-dev` Pixi environment still cannot import `max.experimental.torch`; this means the nested `packages/mast3r-slam/max-custom-ops/pixi.toml` remains necessary today.
- [x] (2026-04-09 16:02 CDT) Added `packages/mast3r-slam/mast3r_slam/backend/mojo/python_interop.mojo` and moved the shared-lib pointer conversion and cached `DeviceContext` logic into it.
- [x] (2026-04-09 16:05 CDT) Updated `gn.mojo`, `matching.mojo`, and `mast3r_slam_mojo_backends.mojo` to use the centralized interop helpers and added comments at the unsafe boundary.
- [x] (2026-04-09 16:08 CDT) Added comments to the MAX custom-op module and documented in the nested `max-custom-ops/pixi.toml` why that workspace still exists.
- [x] (2026-04-09 16:10 CDT) Marked `matching_kernels.mojo` as legacy reference code instead of a current source-of-truth module.
- [x] (2026-04-09 16:20 CDT) Rebuilt the Mojo backend, reran the GN parity tests and MAX custom-op tests, and reran the real GN fixture benchmark without regression.

## Surprises & Discoveries

- Observation: the root monorepo environment includes the Modular channel and `mojo`, but it still does not provide the Python `max` package needed by `max.experimental.torch`.
  Evidence:
    `pixi run -e mast3r-slam-dev python - <<'PY'`
    `try: import max.experimental.torch as t; print("ROOT_MAX_OK", t.__name__)`
    `except Exception as e: print("ROOT_MAX_FAIL", type(e).__name__, e)`
    `PY`
    prints `ROOT_MAX_FAIL ModuleNotFoundError No module named 'max'`.

- Observation: the current shared-library Mojo path repeats the same `Int(py=tensor.data_ptr())` plus `UnsafePointer[..., MutAnyOrigin](unsafe_from_address=...)` conversion pattern in multiple files.
  Evidence: `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo` and `packages/mast3r-slam/mast3r_slam/backend/mojo/matching.mojo` both perform the conversions inline for every tensor argument.

- Observation: centralizing the unsafe interop did not change the real GN public-path benchmark in any meaningful way.
  Evidence:
    `pixi run -e mast3r-slam-dev python packages/mast3r-slam/tools/bench_gn_real_fixtures.py packages/mast3r-slam/artifacts/gn-fixtures/verify-base/rays-000.pt --warmup 5 --runs 20`
    still reports `public selected ms 22.411`, `ratio 0.433`, with `dtrans 2.98e-07`, `dquat 1.68e-08`, `dscale 5.96e-08`, `ddx 1.75e-07`.

- Observation: the MAX custom-op path still validates cleanly after the comment and structure cleanup.
  Evidence:
    `pixi run --manifest-path packages/mast3r-slam/max-custom-ops/pixi.toml test-max-custom-ops`
    reports `4 passed in 11.10s`.

## Decision Log

- Decision: keep the nested `packages/mast3r-slam/max-custom-ops/pixi.toml` for now.
  Rationale: the root `mast3r-slam-dev` environment cannot import `max.experimental.torch`, so deleting the nested MAX environment would break the MAX custom-op tests and benchmarks.
  Date/Author: 2026-04-09 / Codex

- Decision: isolate unsafe interop rather than trying to eliminate all unsafe pointers from the shared-lib backend.
  Rationale: the `PythonModuleBuilder` path still receives raw PyTorch storage pointers from `tensor.data_ptr()`. The realistic cleanup is to centralize and document that boundary, while leaving the MAX custom-op path as the more idiomatic tensor-based reference implementation.
  Date/Author: 2026-04-09 / Codex

- Decision: keep `matching_kernels.mojo` in the tree, but mark it as legacy instead of deleting it.
  Rationale: it is no longer the build target, but it is still referenced in the in-repo optimization notes and prior handoff artifacts. A prominent legacy note avoids misleading readers without risking accidental documentation breakage during this cleanup pass.
  Date/Author: 2026-04-09 / Codex

## Outcomes & Retrospective

The cleanup succeeded without changing the live behavior. The shared-lib backend now has one obvious place to audit every raw pointer conversion and cached `DeviceContext` lookup, and the MAX custom-op package better documents why it uses tensor abstractions instead of the raw-pointer pattern from the shared-lib extension.

The nested MAX Pixi workspace is still required today because the root monorepo environment cannot import `max.experimental.torch`. That means the cleanup should stop at documentation and structure for now; removing that workspace would be a separate environment project.

No accuracy or throughput regression was observed on the real GN verification fixture. The public-path benchmark remained faster than CUDA, and the GN drift relative to CUDA stayed within the previously verified tiny tolerances.

## Context and Orientation

The `mast3r-slam` package currently has two Mojo-facing integration styles.

`packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo` builds a Python extension module through `PythonModuleBuilder`. That module exposes `iter_proj`, `refine_matches`, and the GN entrypoints. The shared-lib path talks directly to PyTorch tensors by reading `tensor.data_ptr()` in Mojo and then reinterpreting the address as an `UnsafePointer`. This is the area with the most cleanup debt.

`packages/mast3r-slam/mast3r_slam/backend/max_ops/gn_ops/__init__.mojo` implements MAX PyTorch custom ops. That code uses `InputTensor`, `OutputTensor`, `DeviceContextPtr`, and `to_layout_tensor()`. It is already closer to the current Modular examples, but it still needs comments and clearer structure.

`packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo` contains the GN Python interop layer for the shared-lib backend. `packages/mast3r-slam/mast3r_slam/backend/mojo/matching.mojo` contains the matching interop layer. `packages/mast3r-slam/mast3r_slam/backend/mojo/gn_kernels.mojo` contains the GN GPU kernels. `packages/mast3r-slam/mast3r_slam/backend/mojo/matching_kernels.mojo` is an older combined file from before the split into `matching.mojo`, `gn.mojo`, and `gn_kernels.mojo`.

In this repository, an unsafe pointer boundary means the point where Mojo reinterprets a raw integer address from Python or PyTorch as a typed pointer. That boundary is acceptable only if the tensor has already been made contiguous with the expected dtype and if the tensor object itself stays alive for the full kernel launch.

## Plan of Work

First, add a new helper module under `packages/mast3r-slam/mast3r_slam/backend/mojo/` that owns all shared-lib Python interop utilities. Move the cached `DeviceContext` lookup there, add helper functions that turn a contiguous PyTorch tensor `PythonObject` into a typed `UnsafePointer`, and put comments directly at the unsafe boundary describing the safety requirements. Then update `gn.mojo`, `matching.mojo`, and the extension-module initializer to use those helpers instead of repeating pointer conversions inline.

Second, clean up the MAX custom-op module in `packages/mast3r-slam/mast3r_slam/backend/max_ops/gn_ops/__init__.mojo`. Keep the existing behavior, but add concise comments that explain why the code uses `InputTensor` / `OutputTensor`, when `foreach` is used, and why the launched-kernel path is preferred on GPU. Avoid introducing raw-pointer style APIs there because the MAX runtime already provides higher-level tensor abstractions.

Third, decide how to handle `packages/mast3r-slam/mast3r_slam/backend/mojo/matching_kernels.mojo`. If it is truly stale and no longer part of the build path, remove it. If it still compiles as part of the wildcard `_build-mojo-kernels` task or serves as documentation, keep it but mark it clearly as legacy so readers are not misled.

Finally, rerun the build, tests, and real-fixture benchmark, then update this document with the observed outputs and any throughput changes.

That validation is now complete. The helper-module refactor did not change the selected-backend GN public-path result on the captured `rays-000` fixture in any material way.

## Concrete Steps

Run all commands from the repository root `/var/tmp/vibe-kanban/worktrees/4658-mast3r-slam-mojo/examples-monorepo`.

Create the cleanup helper module and update the shared-lib backend:

    pixi run -e mast3r-slam-dev _build-mojo-kernels

Run the GN parity tests:

    pixi run -e mast3r-slam-dev pytest \
      packages/mast3r-slam/tests/test_gn_fixture_utils.py \
      packages/mast3r-slam/tests/test_gn_step_api.py -q

Run the real GN fixture benchmark:

    pixi run -e mast3r-slam-dev python \
      packages/mast3r-slam/tools/bench_gn_real_fixtures.py \
      packages/mast3r-slam/artifacts/gn-fixtures/verify-base/rays-000.pt \
      --warmup 5 --runs 20

If MAX code is edited, rerun the nested MAX custom-op tests from the repository root:

    pixi run --manifest-path packages/mast3r-slam/max-custom-ops/pixi.toml test-max-custom-ops

Observed outputs from this cleanup pass:

    pixi run -e mast3r-slam-dev pytest \
      packages/mast3r-slam/tests/test_gn_fixture_utils.py \
      packages/mast3r-slam/tests/test_gn_step_api.py -q
    ...
    9 passed in 0.88s

    pixi run --manifest-path packages/mast3r-slam/max-custom-ops/pixi.toml test-max-custom-ops
    ...
    4 passed in 11.10s

    pixi run -e mast3r-slam-dev python \
      packages/mast3r-slam/tools/bench_gn_real_fixtures.py \
      packages/mast3r-slam/artifacts/gn-fixtures/verify-base/rays-000.pt \
      --warmup 5 --runs 20
    fixture                  scope    kind        cuda ms  selected ms    ratio     dtrans      dquat     dscale        ddx
    rays-000                 step     rays          5.817        2.317    0.398        nan        nan        nan   1.64e+04
    rays-000                 public   rays         51.727       22.411    0.433   2.98e-07   1.68e-08   5.96e-08   1.75e-07

## Validation and Acceptance

Acceptance means the code is cleaner without changing behavior. The Mojo shared library must still build. The GN tests must still pass. The real fixture benchmark must still report tiny GN drift relative to CUDA and must not show a throughput regression relative to the previously recorded signoff numbers for the selected backend.

The nested MAX environment decision is accepted when this document explicitly states whether that environment is still required and includes a command proving the answer.

This pass meets those acceptance criteria:

- `pixi run -e mast3r-slam-dev _build-mojo-kernels` succeeded.
- The GN tests reported `9 passed`.
- The nested MAX tests reported `4 passed`.
- The real GN fixture benchmark kept the public-path ratio at `0.433` and the pose/update drift at the previously tiny levels.
- The command proving the root environment still lacks MAX support is captured in `Surprises & Discoveries`.

## Idempotence and Recovery

These steps are safe to rerun. The build task overwrites the shared library in place. The tests are read-only. The benchmark reads a captured fixture and does not mutate repository source files. If a Mojo refactor breaks compilation, rerun `_build-mojo-kernels` after each patch until the helper module and imports stabilize.

## Artifacts and Notes

Expected proof snippets will be added here after the implementation is complete.

Key files changed in this cleanup:

- `packages/mast3r-slam/mast3r_slam/backend/mojo/python_interop.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/gn.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/matching.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo`
- `packages/mast3r-slam/mast3r_slam/backend/max_ops/gn_ops/__init__.mojo`
- `packages/mast3r-slam/max-custom-ops/pixi.toml`
- `packages/mast3r-slam/mast3r_slam/backend/mojo/matching_kernels.mojo`

## Interfaces and Dependencies

The shared-lib backend must continue to expose the same Python-facing functions from `packages/mast3r-slam/mast3r_slam/backend/mojo/mast3r_slam_mojo_backends.mojo`: `iter_proj`, `refine_matches`, `pose_retr`, `gauss_newton_rays_step`, and the three GN implementation entrypoints.

The new helper module must provide stable utilities for:

    get_cached_context_ptr() raises -> UnsafePointer[DeviceContext, MutAnyOrigin]

and typed conversions from a `PythonObject` tensor to an `UnsafePointer` with `MutAnyOrigin`.

The MAX custom-op package must continue to expose the registered ops loaded by `packages/mast3r-slam/mast3r_slam/max_ops.py` through `max.experimental.torch.CustomOpLibrary`.

Revision note: created this ExecPlan to drive the Mojo/MAX cleanup requested after the GN port stabilized. The plan records that the nested MAX Pixi project remains necessary because the root monorepo environment cannot currently import `max.experimental.torch`.

Revision note: updated after implementation to record the centralized interop helper module, the MAX documentation cleanup, the legacy status of `matching_kernels.mojo`, and the no-regression validation results.
