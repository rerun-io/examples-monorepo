# CUDA 13 / torch 2.12 / torchcodec 0.13 — monorepo migration plan

**Status: EXECUTED on branch `cuda13-migration` (see "Execution results" at the
bottom for what actually happened — including two of this plan's premises that
the hardware disproved).** Original plan retained below for reference.

## 0. Validated target stack (all confirmed on conda-forge / NVIDIA PyPI)
| component | spec | platforms |
|---|---|---|
| pytorch-gpu | `2.12.1` build `cuda130_mkl` | linux-64 **+ linux-aarch64** |
| torchcodec | `0.13.0` build `cuda130_py312` | linux-64 + aarch64 |
| tensorrt-cu13 | `==10.13.3.9` (same API ver as current cu12) | rebuild engine |
| cuda-version | `13.0.*` | |
| onnxruntime-gpu | already pinned to `ort-cuda-13-nightly` index | (unchanged) |

torchcodec 0.13 includes the NVDEC-cache fix (PR #1232, landed 0.11) → the in-process
multi-stream decode no longer thrashes → the `MultiprocessDecoder` hack can be deleted.

**Current-state motivation (mamma realtime gate):** as of the PR #48 merge, the mamma
`mamma-goal-check` **realtime** clause is RED purely from an env mismatch —
`torchvision 0.26` resolved against the pinned `torch 2.10` (the env even warns
"torchvision==0.26 is incompatible with torch==2.10"), which slows the SAM2
image-encoder (`track`) stage to ~72 ms/call (~7 ticks/s vs the ~15 needed). This was
A/B-proven identical on the pre-slim sam2 code, so it is NOT caused by the slim. The
consistent torch-2.12 stack in this migration is expected to resolve it; the
validation signal is re-running `packages/mamma/tools/benchmark.py --trt-engine <plan>`
(or `mamma-goal-check`) after the migration and confirming realtime goes green.

## 1. Blast radius
Flipping the shared `[feature.cuda]` from 12.9→13.0 cascades to **16 GPU packages**:
dpvo, egoexo-forge, mamma, mast3r-slam, monoprior, mv-api, prompt-da, pysfm, sam3,
sam3d-body, sapiens-coco133-pose, sapiens2-pose, simplecv, vistadream, wilor-nano
(+ each `-dev`). The entire `pixi.lock` re-solves; every GPU env re-installs; TRT
engines rebuild; each package needs re-validation.

## 2. Exact pixi.toml edits

### A. Shared `[feature.cuda]` (the switch)
- `[feature.cuda.system-requirements]` line 88: `cuda = "12.9"` → `cuda = "13.0"`
- `[feature.cuda.dependencies]` line 93: `cuda-version = "12.9.*"` → `cuda-version = "13.0.*"`
- line ~115: `pytorch-gpu = ">=2.8.0"` → `pytorch-gpu = ">=2.12,<2.13"` (forces cuda130/2.12.1; explicit beats relying on cuda-version alone)
- All toolchain deps (`cuda-compiler`, `cudnn`, `libcublas-dev`, `cusparselt`, …) stay `*` → float to cuda13 builds automatically.
- `onnxruntime-gpu` already on the cuda-13 nightly index → leave as-is.

### B. Remove the two torchcodec `<0.11` caps
- line 476 `[feature.wilor-nano.dependencies]`: `torchcodec = ">=0.10.0,<0.11"` → `torchcodec = "*"`
- line 862 `[feature.sapiens-coco133-pose.dependencies]`: same → `torchcodec = "*"`
- (`*` pins at 174/665/1201/1703 float to 0.13 automatically once torch is 2.12.)

### C. `tensorrt-cu12` → `tensorrt-cu13` (version stays `==10.13.3.9`, index same)
4 sites: line 487 (wilor-nano), 786 (sapiens2-pose), 876 (sapiens-coco133-pose), 1715 (mamma).

### D. `[feature.mamma]`
- line 1697-1698: drop the `# Keep the torch everything was measured/gated on` comment and change
  `pytorch-gpu = "2.10.*"` → remove it (let the shared cuda floor drive 2.12.1) **or** `= "2.12.*"`.
- torchcodec `*` (line 1703) unchanged → floats to 0.13.

### E. Leave alone / verify
- vistadream line 1074 `pytorch-gpu = ">=2.10.0"` — floor, floats up; no change.
- robocap line 914 `cuda-version = "13.*"` (CPU-torch, already cuda13) — once the shared
  feature is also 13, the conflict note at line ~922 is moot; optional cleanup, low priority.

## 3. Code changes — delete the mp_decode hack (mamma)
- **Delete** `packages/mamma/src/mamma/engine/mp_decode.py`
- **Delete** `packages/mamma/tests/test_mp_decode_startup.py` (it guards the removed hack)
- `packages/mamma/src/mamma/engine/pipeline.py`:
  - drop `use_mp_decode` param (StreamingPipeline.__init__ + build_streaming_pipeline),
    the `self.mp_decoder` construction, the `iter_chunks` mp branch in `chunk_iter()`,
    and the mp handling in `close()`. The in-process prefetcher branch (already the
    fallback) becomes the only decode path.
- `packages/mamma/tools/dump_artifacts.py` + `tools/validate_dynamic.py`:
  drop the `mp_decode: bool` config field and the `use_mp_decode=` arg.
- **Perf check (the whole point):** measure in-process torchcodec-0.13 multi-stream cam-fps
  on 4×4K HEVC; confirm it's ~400 (cache fixed), not ~140. If it regresses, reconsider.

## 4. Engine rebuild
- mamma: `tools/build_trt_engine.py` on the cu13 stack → new plan
  (filename ver bumps, e.g. `mammanet_b4_fp16_trt<cu13ver>_sm120.plan`). Update the
  `--trt-engine` default / sweep references.
- sapiens2/sapiens-coco133/wilor: rebuild any cached TRT engines on cu13.

## 5. aarch64 / DGX Spark (separate axis — do AFTER x86 cuda13 is green)
- `[feature.cuda]` + `[feature.common]` are `platforms = ["linux-64"]`. Add
  `"linux-aarch64"` to enable DGX Spark. This is a bigger, independent effort:
  every PyPI-only dep (tensorrt, gradio, model wheels, …) must have aarch64 wheels;
  expect gaps. Treat as Phase 3.

## 6. Execution order
1. New branch off `main` (e.g. `cuda13-migration`) — keep the validated mamma PR clean.
2. Apply §2 edits.
3. `pixi lock` (full re-solve; slow).
4. `pixi install -e <each GPU env>`.
5. Rebuild TRT engines (§4).
6. Remove mp_decode hack (§3) + simplify.
7. Re-validate:
   - **mamma**: re-run the 42-scene sweep; diff gate.json vs `old_…_rrds` (expect tiny
     torch-2.12 numeric drift; single-person margins are wide so verdicts should hold);
     confirm in-process decode cam-fps.
   - **each other GPU package**: `pixi run -e <pkg>-dev tests` + one smoke demo.
8. `lint` / `typecheck` / `deadcode` on every changed package.

## 7. Risks
- torch 2.10→2.12 numeric drift → mamma gate verdicts *could* shift (low risk: single-person
  ~13–17 mm vs 30 mm bar). Must re-run the sweep to confirm.
- TRT engines are CUDA/stack-specific → mandatory rebuild; DGX engine differs again.
- `onnxruntime-gpu` is on a **nightly** cuda-13 index → stability risk; pin/verify.
- 16 packages × validation is large; several need GPU + downloaded weights/data.
- aarch64 wheels for PyPI-only deps may be missing → Phase 3 will surface gaps.

## 8. Suggested phasing
- **Phase 1 (proves the stack):** shared `cuda` feature + simplecv + mamma → cuda13;
  remove the hack; re-validate the mamma sweep. Smallest set that exercises the full path.
- **Phase 2:** remaining GPU packages, one at a time, each validated.
- **Phase 3:** add `linux-aarch64` for DGX Spark.

---

## 9. Execution results (branch `cuda13-migration`, RTX 5090 / sm120)

### What landed
- **Shared stack flipped:** `[feature.cuda]` cuda 12.9→13.0, `cuda-version 13.0.*`,
  `pytorch-gpu >=2.12,<2.13`. Resolves to `pytorch 2.12 cuda130_mkl`,
  `torchvision 0.27.1 cuda130`, `torchcodec 0.14.0 cuda130` (the `*` float landed on
  0.14, newer than the planned 0.13 — supersedes it, NVDEC fix included),
  `triton 3.7 cuda130`. **NB:** the `pytorch-gpu` *metapackage* keeps a cosmetic
  `cuda129`-labelled build string but pulls the real cuda130 `pytorch` — §0's
  "cuda130_mkl pytorch-gpu build" doesn't exist as such; the underlying `pytorch`
  does.
- **`pixi lock` re-solves** for all 15 reachable GPU envs (every one except pysfm).
- **torchcodec caps removed** (wilor-nano, sapiens-coco133); wilor-nano `torchvision`
  un-capped from `<0.26` (torch 2.12 needs 0.27).
- **`tensorrt-cu12`→`tensorrt-cu13`** (4 sites). **Gotcha:** `tensorrt-cu13-libs`
  has stale metadata requiring `nvidia-cuda-runtime-cu13`, which NVIDIA deprecated
  for cu13 (only broken 0.0.x sdist stubs exist; the real package is the unsuffixed
  `nvidia-cuda-runtime`). conda provides the runtime, so each tensorrt feature pins
  `nvidia-cuda-runtime-cu13 = "==0.0.0a0"` (NVIDIA's empty no-op placeholder wheel).
- **mast3r-slam:** the `mkl <2026` cap can't coexist with torch 2.12 (needs mkl 2026
  `.so.3`) and faiss `cpu_mkl` (`.so.2`). Routed this env's BLAS through OpenBLAS
  (`libblas *openblas` + `faiss *openblas*` → cuda130_generic pytorch, nomkl) — GPU
  kernels identical, only CPU BLAS differs.
- **lietorch:** no cuda13 build existed (ai-demos only had cuda12.9). Rebuilt for
  cuda13 in the `ai-demos` channel (recipe: added a 13.0 CUDA variant to
  `conda_build_config`, stripped sm_60/61/70 gencode — CUDA 13 dropped
  Maxwell/Pascal/Volta — and bumped CI `CONDA_OVERRIDE_CUDA`). Published as
  `lietorch-1.0-hecd5ce2_1` (cuda-version >=13). Validated: `dpvo` + `mast3r-slam`
  import `lietorch.SE3` on cuda13.
- **mamma:** mp_decode hack removed (mp_decode.py + its test deleted; `use_mp_decode`
  dropped from pipeline/tools). `vit.py PatchEmbed.forward` coerces patch-grid dims
  to `int` (torch 2.12 returns 0-dim tensors from shape access under the ONNX trace).
  **MammaNet TRT engine rebuilt on cu13** → `.trt_cache/mammanet_b4_fp16_trt101339_sm120.plan`
  (joints2d diff vs eager max 0.009; TRT 3.6 ms/call).
- **simplecv:** `required_torchcodec_float` widened to accept `int` (torchcodec 0.14
  reports whole-number `average_fps` as `int`).

### Per-env validation (`pixi run -e <pkg>-dev tests`)
PASS (15): simplecv (132), mamma (15 + golden gate PASS), dpvo (no tests;
lietorch cuda13 import ok), monoprior (1), sapiens2-pose (24), mast3r-slam (27),
sam3 (1), sam3d-body (1), prompt-da (3), mv-api (55), mv-api-catalog (55),
egoexo-forge (1), sapiens-coco133-pose (31), **vistadream** (1; lint/typecheck/
deadcode clean — recovered via the PyPI gsplat fix below).
- **wilor-nano:** 24 pass, 1 fail — `test_pixi_tasks_keep_generic_wilor_entrypoints`
  reads a `[feature.wilor]` that doesn't exist (only `wilor-nano`). **Pre-existing
  on `main`** (fails there too); not a migration regression — stale since the
  `wilor`→`wilor-nano` rename.
- **vistadream: FIXED.** Original failure: the `gsplat` *git* dep built in an
  isolated uv env that pulled a PyPI cu128 torch (`no-build-isolation` is **not
  applied to git deps**), so gsplat's `_check_cuda_version` saw nvcc 13.0 vs torch
  12.8 (worked pre-migration only because 12.9-vs-12.8 was a tolerated *minor*
  mismatch; the cuda13 *major* bump trips it). `conda-pypi-map pytorch→torch` did
  not help (no-build-isolation still skipped the git dep). **Fix:** switch gsplat
  from the git rev to the PyPI dep `gsplat ==1.5.3` — the pinned rev `970dd84`
  *is* v1.5.3, and PyPI ships a `py3-none-any` wheel that installs without any
  build and JIT-compiles its CUDA kernels at runtime against the conda cuda13
  torch. vistadream-dev now installs + imports + tests pass + lint/typecheck/
  deadcode clean on cuda13.
- **pysfm:** still disabled (multiple compounding cuda13 conflicts; the libfaiss
  one is now solved, two remain):
  1. *libfaiss (SOLVED):* `pycolmap ==4.0.2` hard-requires `libfaiss * *_cuda`
     (>=1.9,<2) and no cuda13 faiss existed. Built **libfaiss 1.10.0 cuda13**
     (`cuda130h*_cuda`, FAISS_ENABLE_GPU=ON) in ai-demos → pycolmap (cpu_py312)
     resolves on cuda13. Verified by an isolated solve.
  2. *Qt5/Qt6:* pycolmap pulls `qt-main` (Qt5) but the floated py-opencv 4.13
     wants a Qt6 `libopencv`. A `headless` opencv sidesteps Qt but pulls in (3).
  3. *typing-extensions:* `tyro >=0.9.1` (recent builds) needs
     `typing-extensions >=4.13`, but `[feature.common]` caps it `<4.13` (shared by
     every env — not safe to relax for one).

  **Attempted (conda pycolmap):** pinning `py-opencv` to a Qt5 build (to match
  pycolmap's Qt5) fails too — the only Qt5 libopencv for py312 is 4.10.0, whose
  deps pin `imath <3.2` which conflicts with the cuda13 stack's imath. The conda
  pycolmap 4.0.2 is a **Qt5-era package** whose transitive deps (Qt5, old imath,
  faiss `_cuda`) are incompatible with the modern cuda13/py312 ecosystem.

  **Attempted (PyPI pycolmap — closest path):** switching to the self-contained
  PyPI wheel `pycolmap ==4.0.2` (cp312 manylinux) makes the env **solve + install
  + `import pycolmap`** on cuda13 (it bundles COLMAP/faiss, sidestepping all the
  conda Qt5/faiss/imath clashes; lint + deadcode clean, the blueprint test passes).
  BUT it **core-dumps** in actual feature extraction
  (`test_streamed_extraction_equivalence_synthetic`) — the wheel's bundled native
  libs crash against the conda env. So pysfm stays disabled (the env's `pycolmap`
  dep is left on the PyPI wheel as the closest-working direction). Real fix is
  upstream: a cuda13/py312-compatible pycolmap (a conda Qt6 build, or a PyPI wheel
  whose bundled libs don't crash in a conda env).

  **ai-demos faiss side effect (handled):** publishing the cuda13 `libfaiss` to
  ai-demos makes pixi's strict channel priority route *all* `libfaiss` through
  ai-demos, which would break **mast3r-slam** (it needs the `cpu_openblas`
  libfaiss). pixi merges workspace channels ahead of per-feature ones, and
  `channel-priority = "disabled"` is not honored for the pixi-build envs
  (mast3r-slam/dpvo), so per-feature ordering can't fix it. Fix: ai-demos also
  ships a **libfaiss 1.10.0 cpu_openblas** build, so both pysfm (cuda13) and
  mast3r-slam (cpu_openblas) resolve from ai-demos under strict priority.
  mast3r-slam re-validated on the cpu_openblas build: `faiss` import + search OK,
  27 tests pass.

### Two plan premises the hardware disproved
1. **The mamma realtime gate is NOT fixed by the migration.** §0 hypothesised the
   torchvision/torch mismatch slowed the SAM2 `track` stage and that torch 2.12 would
   resolve it. Empirically, on the consistent torch-2.12/cuda13 stack `track` is
   **~87 ms/call → 7.7 ticks/s** (gate needs ≥15), essentially unchanged from the
   pre-migration ~72 ms / ~7 ticks/s. SDPA flash/cudnn backends are enabled and fast
   (0.1 ms) and the GPU sits at ~61 % util, so it is not an attention-backend issue —
   the track stage is simply not sped up by the torch bump. Reaching realtime needs
   separate perf work (track-stage profiling / `torch.compile` / overhead reduction),
   out of this migration's scope.
2. **In-process torchcodec decode is NOT ~400 cam-fps.** §3 justified deleting the
   mp_decode hack on torchcodec 0.13+'s NVDEC-cache fix giving ~400 cam-fps in-process.
   Measured on 4×4K HEVC→CUDA with torchcodec 0.14: **114 cam-fps** (worse than the
   "~140 broken" baseline, far from ~400). Decode is not the realtime bottleneck here
   (track is), but the removal's stated justification does not hold on this hardware.
