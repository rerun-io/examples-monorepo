# CUDA 13 / torch 2.12 / torchcodec 0.13 — monorepo migration plan

**Status: PLAN ONLY — no dependency changes made.**

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
