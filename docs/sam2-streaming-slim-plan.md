# sam2-streaming slim-down — Hypothesis-gated cleanup plan

> **Progress (in flight).** Harness built and validated; oracle recorded at 1000
> examples/seam from the full code. Executed and verified equivalence-exact
> (replay green) after each:
> - **Step 0** — `tests/` harness (`equiv_util` + encode_memory / streaming /
>   gpu_smoke seams). Oracle corpus: encode_memory 1000×2, streaming 1000×2,
>   gpu_smoke 1×2.
> - **Step 1** — deleted 6 dead files (benchmark, automatic_mask_generator,
>   sam2_image_predictor, utils/amg, sam2_video_predictor, _legacy).
> - **Step 2** — removed the 8 dead builders + HF map from `build_sam.py`.
> - **Step 3 (partial)** — removed `SAM2GenericVideoPredictorVOS` + `vos_optimized`,
>   and the training-only `Hiera.get_layer_id`/`get_num_layers`.
>
> **Equivalence PROVEN at the full ≥1000-example bar for Steps 1–3** (6 passed,
> 11:52: encode_memory 1000×2 + streaming 1000×2 bitwise-exact, gpu_smoke
> allclose); the hieradet removal re-verified at limit-150. Net so far:
> **−4,138 LOC** (sam2/ python 10,211 → 6,074), all 11 model variants still build.
> **Remaining:** rest of Step 3 (reverse_tracking, create_memory,
> soft_no_obj_ptr, pred_obj_scores fallbacks, get_layer_id/num_layers,
> get_sdpa_settings, max_sprinkle_area, point/mask-prompt paths), Step 4 inlining,
> `mamma-goal-check` (golden + 42-scene — the long pole), lint/typecheck/deadcode,
> retire scaffolding + update VENDOR.md.



Strip the vendored `packages/sam2-streaming` fork to a **streaming-inference-only**
package that still loads **every** model variant, using a Hypothesis differential
test suite to prove the slim version is functionally equivalent to the current one.

This is a one-time cleanup landed **inside PR #48** (full sam2 never reaches `main`).

## Decisions (settled)

| # | Decision | Choice |
|---|---|---|
| 1 | End state | Slim is the product. The current ("full") fork is an oracle only, deleted once green. One maintained fork. |
| 2 | Equivalence bar | **Allow refactoring → `allclose`** (not delete-only/bitwise). Hypothesis verifies refactors preserve behavior. |
| 3 | Test lifetime | **One-shot scaffolding.** Equivalence tests + oracle removed before the PR finalizes. mamma's golden + 42-scene gate is the permanent net. |
| 4 | Refactor depth | **Structural slim, upstream style.** Delete dead files/branches, inline, collapse, prune unused params/knobs. No jaxtyping/beartype re-annotation. Stays diff-able vs upstream. |
| 5 | Variant scope | Keep **all** model variants (sam2 / sam2.1 / efficienttam configs + `hiera` + `vitdet` backbones) loadable via `build_sam2_generic_video_predictor`. Delete the non-streaming predictor **types** + their builders. |
| 6 | Sequencing | **Block #48 — slim first.** Oracle = today's branch state, frozen. |
| 7 | Coverage shape | Per-seam Hypothesis differential (workhorse) + 1–2 real-frame end-to-end smoke tests. |
| 8 | Platform | Seam tests on **CPU + deterministic algorithms** (near-bitwise, fast, broad). Real-frame smoke on **GPU** (exercises `_C.so` + production numerics, `allclose`). |

## Why Hypothesis earns its keep here

Whole-file deletion is cheap to verify (import-time failure catches a reachable
module). The risk lives in the **in-file dead branches** we're removing inside
*kept* files — `reverse_tracking`, `create_memory=False`, the VOS subclass,
`soft_no_obj_ptr`, the `pred_obj_scores`/`pred_obj_scores_mlp` decoder fallbacks,
etc. Cutting one of those wrong silently changes outputs. Per-seam differential
testing with broad synthetic inputs is exactly the tool that catches it, and the
stateful memory-bank logic is a textbook property-based-testing target.

## The oracle: `sam2_ref` live mirror (primary)

Both versions must be importable in one pytest process for live differential
comparison with fresh Hypothesis examples each run. Same-name install is
impossible, so generate a **renamed frozen mirror** of the pre-slim tree:

```
scripts/build_sam2_ref.sh   (scaffolding, deleted at the end)
  1. cp -r packages/sam2-streaming /tmp/sam2-ref-build
  2. git -C ... stash-free: copy is taken from the pre-slim commit (the oracle)
  3. mv .../sam2  .../sam2_ref
  4. rewrite module paths only — NOT class names / "sam2.1" / "SAM2*":
       grep -rl --include='*.py' --include='*.yaml' '\bsam2\.' | xargs sed -i 's/\bsam2\./sam2_ref./g'
     this catches  `from sam2.x import`, `import sam2.x`,
     and hydra `_target_: sam2.modeling.…` strings in the YAMLs
  5. fix the builder override in build_sam.py:150,169 (++model._target_=sam2.… → sam2_ref.…)
     — already covered by step 4's sed, verify it
  6. set distribution name `sam2_ref` in setup.py/pyproject, install editable:
       sam2_ref = { path = "<mirror>", editable = true }  under [feature.mamma-dev]
```

Validation gate before any cleanup: `import sam2_ref; build_sam2_generic_video_predictor(...)`
must build **all** variants (hiera-s@1024, efficienttam-ti@512), proving the
mirror + hydra config discovery work under the new module name.

**Fallback (if the rename/hydra-config-module proves fiddly):** capture/replay —
run the seam corpus once under the frozen oracle, serialize `(input, output)` to
`.pt`, replay against slim. Same differential guarantee, one version loaded at a
time, but no fresh examples on the slim side. Decide at Step 0; don't sink >~1h
into the rename before falling back.

## Determinism + tolerance

**CPU seam tests** (the workhorse — clean signal):
```python
torch.use_deterministic_algorithms(True)
torch.manual_seed(seed)           # per Hypothesis example
model_full.eval(); model_slim.eval()
# methods are already @torch.inference_mode(); dtype float32; device cpu
```
Build `full` and `slim` from the **same checkpoint + same config**. Start the bar
at `torch.equal` (exact) — pure deletions must be bitwise; if structural inlining
reorders float ops, relax that seam to `allclose(rtol=1e-5, atol=1e-6)` and log
which seam needed it (a signal of where math moved).

**GPU real-frame smoke**:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False    # tighten, then small atol
# bar: torch.allclose(out_full, out_slim, rtol=1e-3, atol=1e-3)
```
The `_C.so` connected-components kernel is deterministic given its input; the
tolerance absorbs upstream CUDA float variation, not the kernel.

## Seam test list (CPU, synthetic inputs)

Shapes are grounded in `sam2/modeling/sam2_generic.py` and `memory.py`.

| Seam | Synthesized input | Compare |
|---|---|---|
| `SAM2*ObjectMemoryBank` op-sequences | random sequence of `try_add_memories`/`select_memories`/`prune_memories` over frame indices + `is_prompt` flags (use Hypothesis `RuleBasedStateMachine`) | selected `ObjectMemory` stacks + all `count_*` |
| `encode_memory(img_embeddings, masks_logits, obj_score_logits, is_prompt)` | `img_embeddings`: list of `num_feature_levels` `(B,C,H,W)`; `masks_logits` `(B,1,H,W)`; `obj_score_logits` `(B,1)`; `is_prompt` `(B,)` bool | `(memory_embeddings, memory_pos_embeddings)` |
| `condition_image_embeddings_on_memories(...)` | synthetic embeddings + `ObjectMemory(mem_emb, mem_pos, ptr)` | conditioned embeddings |
| `generate_masks(...)` | conditioned embeddings + prompt embeddings, `multimask_output ∈ {True,False}` | `SAM2Result` (`masks_logits`, `ious`, `obj_ptrs`, `obj_score_logits`) |
| `empty_prompt_embeddings` (property) | — | tensors |
| `SAM2Result.cat` / `select_best` | random list of `SAM2Result` with **mixed** mask-hypothesis counts (the recent multi-person fix) | concatenated / selected result |

The heavy backbone seam (`encode_image`) is **not** synthesized on CPU — it's
covered by the GPU real-frame smoke (run a real frame through both, diff the
multi-level embeddings), since it's identical code and expensive.

## Execution: delete-and-verify, gate after each step

Codex confirmed a deletion-order dependency — respect it.

- **Step 0 — Harness.** Build `sam2_ref`, the seam suite, determinism config, GPU
  smoke. Prove `sam2_ref ≡ sam2` GREEN with both unmodified (validates the harness
  itself, not the cleanup).
- **Step 1 — Whole dead files**, in order: `benchmark.py` → `sam2_video_predictor.py`
  → `sam2_video_predictor_legacy.py` → `automatic_mask_generator.py` →
  `sam2_image_predictor.py` → `utils/amg.py`. Gate after each.
- **Step 2 — Dead builders** in `build_sam.py`: HF map + `*_hf`, `build_sam2`,
  `build_sam2_generic`, `build_sam2_video_predictor`, the `vos_optimized` branch.
  Gate.
- **Step 3 — In-file dead branches** (high-confidence first): `SAM2GenericVideoPredictorVOS`,
  `reverse_tracking`, `create_memory=False`, `soft_no_obj_ptr`,
  `pred_obj_scores`/`pred_obj_scores_mlp` decoder fallbacks,
  `get_layer_id`/`get_num_layers`, `get_sdpa_settings`, `max_sprinkle_area`,
  point/mask prompt paths, legacy `track_step` stack. Gate after each cluster.
- **Step 4 — Structural inlining/collapse** for YAGNI clarity. Gate.
- **Step 5 — Full acceptance:** seam suite (N≥1000 examples/seam) + GPU smoke +
  mamma `mamma-goal-check` (golden + 42-scene) unchanged + ruff/pyrefly/vulture green.
- **Step 6 — Retire scaffolding:** delete `sam2_ref`, the equivalence tests, the
  build script, and the `sam2_ref` pixi dep. Update `VENDOR.md`: it is no longer
  "Code modifications: none" — record the slim divergence, the pinned upstream
  commit it derives from, and the removed surface.

## Keep-list (do NOT delete — Codex-flagged)

- All backbones (`hiera`, `vitdet`, `image_encoder`, `utils`) — reached by hydra
  `_target_` for the model zoo.
- `_C.so` + `csrc/connected_components.cu` — on the `++model.max_hole_area=8` path.
- `SAM2ObjectMemoryBank` import in `sam2_generic_video_predictor.py` — module-level
  default factory, even though mamma passes `SAM2ForgetfulObjectMemoryBank`.
- All model config YAMLs (sam2 / sam2.1 / efficienttam), incl. the EfficientTAM
  configs whose `_target_` is overridden by the builder.
- The dangling `configs/sam2.1_training/` YAML — harmless, leave it.

## Acceptance criteria

1. Every seam Hypothesis test passes at its tolerance, N≥1000 examples/seam, 0 failures.
2. GPU real-frame smoke `allclose` passes for hiera-s@1024 and efficienttam-ti@512.
3. `pixi run -e mamma --frozen mamma-goal-check` exits 0 (golden + 42-scene unchanged).
4. `import sam2; build_sam2_generic_video_predictor(...)` builds all variants.
5. ruff + pyrefly + vulture green across the monorepo.
6. Scaffolding removed; `VENDOR.md` updated.
