# Worked example — Fast-FoundationStereo (2026-08-30), the skill's first run

Second stereo model, done *through* this skill by Codex (phase 1, PRs 1–3) with Claude reviewing and validating.
Fork: `pablovela5620/Fast-FoundationStereo` (`master` = upstream a290ba0 frozen, `pixi` default).
Monorepo stack `fast-foundationstereo/1-vendor → 2-predictor → 3-typed` on top of the skill PR (#160).
No `4-app`/`5-catalog`: the stereo family already had both, and they gained the model through the registry
(`--predictor-name FastFoundationStereoPredictor`, app dropdown).

## Scope answers that mattered

- License: NVIDIA Source Code License (code, research only) **and** NVIDIA Open Model Agreement (weights) — two licenses.
- Weights: official HF `nvidia/c-fast-foundationstereo` (rev `9b446878…`), no safetensors, no personal mirror needed.
  One pickled whole module + `cfg.yaml`; the Drive trio (`23-36-37` …) is not on the Hub and was left out.
- Inference subset: `core/{foundation_stereo,extractor,update,geometry,submodule,distill_block}.py` +
  `core/utils/utils.py` (InputPadder **and** the bilinear samplers `geometry.py` needs — not only the padder).
- Hard deps: `xformers` is in the README but nothing in `core/` imports it; triton GWC kernel has a torch fallback;
  `open3d`/`turbojpeg` only through `Utils.py`/`frame_utils` → not imported.
- No eval script → ETH3D `playground_1l` number recorded as a baseline.

## Numbers

ETH3D `playground_1l` (non-occluded, gt < 416): EPE 0.241 px, bad1 **0.48 %** (LAS2-H: 1.12 %, LAS2-M: 2.24 %).
5090 fp16 autocast, 490×941 (padded 512×960): ~83–95 ms/frame — ~10× slower than LAS2-M at this size.
Triton vs torch GWC volume: identical metrics.

## Gotchas hit

- Pickled `args` lacks `normalize` (checkpoint predates the code); `True` measured → 0.48 %, `False` → 45 %.
- The pickle references `core.*` **and** `foundation_stereo_ori.*` class paths, plus `omegaconf` (new dep).
- The checkpoint is NAS-pruned: 210 tensors only in the pickle, 54 shape changes, `distill_block.ForwardHelper`
  wrappers — `cfg.yaml` cannot rebuild it, so the loader unpickles + deep-copies + strict-reloads and the typed
  fork must not rename modules/classes/parameters. Equivalence = fast config-built upstream-vs-owned (CPU,
  bit-identical) + slow real-checkpoint unpickled onto both packages (deterministic cuDNN, autocast off:
  CUDA `ConvTranspose2d` otherwise drifts ~2e-5 between identical runs) + triton/torch parity with tolerance.
- beartype claw cannot decorate a module-level `triton.autotune` object → build the kernel inside a factory;
  check with `warnings.simplefilter('error', BeartypeClawDecorWarning)`, not `-W error::UserWarning`
  (that also trips the required PEP 613 `TypeAlias` warnings).
- `low_memory=False` is the *required* 2D sampler path; only the `low_memory=True` 1D/chunked branch was dead.
- Codex sandbox: CUDA virtual package hidden (`CONDA_OVERRIDE_CUDA=13.0`), default pixi cache read-only
  (`PIXI_CACHE_DIR=/tmp/...`); GPU runs verified outside the sandbox.
- Running pytest from the repo root collects other packages whose envs are absent — run from `packages/monoprior`.
- The catalog server (`:51235`) keeps registrations in memory: after a restart, `LookupError: No dataset found
  with name 'robocap'` → `python /mnt/nas/datasets/robocap/rrd/reregister.py` first.
- A background catalog run launched from the agent session died with the session and left a truncated `.rrd`
  (video relay covers the whole clip, so the timeline *looks* complete); run long jobs in tmux with `python -u`.

## Skill fixes this run produced

Five gates not four; HF revision + SHA-256; weights vs code license; upstream entry-point scoping in phase 0;
PR-1 "import paths only"; equivalence vs kernel parity; registry with differing constructors; `PYREFLY_TARGET`
(which also exposed that the LAS2 typed fork was never typechecked — fixed in #154); frozen default branch may be
`master`; fresh-clone timing definition; interactive upstream demos; pickled NAS architectures.
