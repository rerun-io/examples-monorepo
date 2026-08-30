# Phase 1 — Fork + pixify

Goal: `git clone <fork> && cd <fork> && pixi run demo` runs upstream inference on one real sample and
shows it in Rerun, on the monorepo's pins, with upstream code untouched.

## Layout of the fork

- `main` = upstream commit, frozen (record the SHA in NOTES.md). Never push to it.
- `pixi` = default branch. Change set is limited to: `pixi.toml`, `pixi.lock`, `.gitignore`,
  `demo_rerun.py`, `NOTES.md`, one README section "Run with pixi". Any upstream `.py` edit needs a
  reason under "Upstream edits" in NOTES.md (target: none; `git diff main -- '*.py'` shows only the demo).

```bash
GH_TOKEN=$(gh auth token --user pablovela5620) gh repo fork <org>/<repo> --clone=false
git clone git@github.com:pablovela5620/<repo> ~/0Dev/forks/<repo>   # personal SSH identity
git -C ~/0Dev/forks/<repo> checkout main && git -C ~/0Dev/forks/<repo> reset --hard <upstream-sha>   # main == the frozen SHA even if upstream moved
git -C ~/0Dev/forks/<repo> checkout -b pixi
# after pushing: make pixi the default branch
GH_TOKEN=$(gh auth token --user pablovela5620) gh repo edit pablovela5620/<repo> --default-branch pixi
```

## pixi.toml

- Channels `conda-forge` only; platforms `linux-64` (+ `osx-arm64` only if it actually solves).
- Copy pins from the monorepo `pixi.toml` (`common`/`monoprior` features): python, cuda-version, pytorch-gpu,
  torchvision, rerun-sdk, timm, einops, tyro, numpy, opencv. Keep ranges, not exact pins.
- `simplecv` as a git PyPI dep on monorepo `main` — it gives `RerunTyroConfig`, camera dataclasses, and
  `log_rig_static`. It does not carry its runtime deps: add `av`, `pyarrow`, `einops`, and pin
  `typing-extensions = ">=4.1,<4.16"` (pyserde<0.32).
- Packages missing from conda-forge go to `[pypi-dependencies]`. Drop upstream's training/export/profiling deps.
- Tasks (single-line `&&` chains only — pixi collapses multiline `cmd`):
  - `_download-checkpoints`, `_download-data`: guarded by shell `test -f`/`test -d` (pixi `inputs`/`outputs`
    skip gitignored files); use `hf download` (never `huggingface-cli`).
  - `demo` (depends-on both downloads) → `python demo_rerun.py`; `demo-upstream` → upstream's own script.
  - `eval` when upstream has an eval script, on the single hosted sample.
- Verify: `pixi lock --check` clean; fresh clone → `pixi run demo` timing in NOTES.md.

## Weights and sample data

- Weights: official HF repo if one exists (pin revision SHA); else mirror the release files unmodified to
  `pablovela5620/<name>` with the upstream LICENSE in the repo. Non-commercial licenses: say so in the card.
- Sample: one scene with ground truth, in *upstream's exact dataset layout* so upstream's eval script works
  unchanged. Stereo: ETH3D two-view `playground_1l` already lives at HF `pablovela5620/monoprior-example`
  (`stereo/eth3d`, Middlebury-v3 `calib.txt` with baseline in mm).

## demo_rerun.py

- tyro `Config` with a nested `rr_config: RerunTyroConfig` (spawn/save/connect/headless for free).
- Log an exoego:v2 rig via `simplecv.rerun_rig_logger.log_rig_static` (`/world/rig_00/cam_NN`,
  `Transform3D(from_parent=True)`, `Pinhole`, image + depth under `pinhole/`). Camera params are simplecv
  `PinholeParameters` (`Intrinsics.from_k_matrix`, `Extrinsics`).
- Depth from disparity: `fx * B / (d + doffs)`; mask `depth > max_depth_m` (sky sub-pixel disparities
  streak the cloud). Keep 2D-only entities (disparity/GT/error) out of the 3D view's contents.
- Print the metric against GT in-script (EPE, bad-N) and check it equals upstream's eval to the printed digit.

## NOTES.md (mandatory — it is what feeds this skill back)

Sections: Decisions, Commands (fork, sample prep, upload), Upstream edits, Gotchas found,
Reproduction table (ours vs paper), fresh-clone timing.
