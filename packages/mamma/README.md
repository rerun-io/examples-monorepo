# mamma — streaming multiview SMPL-X body capture

A causal streaming port of the MAMMA multiview body-capture pipeline as one dagkit graph
(`src/mamma/graphs/capture.py`): NVDEC decode → GPU resize → SAM2 person tracking → MammaNet dense landmarks →
GPU triangulation → sliding-window SMPL-X fitting → Rerun (and, on request, a `mamma_fit` catalog layer).

## Quickstart

Data and weights are license-gated (MAMMA registration, SMPL-X) and live in the
**private** HF dataset repo `pablovela5620/mamma-streaming-data` — authenticate
with `hf auth login` first. Downloads happen automatically via task
dependencies.

```bash
# from the repo root
pixi run -e mamma --frozen mamma-capture                 # a catalog segment (dataforge-mamma-eval) -> Rerun viewer
pixi run -e mamma --frozen mamma-demo-crossing-arms      # the crossing_arms files -> Rerun viewer
pixi run -e mamma --frozen verify                        # graph tier: MPJPE / PVE / mask IoU vs mosh_gt, banded (VERIFY.toml)
```

`tools/apps/capture.py --help` lists the App options (`--preset quality|fast`, `--subjects`, `--output-dir` for the
layer); presets are node parameters in `profiles/x86_64-cuda.toml`. Optional TensorRT engine for MammaNet (used
when present): `pixi run -e mamma --frozen python packages/mamma/tools/build_trt_engine.py`.

## Downloading the MAMMA dataset

`mamma-download-dataset` fetches the full public MAMMA corpus (~6.85 TB): the
four MPI capture collections (eval, dance, multi-people, iPhone — all videos at
H.265 CRF 16, no previews) plus the MammaSyn synthetic set from its Hugging
Face mirror (6.63 TB, much faster than the MPI server).

```bash
# register at https://mamma.is.tue.mpg.de/ first, then:
export MAMMA_USERNAME=... MAMMA_PASSWORD=...
pixi run -e mamma --frozen mamma-download-dataset --output-root /mnt/nas/datasets/mamma
```

Scope with `--no-eval`, `--no-dance`, `--no-multi-people`, `--no-iphone`,
`--no-synthetic`. Valid existing files are skipped, so re-running is a cheap
verification sweep that retries only missing or invalid files — safe to
interrupt and resume. For multi-day runs, launch it inside tmux (and make sure
`loginctl enable-linger` is set so logind doesn't reap the session on logout).

Implementation notes, learned the hard way:

- MPI transfers shell out to **wget**: `download.is.tue.mpg.de` serves its HTML
  landing page to every Python TLS client, whatever the headers.
- The Hugging Face phase swaps in `SoftFileLock` before importing
  `huggingface_hub` — flock hangs forever on NFSv3 mounts with
  `local_lock=none` — and sweeps stale locks left by killed runs.
- File manifests are vendored in `src/mamma/apis/download_manifest.py`,
  mirroring the upstream `data/download_mamma_*.sh` arrays; dance person
  counts derive from the sequence naming convention (one 5-digit subject id
  per dancer).

## Layout

```
src/mamma/
  types.py     edge Types (tracks, landmarks, SMPL-X fits, 3D landmarks) and their Rerun mapping
  models/      YOLO12x, open_clip, SAM2 predictor, MammaNet (+ TensorRT, posekit role)
  ops/         triangulation policy, identity scoring, SMPL-X body + window fit, skeleton regressor
  nodes/       track, landmarks, fit
  sources/     the MAMMA-format file source (catalog segments use simplecv's CatalogVideoSource)
  graphs/      capture.py
  apis/        capture App (Config, build), blueprint, downloads, TensorRT build, checkpoint conversion
  verify.py    graph-tier check (VERIFY.toml)
tools/         thin tyro shims over apis/
```
