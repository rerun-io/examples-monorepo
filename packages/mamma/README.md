# mamma — streaming multiview SMPL-X body capture

A causal streaming port of the MAMMA multiview body-capture pipeline: NVDEC
decode → GPU resize → SAM2 person tracking → MammaNet dense landmarks → GPU
triangulation → sliding-window SMPL-X fitting → Rerun, in one resident process
with **no disk writes anywhere in the loop**. The original 5-subprocess offline
DAG ran at ~12 s/frame (over an hour for this 4-camera clip); this runs the same
clip in seconds at full per-frame mask quality.

## Quickstart

Data and weights are license-gated (MAMMA registration, SMPL-X) and live in the
**private** HF dataset repo `pablovela5620/mamma-streaming-data` — authenticate
with `hf auth login` first. Downloads happen automatically via task
dependencies.

```bash
# from the repo root
pixi run -e mamma --frozen mamma-demo-crossing-arms      # full pipeline -> Rerun viewer
pixi run -e mamma --frozen mamma-validate-golden         # 3D accuracy gate vs frozen original output
pixi run -e mamma --frozen mamma-benchmark               # wall-time gate (wall <= 2x clip duration = >=50% realtime)
pixi run -e mamma --frozen mamma-goal-check              # all six acceptance clauses
```

Optional TensorRT engine for MammaNet (used automatically by goal-check when
present): `pixi run -e mamma --frozen python packages/mamma/tools/build_trt_engine.py`.

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

## Acceptance gates (`mamma-goal-check`, all PASS)

| Clause | Criterion |
|---|---|
| golden | MPJPE/PVE vs the frozen original pipeline ≤ 30 mm on crossing_arms frames 60:90 |
| dynamic | running_jumping per-frame PVE p95 AND max ≤ 30 mm vs the original DAG over all frames |
| realtime | both clips (crossing_arms + running_jumping): wall ≤ 2× clip duration (≥50% realtime) incl. Rerun logging |
| datasets | HOCap + Assembly101 ingest end-to-end → validated RRDs |
| no-writes | streaming loop creates no files (runtime check + static write-call audit) |
| hygiene | ruff + pyrefly + pytest clean in `mamma-dev` |

Defaults trade the last 2.5 s for mask quality: `track_stride=1` keeps masks
pixel-identical to the original tracker (stride 4 reaches 11.8 s but
transiently collapses masks to the head during fast motion — evidence in
`implementation-notes.html`).

## Layout

```
src/mamma/
  calibration/   camera contracts + loaders (NPZ meta format)
  datasets/      MultiViewSequence adapters: mamma NPZ, HOCap, Assembly101
  tracking/      YOLO bootstrap + CLIP/epipolar identity + streaming SAM2 fork
  landmarks/     MammaNet port (512 dense 2D landmarks) + TensorRT backend
  fitting/       GPU triangulation + CUDA-graphed sliding-window SMPL-X fitter
  engine/        per-tick streaming pipeline, in-process torchcodec decode, profiler
  viz/           Rerun stream logger (NVENC H.264 -> VideoStream) + blueprint
tools/           tyro CLIs: demos, validate_golden, benchmark, goal_check, profiler
implementation-notes.html   running decision log with viewer-validation evidence
```

## What's next

- **True realtime at full quality:** ~2.4 s remain between stride-1 (14.5 s)
  and the 12.1 s clip. Highest-leverage tranche: a TensorRT engine for the
  EfficientTAM image encoder (same recipe as MammaNet — the encoder dominates
  per-tick cost at stride 1) + columnar `rr.log` batching / worker
  consolidation. Fallback: collapse-triggered re-track (stride 4 + densify when
  mask area drops >40% vs running median).
- **Multi-person hardening:** the batched steady-state tracker path assumes a
  single subject (`obj_id=0`); Assembly101 scenes have several people.
- **Live sources:** the pipeline is causal end-to-end; an RTSP/webcam adapter
  producing `MultiViewSequence`-shaped input would make it a live capture demo.
