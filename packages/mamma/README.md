# mamma — streaming multiview SMPL-X body capture

A causal streaming port of the MAMMA multiview body-capture pipeline: NVDEC
decode → GPU resize → SAM2 person tracking → MammaNet dense landmarks → GPU
triangulation → sliding-window SMPL-X fitting → Rerun, in one resident process
with **no disk writes anywhere in the loop**. The original 5-subprocess offline
DAG took 240 s for the 12.1 s / 4-camera reference clip; this runs it in
~14.5 s at full per-frame mask quality (~12 s with `--tracker.track-stride 4`).

## Quickstart

Data and weights are license-gated (MAMMA registration, SMPL-X) and live in the
**private** HF dataset repo `pablovela5620/mamma-streaming-data` — authenticate
with `hf auth login` first. Downloads happen automatically via task
dependencies.

```bash
# from the repo root
pixi run -e mamma --frozen mamma-demo-crossing-arms      # full pipeline -> Rerun viewer
pixi run -e mamma --frozen mamma-validate-golden         # 3D accuracy gate vs frozen original output
pixi run -e mamma --frozen mamma-benchmark               # wall-time gate (15 s = >=80% realtime)
pixi run -e mamma --frozen mamma-goal-check              # all five acceptance clauses
```

Optional TensorRT engine for MammaNet (used automatically by goal-check when
present): `pixi run -e mamma --frozen python packages/mamma/tools/build_trt_engine.py`.

## Acceptance gates (`mamma-goal-check`, all PASS)

| Clause | Criterion |
|---|---|
| golden | MPJPE/PVE vs the frozen original pipeline ≤ 30 mm on crossing_arms frames 60:90 (currently 23.3 / 21.4 mm) |
| realtime | 363-frame 4-cam clip ≤ 15 s wall incl. Rerun logging (currently ~14.8 s) |
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
  engine/        per-tick pipeline, multiprocess decode, profiler
  viz/           Rerun stream logger (NVENC H.264 -> VideoStream) + blueprint
tools/           tyro CLIs: demos, validate_golden, benchmark, goal_check, profiler
docs/perf-plan.md           full 240 s -> 14.5 s optimization campaign log
implementation-notes.html   running decision log with viewer-validation evidence
```

## What's next

- **Delete `engine/mp_decode.py` when conda-forge ships torchcodec ≥ 0.11.**
  torchcodec 0.10 caches a single NVDEC decoder instance per (codec,
  resolution), so four identical in-process streams thrash with ~60 ms re-inits
  (~140 cam-fps ceiling). `mp_decode` works around this with one persistent
  spawn worker per camera (~400 cam-fps) at the cost of ~150 lines of subtle
  multiprocessing (pinned-CPU transport, spawn re-import hazards, worker
  lifecycle). Upstream PR #1232 fixes the cache in 0.11 — once it lands in
  conda-forge, switch `StreamingPipeline(use_mp_decode=False)` on and remove
  the module plus its pipeline branch.
- **True realtime at full quality:** ~2.4 s remain between stride-1 (14.5 s)
  and the 12.1 s clip. Sized tranche in `docs/perf-plan.md`: TensorRT engine
  for the EfficientTAM image encoder (same recipe as MammaNet — the encoder
  dominates per-tick cost at stride 1) + columnar `rr.log` batching / worker
  consolidation. Fallback: collapse-triggered re-track (stride 4 + densify when
  mask area drops >40% vs running median).
- **Multi-person hardening:** the batched steady-state tracker path assumes a
  single subject (`obj_id=0`); Assembly101 scenes have several people.
- **Live sources:** the pipeline is causal end-to-end; an RTSP/webcam adapter
  producing `MultiViewSequence`-shaped input would make it a live capture demo.
