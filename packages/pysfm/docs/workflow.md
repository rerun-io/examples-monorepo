# Multicam Workflow

This is the compact source of truth for the multicam dataset.

- `videos/` is the source of truth
- `images/` contains synchronized extracted frames
- `colmap/` contains derived reconstruction state

The important invariant is simple: the same filename across `cam1`, `cam2`,
and `cam3` must be the same timestamp.

## 1. Layout

```text
multicam/
  videos/
    <batch>/<capture>/rig1/cam1.mp4
    <batch>/<capture>/rig1/cam2.mp4
    <batch>/<capture>/rig1/cam3.mp4

  images/
    <batch>/<capture>/<count>/
      rig1/
        cam1/image0001.jpg
        cam2/image0001.jpg
        cam3/image0001.jpg
      manifest.json
      rig_config.json

  colmap/
    <batch>/<capture>/<count>/
      rig/
        db/{features.db,no_rig.db,rig.db}
        sparse/no_rig/0
        sparse/0
        logs/
      mono/
        cam1/
          db/database.db
          sparse/0
          logs/
```

For cross-count hybrid runs, use a sibling capture-level run root such as:

```text
colmap/<batch>/<capture>/0050seed-0100final-.../{seed0050,final0100}
```

Rules:

- keep the capture name in the path
- use zero-padded counts: `0050`, `0100`, `0200`
- use one-based image names: `image0001.jpg`, `image0002.jpg`, ...
- choose frame indices once per count and reuse them across all cameras
- keep reconstruction outputs out of `images/`

## 2. Pipeline

```text
videos/ ──> synchronized extraction ──> images/<count>/rig1/cam*/
                                              │
                                              ├──> mono baseline
                                              │    feature_extractor
                                              │    sequential_matcher
                                              │    mapper/global_mapper
                                              │
                                              └──> rig reconstruction
                                                   feature_extractor
                                                   sequential_matcher (no_rig)
                                                   mapper/global_mapper (no_rig)
                                                   rig_configurator
                                                   sequential_matcher (rig)
                                                   mapper/global_mapper (rig)
```

Rig flow at a glance:

```text
┌───────────────┐      ┌────────────────────┐      ┌──────────────────┐
│ images/<count>│ ───> │ no_rig bootstrap   │ ───> │ rig_configurator │
└───────────────┘      └────────────────────┘      └──────────────────┘
                                                            │
                                                            ▼
                         ┌──────────────────────────────────────────────┐
                         │ final rig-aware matching + mapping           │
                         │ output: colmap/.../rig/sparse/0              │
                         └──────────────────────────────────────────────┘
```

## 3. Defaults

Current working defaults:

- feature extractor: `ALIKED_N16ROT`
- matcher: `ALIKED_LIGHTGLUE`
- matcher family: `sequential_matcher`
- `SequentialMatching.quadratic_overlap = 0`
- `FeatureMatching.skip_image_pairs_in_same_frame = 0`
- `SequentialMatching.expand_rig_images = 1` on the rig-aware pass
- `ba_refine_sensor_from_rig = 1` on the final rig-aware solve

Run modes:

- fast smoke test: `0050`, `global_mapper`, `overlap=5`
- fast full run: `0050` seed with `global_mapper`, `0100` final with
  `global_mapper`
- robust full run: `0050` seed with incremental `mapper`, `0100` final with
  `global_mapper`
- use the robust full run as the default production path

GPU policy:

```bash
COLMAP="CONDA_OVERRIDE_CUDA=12.9 pixi run --frozen -e pysfm colmap"
```

- feature extraction uses GPU
- matching uses GPU
- BA is requested on GPU, but this build falls back to CPU because Ceres was
  built without `cuDSS`

## 4. Default Rig Commands

This is the preferred production workflow:

- seed on `0050` with incremental `mapper`
- final solve on `0100` with `global_mapper`

```bash
COLMAP="CONDA_OVERRIDE_CUDA=12.9 pixi run --frozen -e pysfm colmap"
SEED_IMAGE_SET="/home/pablo/0Dev/data/colmap-data/multicam/images/<batch>/<capture>/0050"
FINAL_IMAGE_SET="/home/pablo/0Dev/data/colmap-data/multicam/images/<batch>/<capture>/0100"
RUN_ROOT="/home/pablo/0Dev/data/colmap-data/multicam/colmap/<batch>/<capture>/0050seed-0100final"
SEED_RUN="$RUN_ROOT/seed0050"
FINAL_RUN="$RUN_ROOT/final0100"

# Prepare output folders.
mkdir -p "$SEED_RUN/db" "$SEED_RUN/sparse/no_rig" "$SEED_RUN/logs"
mkdir -p "$FINAL_RUN/db" "$FINAL_RUN/sparse" "$FINAL_RUN/logs"

# Extract seed features once for all cameras.
$COLMAP feature_extractor \
  --image_path "$SEED_IMAGE_SET" \
  --database_path "$SEED_RUN/db/features.db" \
  --ImageReader.single_camera_per_folder 1 \
  --FeatureExtraction.type ALIKED_N16ROT \
  --FeatureExtraction.use_gpu 1 \
  --FeatureExtraction.gpu_index 0 \
  > "$SEED_RUN/logs/01_feature_extractor.log" 2>&1

# Build the `0050` no-rig seed.
cp "$SEED_RUN/db/features.db" "$SEED_RUN/db/no_rig.db"
$COLMAP sequential_matcher \
  --database_path "$SEED_RUN/db/no_rig.db" \
  --FeatureMatching.type ALIKED_LIGHTGLUE \
  --FeatureMatching.use_gpu 1 \
  --FeatureMatching.gpu_index 0 \
  --SequentialMatching.overlap 5 \
  --SequentialMatching.quadratic_overlap 0 \
  > "$SEED_RUN/logs/02_sequential_matcher_no_rig.log" 2>&1

$COLMAP mapper \
  --image_path "$SEED_IMAGE_SET" \
  --database_path "$SEED_RUN/db/no_rig.db" \
  --output_path "$SEED_RUN/sparse/no_rig" \
  --Mapper.multiple_models 0 \
  --Mapper.ba_use_gpu 1 \
  --Mapper.ba_gpu_index 0 \
  > "$SEED_RUN/logs/03_mapper_no_rig.log" 2>&1

$COLMAP model_analyzer \
  --path "$SEED_RUN/sparse/no_rig/0" \
  > "$SEED_RUN/logs/04_model_analyzer_no_rig.log" 2>&1

# Extract final features for `0100`.
$COLMAP feature_extractor \
  --image_path "$FINAL_IMAGE_SET" \
  --database_path "$FINAL_RUN/db/features.db" \
  --ImageReader.single_camera_per_folder 1 \
  --FeatureExtraction.type ALIKED_N16ROT \
  --FeatureExtraction.use_gpu 1 \
  --FeatureExtraction.gpu_index 0 \
  > "$FINAL_RUN/logs/01_feature_extractor.log" 2>&1

# Estimate the rig on the `0100` DB from the `0050` seed.
cp "$FINAL_RUN/db/features.db" "$FINAL_RUN/db/rig.db"
$COLMAP rig_configurator \
  --database_path "$FINAL_RUN/db/rig.db" \
  --input_path "$SEED_RUN/sparse/no_rig/0" \
  --rig_config_path "$FINAL_IMAGE_SET/rig_config.json" \
  > "$FINAL_RUN/logs/02_rig_configurator.log" 2>&1

# Rerun matching with rig-aware expansion on `0100`.
$COLMAP sequential_matcher \
  --database_path "$FINAL_RUN/db/rig.db" \
  --FeatureMatching.type ALIKED_LIGHTGLUE \
  --FeatureMatching.use_gpu 1 \
  --FeatureMatching.gpu_index 0 \
  --FeatureMatching.skip_image_pairs_in_same_frame 0 \
  --SequentialMatching.overlap 10 \
  --SequentialMatching.quadratic_overlap 0 \
  --SequentialMatching.expand_rig_images 1 \
  > "$FINAL_RUN/logs/03_sequential_matcher_rig.log" 2>&1

# Solve the final rig-aware reconstruction on `0100`.
$COLMAP global_mapper \
  --image_path "$FINAL_IMAGE_SET" \
  --database_path "$FINAL_RUN/db/rig.db" \
  --output_path "$FINAL_RUN/sparse" \
  --GlobalMapper.gp_use_gpu 1 \
  --GlobalMapper.gp_gpu_index 0 \
  --GlobalMapper.ba_ceres_use_gpu 1 \
  --GlobalMapper.ba_ceres_gpu_index 0 \
  --GlobalMapper.ba_refine_sensor_from_rig 1 \
  > "$FINAL_RUN/logs/04_global_mapper_rig.log" 2>&1

$COLMAP model_analyzer \
  --path "$FINAL_RUN/sparse/0" \
  > "$FINAL_RUN/logs/05_model_analyzer_rig.log" 2>&1
```

## 5. Brush

Brush expects one source tree that contains both:

- the COLMAP model files
- the referenced images

Our dataset keeps `images/` and `colmap/` separate, so `pysfm` stages a
symlinked Brush source tree:

```text
<work-root>/
  source/
    images -> /.../images/<batch>/<capture>/0100
    sparse/
      0 -> /.../colmap/.../rig/sparse/0
  runs/
    5k/
      train.log
      export_5000.ply
      eval_1000/
      ...
```

Brush is headless by default when a source path is provided. Use
`--with-viewer` only when you want the GUI.

Use `pixi run --frozen` for Brush runs unless the environment is actually being
changed. Otherwise Pixi spends a long time re-solving before every launch.

First kitchen-train run:

```bash
CONDA_OVERRIDE_CUDA=12.9 pixi run --frozen -e pysfm brush-train-kitchen-5k
```

Equivalent direct helper command from `packages/pysfm/`:

```bash
CONDA_OVERRIDE_CUDA=12.9 pixi run --frozen -e pysfm python tools/brush/train_colmap_brush.py \
  --images-dir /home/pablo/0Dev/data/colmap-data/multicam/images/6g-night-nov-29-2024/kitchen-nov-29-2024-train/0100 \
  --model-dir /home/pablo/0Dev/data/colmap-data/multicam/colmap/6g-night-nov-29-2024/kitchen-nov-29-2024-train/0100-aliked-lightglue-globalmapper-overlap10-timed-20260320/rig/sparse/0 \
  --work-root /home/pablo/0Dev/data/colmap-data/multicam/brush/6g-night-nov-29-2024/kitchen-nov-29-2024-train/0100-global-rig \
  --steps 5000 \
  --eval-split-every 10 \
  --eval-every 1000 \
  --export-every 5000 \
  --experiment-name 5k
```

View the exported splat:

```bash
CONDA_OVERRIDE_CUDA=12.9 pixi run -e pysfm brush_app \
  /home/pablo/0Dev/data/colmap-data/multicam/brush/6g-night-nov-29-2024/kitchen-nov-29-2024-train/0100-global-rig/runs/5k/export_5000.ply \
  --with-viewer
```

Notes:

- the packaged `brush` dependency is currently scoped to `linux-64` only
- the packaged executable name is `brush_app` even though upstream docs call it `brush`
- the shared `pysfm` env still needs `CONDA_OVERRIDE_CUDA=12.9` on this host
- `--eval-split-every 10` is image-level, not synchronized rig-frame-level

If you want the faster but less robust variant, replace the `0050` seed
`mapper` call with `global_mapper`. Keep the `0100` final stage as
`global_mapper`.

## 5. How `rig_configurator` Works

In the unknown-rig workflow, `rig_configurator` estimates fixed camera-to-rig
poses from the no-rig reconstruction.

It uses:

- the no-rig sparse model for per-image poses
- `rig_config.json` for sensor grouping and reference-sensor identity
- identical filenames across cameras to define synchronized rig frames

Conceptually, for a synchronized frame:

```text
sensor_from_ref = sensor_from_world * inverse(ref_from_world)
```

That gives a per-frame estimate of the non-reference sensor's pose relative to
the reference sensor. COLMAP computes these relative poses across many usable
frames and averages them into one fixed `sensor_from_rig` for each sensor.

Then the final rig-aware solver starts from those estimates and, because we use
`ba_refine_sensor_from_rig = 1`, it can refine them further.

The important consequence is:

- the no-rig bootstrap does not need to be the final map
- it only needs to be good enough to estimate a stable rig

## 6. Mono Baseline

Mono is just a sanity-check path. Use one camera folder directly as
`image_path`, for example:

```text
images/<batch>/<capture>/<count>/rig1/cam1/
```

Then run:

```text
feature_extractor -> sequential_matcher -> mapper/global_mapper
```

This is useful when you want a cheap baseline or to isolate whether a problem
is rig-specific.

## 7. What Actually Worked

Validated takeaways:

- the earlier SIFT-based setup was not reliable enough on this sampled data
- `ALIKED_N16ROT + ALIKED_LIGHTGLUE` avoided the need for custom pair files
- on `0050`, `overlap=5` was the best speed / completeness tradeoff
- `global seed -> global final` is the fast path
- `incremental seed -> global final` is the more robust path
- `global seed -> global final` worked on `5/6` captures
- `incremental seed -> global final` rescued `livingroom-nov-29-2024-eval`
- a bad-looking no-rig global map can still be good enough to seed a strong
  final rig
- the straight `0100` and hybrid `0050 -> 0100` runs produced essentially the
  same rig after removing global scale
- having the rig constraints matters much more than making the no-rig bootstrap
  look perfect

Representative reference numbers:

| Run | Result | Time |
| --- | --- | --- |
| `0050`, global, overlap `5` | rig `150 / 150` images | about `75s` |
| `0050 -> 0100`, fast hybrid | rig `300 / 300` images | about `191s` |
| `0050 -> 0100`, robust hybrid on a hard eval capture | rig `300 / 300` images | about `246s` |
| `0100`, global only | rig `300 / 300` images | about `244s` |

Practical interpretation:

- use the robust hybrid as the default production workflow
- use the fast hybrid for smoke tests and quick iteration
- if final `global_mapper` fails in rotation averaging, change the seed solver
  before changing the final stage
- judge `no_rig` mainly by whether it seeds a good final rig
- compare rig extrinsics up to scale; raw translation magnitudes can differ a
  lot across runs
- judge the final `rig` model by completeness and visual consistency
- compare `rig` and `no_rig` in the GUI when both register the same images

## 8. GUI

Pattern:

```bash
$COLMAP gui \
  --import_path <sparse_model_dir> \
  --image_path <image_set_root> \
  --database_path <database_path>
```

For the final rig model:

- `--import_path` -> `colmap/.../rig/sparse/0`
- `--image_path` -> `images/.../<count>`
- `--database_path` -> `colmap/.../rig/db/rig.db`

For the bootstrap model:

- `--import_path` -> `colmap/.../rig/sparse/no_rig/0`
- `--image_path` -> `images/.../<count>`
- `--database_path` -> `colmap/.../rig/db/no_rig.db`

Open both in separate GUI windows when you want to compare `no_rig` and `rig`.
