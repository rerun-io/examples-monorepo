# metrictrail

NVIDIA MetricTrail is a monocular SLAM method in ViPE (`pipeline=metrictrail`). It is DPVO with two frozen priors:
MoGe-2 metric depth starts each new patch at a depth in metres and holds the bundle adjustment to that scale, and
DINOv3 ViT-S/16 features improve the patch matching. The output is camera poses and intrinsics at metric scale, plus
per-frame metric depth and, from the same MoGe-2 forward, surface normals.

This package holds our code around upstream: benchmark presets, a run CLI with a Rerun logger, the VSLAM-LAB / paper
evaluation and the catalog layer. ViPE itself comes from `recipes/vipe-src`.

## Quick start

On Linux with an NVIDIA GPU, from the repository root:

```bash
pixi run -e metrictrail --frozen metrictrail-example
```

This runs the 86 s iPhone clip `data/iphone/pool.mp4` with the `video` preset and opens the Rerun viewer. Without a
display, add `--rr-config.save /abs/path/pool.rrd`. Outputs go to `packages/metrictrail/data/example/pool`. The first
run builds the Mojo kernels, downloads the weights and the clip, and builds TensorRT engines; later runs reuse them:

| What | Cached in |
| --- | --- |
| DINO-DPVO (`nvidia/MetricTrail`), DINOv3 ViT-S/16 (`xysun917/dinov3-vits16`, a mirror of Meta's file) | `packages/metrictrail/data/weights/` |
| The example clip (`pablovela5620/dpvo-example-data`) | `packages/metrictrail/data/iphone/` |
| MoGe-2 `Ruicheng/moge-2-vitl-normal` | the Hugging Face cache (`HF_HOME`) |
| DROID `droid.pth` (SharedK calibration) | the torch hub (`TORCH_HOME`, default `~/.cache/torch`) |
| ONNX files and TensorRT engines | `~/.cache/monoprior` (`MONOPRIOR_TRT_CACHE`), `~/.cache/trtkit/trt` (`TRTKIT_TRT_CACHE`) |

To use weight files kept elsewhere, export `VIPE_METRICTRAIL_DPVO_CHECKPOINT` and `VIPE_DINOV3_VITS16_CHECKPOINT`
before `pixi run`: a value set in the shell wins over the feature's defaults.

## Packaging

| Part | Where it comes from | Why |
| --- | --- | --- |
| ViPE + MetricTrail | `recipes/vipe-src`: the GitHub archive of ViPE `ead2e00`, sha256-pinned | `ead2e00` is the first commit with `vipe/metrictrail`; no release has it (v1.2.0 predates it). |
| Kernels | Mojo, one source for CPU and CUDA, compiled for the local GPU. ViPE's CUDA extension `vipe_ext` is not built: patch 0007 makes `vipe.ext` import `packages/vipe-mojo`'s `vipe_mojo` (the ops MetricTrail calls; the others raise `NotImplementedError` naming the op). DPVO's kernels are `packages/dpvo`'s `dpvo.mojo_backend`. Built by the `vipe-mojo-build` and `dpvo-mojo-build` tasks (a `depends-on` of `metrictrail-run`, `metrictrail-example`, `metrictrail-app` and `tests-integration`) | no nvcc build at `pixi install`; one kernel source per op for every device. |
| lietorch | patch 0001: ViPE's vendored lietorch (`vipe/ext/lietorch`) imports `lietorch_mojo` and `build.sh` deletes the vendored code; the DPVO shim's `lietorch` re-exports it too (the DPVO copies go with ViPE's DPVO) | one lietorch for the workspace (lietorch-mojo). ViPE's copies are code-identical to upstream lietorch. |
| DINOv3 | patch 0002: `recipes/dinov3-src` (a run dependency) | the same ViT-S/16; patch tokens are bit-identical. |
| MoGe-2 (`moge2-l-normal`) | patch 0003: `packages/monoprior`'s MoGe-2 code for ViPE's own `moge2-l` (`build.sh` deletes ViPE's copy). Our port's prior is this package's `metrictrail.moge2`, registered in ViPE's `DEPTH_PROVIDERS` (patch 0004 adds only that registry and the `fuse_point_cloud` hook `metrictrail.fusion` uses for the normals) and selected by the run CLI's overrides `pipeline.slam.keyframe_depth` and `pipeline.init.calibration.depth_model`: it runs monoprior's `MoGeV2TrtPredictor` (heads `geometry`, batch 1) on `Ruicheng/moge-2-vitl-normal` @ `b135031`, one TensorRT engine per input token grid (1,200 tokens by default); torch FP16 on the same checkpoint when TensorRT does not import (`VIPE_MOGE2_BACKEND=auto\|tensorrt\|torch`) | one forward gives depth and normals. ViPE's FoV-conditioned post-processing is kept on the raw outputs; the forward does not depend on the FoV, so it is cached per source frame and a new focal re-runs only the post-processing. Depth differs from ViPE's `moge-2-vitl` by 1.96 % AbsRel (median +1.3 %) on 64 TUM frames; accepted. |
| DPVO | `packages/dpvo`'s `dpvo.metric.MetricDPVO` (an editable path dependency of the `metrictrail` feature). The recipe deletes ViPE's vendored DPVO, its kernels and the stereo runner, and installs re-export shims at `vipe.metrictrail.dpvo` (`recipes/vipe-src/dpvo_shim`), so ViPE's `slam.py` builds `MetricDPVO` from its yacs config unchanged; patch 0005 hands the DINO patchifier `packages/dpvo`'s patchify keywords | one DPVO for the workspace. ViPE's fork (metric-depth local BA, the metric global-BA backend, the `cuda_ba` prior kernel) is ported onto the typed core's hooks; the three presets run ViPE's operations in ViPE's order (lockstep against ViPE's copy on TUM, 7-Scenes and KITTI). MetricTrail does not use DBoW loop closure, so `packages/dpretrieval` is not used. `metrictrail.capture` observes DPVO through `dpvo.hooks.add_global_observer`. |

The hidden task `_download-metrictrail-weights` (a `depends-on` of `metrictrail-run`, `metrictrail-app`,
`metrictrail-example` and `tests-integration`) fetches DINO-DPVO and DINOv3 at pinned revisions into `data/weights/`;
the `metrictrail` feature points ViPE's checkpoint variables at these files. MoGe-2's `moge-2-vitl-normal` comes from
the Hugging Face cache (pinned revision) when monoprior exports the ONNX for a new token grid or when the torch fallback runs.
ONNX files and TensorRT engines are cached under `~/.cache/monoprior` (`MONOPRIOR_TRT_CACHE`); the first run on a new
raster aspect builds its engines (about 40 s each on the RTX 5090); tracking and SharedK calibration can need
separate grids.

The CLI and app use MoGe-2 detail level 0 (1,200 tokens) for calibration and tracking, and 64 patches per frame.
These are deliberate accuracy/speed trades. Set `--depth-resolution-level 9 --patches-per-frame 96` to restore the
original depth grid and patch count. Each detail level has its own engines and forward cache; the run record stores
the selected level and resolved tracker configuration. The wider paper evaluation below used level 9 and 96 patches
and has not been repeated with these defaults.

The `ours` path replays DINO and both DPVO convolutional encoders with CUDA graphs during inference. Graphs are scoped
to one run and separated by model, raster and precision; FP32 recovery uses its own graph. Returned features own their
storage, and logging stays outside capture. The recurrent update prepares its neighbor/group indices before replay;
its cache holds at most eight recurring shapes, with uncommon shapes evaluated normally.
Factor pruning and motion checks reuse compacted edge indices across their tensors. Motion checks and inference
projection use fused Mojo kernels that preserve per-camera intrinsics, quaternion normalization and near-plane
clipping. Projection keeps its original autograd path when gradients are enabled. CUDA correlation pairs neighboring
half-precision outputs with native half arithmetic, keeping the channel order and separate product/sum rounding.

On the shared RTX 5090, the 642-frame `IMG_0493.mp4` clip measured **68.27 tracking FPS** without capture and
**46.25 FPS** with capture (median of three full runs each, prod environment, no mesh). Process wall medians were
34.96 / 38.52 s, including SharedK calibration at 16.36 / 17.52 s. The original 3,600-token, 96-patch path measured
26.83 / 24.29 tracking FPS. GPU contention matters: final no-capture samples were 69.28, 68.27 and 56.73 FPS.
These are tracking rates; model loading, calibration and final logging add to the process wall time.

The final TUM fr1_desk check (seed 1234) gave **2.307 cm Sim(3) ATE**, scale **1.06327**, and 5.684 cm SE(3) ATE.
The original settings gave 2.186 cm, 1.05873 and 5.313 cm. The new defaults pass the fixed guard of at most 10% more
Sim(3) ATE and a scale change within 0.02. All 613 TUM poses and all 642 poses in each video run were finite.

## Run

lietorch comes from `packages/lietorch-mojo`: both envs compose the `lietorch-mojo` feature, and `metrictrail-run`
depends on `lietorch-mojo-build`, which compiles its Mojo extension for the local GPU on the first run. Use the prod
env `metrictrail` for runs, not `metrictrail-dev`. Paths must be absolute: the tasks run in `packages/metrictrail`.

One end-to-end run on TUM fr1_xyz (798 frames, about 1 min 40 s on the RTX 5090) to a `.rrd`:

```bash
XFORMERS_DISABLED=1 pixi run -e metrictrail --frozen metrictrail-run \
    --input /mnt/nas/ssd-8tb/metrictrail/data/TUM/rgb/xyz --benchmark tum \
    --gt-sequence /mnt/nas/ssd-8tb/datasets/VSLAM-LAB-Benchmark/RGBDTUM/rgbd_dataset_freiburg1_xyz \
    --output /tmp/mt-xyz --seed 1234 --rr-config.headless --rr-config.save /tmp/mt-xyz/xyz.rrd
```

`--input` takes a video, a frame directory or a VSLAM-LAB sequence. `--benchmark` selects upstream's settings
(`tum`, `7scenes`, `kitti`, `kitti_calibrated`, `wild_slam`, `lamaria`, `oxford`, `vbr`, or `video` for the README
quick start). `--no-capture` turns off the live intermediates (faster). Without a display, always pass
`--rr-config.headless` or `--rr-config.save`.

Other tasks:

- `metrictrail-log-run --run-dir <vipe_out> ...`: log a finished output directory (ours or upstream's) without the GPU.
  It has no normals: they exist only in a live capture (`cam/normals`, `world/cloud_normals` with `--fuse-cloud`).

## App

```bash
pixi run -e metrictrail metrictrail-app
```

A Gradio app (`metrictrail/gradio_ui/metrictrail_ui.py`, shim `tools/gradio_app.py`): upload a video or pick an example
(the task downloads dpvo's example clips into `data/`), and the run streams into an embedded Rerun viewer. It runs
`metrictrail.apis.run.run_and_log` with the `video` preset and the default MoGe-2 provider, so it calibrates the camera
itself; there is no intrinsics input. Settings: the tracking stride, `max_frames` (upstream's Hydra keys) and the SharedK
calibration budget (`fast64` by default, half the time of ViPE's `robust128`). The input video is logged first and plays at
once. While SharedK calibrates, the viewer shows its own layout on the `calibration_step` timeline: the DROID keyframe map,
the focal and the keyframes kept, and the newest keyframe's MoGe-2 prior against its bundle-adjusted depth. The capture
then runs in live mode: DPVO's patches (on the video), BA window and
the DINOv3 PCA video stream in as it tracks, with depth and normals of new keyframes only. The trajectory and the
surface arrive at the end; the TSDF mesh grows in the DPVO tracking view from each new keyframe (at DPVO's current pose) and the final mesh (final poses) replaces it. Gradio prints the local URL. Open it over HTTPS (`tailscale serve`): on plain HTTP the
viewer stays empty.

**Video, not images, in the recording** (`metrictrail.rerun_log`, the EgoHand pattern): the input video is logged once as
`rr.VideoStream` from its own samples (`input/video`, no re-encode for an H.264/AV1 MP4; HEVC, other containers and
rotated clips get one H.264 viewer copy) on `video_time` = its PTS, under an `input` pinhole of the source camera (the
tracking raster's K mapped back through ViPE's resize and crop), and every per-frame entity carries that `video_time`.
The DINOv3 PCA (and, for a frame directory, the tracked frames as `cam/rgb`) is encoded per frame to an H.264
`VideoStream` (NVENC; libx264 with a warning without it). Depth stays a millimetre PNG per frame (Rerun has no depth
video); keyframe normals stay JPEG (sparse).

**Video input stays on the GPU** (`metrictrail.gpu_video`): a video decodes on NVDEC (simplecv `TorchCodecVideoReader`,
display rotation applied as OpenCV does), ViPE's input raster (`raster_frame`) resizes and crops on the GPU, and the
prepared clip stays in GPU memory for every pass and for the surface colours. Upstream's video path decodes with
OpenCV, rasters on the host and re-uploads the clip each pass. Frame directories (every benchmark) keep upstream's path.

**The surface is a TSDF mesh** (`metrictrail.surface`, `packages/tsdf-rs`: a sparse colour TSDF in Rust with marching
cubes, from pablovela5620/simplerecon-inference): every tracked frame's metric depth at its final pose, 4 cm voxels to
8 m (`--tsdf-voxel-m`, `--tsdf-max-depth-m`). `--fuse-cloud` adds upstream's fused point cloud
(`fuse_point_cloud`: multi-view consistency filter, 2 cm voxel means, NumPy).

pool (2,585 frames strided to 1,293; RTX 5090, CLI with capture, `metrictrail` env):

| | wall | fusion | surface |
|---|---|---|---|
| upstream video path + fused cloud | 224 s | 94 s | 5.64 M points |
| GPU video input + fused cloud | 188 s | 94 s | 5.64 M points |
| GPU video input + TSDF mesh (default) | 97 s | 4 s | 2.94 M triangles |

The rows above ran on the original iPhone `pool.MOV`. The app's examples are now dpvo's H.264 re-encodes
(`iphone/pool.mp4`, no B-frames): the default row on `pool.mp4` takes 95 s for 2.92 M triangles.

## Evaluate

```bash
pixi run -e metrictrail --frozen metrictrail-eval --runs <out seed1234> <out seed1235> <out seed1236> \
    --dataset rgbdtum --out-dir <new dir> --vslamlab-dir <VSLAM-LAB checkout>
```

The task scores each output directory two ways and writes `report.json`:

- the paper protocol: ATE RMSE after one Sim(3) alignment per sequence, unweighted mean;
- VSLAM-LAB's own `evaluate_sequence` (a subprocess in the VSLAM-LAB checkout's `vslamlab` pixi env, with a private
  evo HOME).

Datasets: `rgbdtum`, `7scenes`, `kitti`.

## Catalog layers

```bash
pixi run -e metrictrail --frozen metrictrail-catalog-layer --runs-root <out seed1234> --report <report.json> \
    --benchmark tum --source metrictrail --producer "examples-monorepo metrictrail @ <sha>" \
    --catalog-url rerun+http://<host>:<port>
```

The task writes one layer `.rrd` per sequence to a new directory, `<results_root>/catalog/<source>/<run_tag>/`, and
never overwrites a layer file. With `--catalog-url` it registers the layers on the `dataforge-vslam-lab-<dataset>`
dataset: `ERROR` where the segment has no such layer yet, `REPLACE` where it has an older run. The run metrics are
recording properties in the layer, not sidecar files. The layer name is the run source (`/world/runs/<source>`):

| layer | run | colour |
| --- | --- | --- |
| `metrictrail` | ours, the shipped path: MoGe-2 `moge-2-vitl-normal` through TensorRT, with surface normals (`--capture-rrd`) | magenta `#E040FB` |
| `metrictrail_paper` | our torch reproduction of the paper configuration (`moge-2-vitl`, FP32) | violet `#A78BFA` |
| `metrictrail_upstream` | unmodified upstream ViPE @ ead2e00 on this machine | amber `#F5A623` |

Ground truth is dataforge's `gt` layer (green `#3DDC97` in both layouts). On the shared Spark catalog
(`rerun+http://dgx-spark:51235`):

| dataset | segments | MetricTrail layers |
| --- | --- | --- |
| `dataforge-vslam-lab-rgbdtum` | 9 | `metrictrail`, `metrictrail_paper`, `metrictrail_upstream` |
| `dataforge-vslam-lab-7scenes` | 18 | `metrictrail_paper`, `metrictrail_upstream` |
| `dataforge-vslam-lab-kitti` | 11 | `metrictrail_paper`, `metrictrail_upstream` |
| `dataforge-lamaria` | `lamaria__R_01_easy` only | `metrictrail_paper` |

The shipped path has run on TUM only, so 7-Scenes, KITTI and LaMAria hold no `metrictrail` layer.

### Layouts

`metrictrail.catalog.blueprint` owns two layouts on dataforge's catalog-blueprint hook (`MetricTrailVslamLab`
and `MetricTrailLamaria` override `catalog_blueprints`; dataforge's `republish_blueprints` publishes them):

- `default`, the dataset default: ground truth and ours (`metrictrail`). Rig and Follow 3D views, rgb, the sensor
  depth, our keyframe depth and, where the layer has them, our keyframe normals. The 3D views hide every other run.
  Where the dataset has no `metrictrail` layer, its pane says "not run on this dataset" and the views show ground
  truth only.
- `comparison`, a development and validation view: ground truth, `metrictrail`, `metrictrail_paper` and
  `metrictrail_upstream` in the colours above, with one keyframe-depth pane per run.

Publish one with `--publish-blueprint default|comparison` together with `--catalog-url` (files go to
`<blueprint_root>/blueprints/`). The flag is off by default. Do not run `dataforge-publish-blueprints` on the
`dataforge-vslam-lab-*` datasets: it publishes dataforge's generic layout, which shows every run in its own palette.
Take the Spark catalog snapshot after every publish.

`dataforge-lamaria` is different: it is dataforge's own dataset (10 segments, MetricTrail's layer is on
`lamaria__R_01_easy` only), so it keeps dataforge's default layout. `default` is refused there. dataforge's Rig and
Follow views already show the `metrictrail_paper` path and points on R_01. For its keyframe-depth pane, use the
`comparison` layout (`--publish-blueprint comparison`; the flag reaches VSLAM-LAB datasets only, so on LaMAria call
`publish_layout(<catalog_url>, "dataforge-lamaria", "comparison", <blueprint_root>)` directly). It replaces dataforge's
default while it is live, so restore dataforge's layout when you are done:

```bash
pixi run --frozen -e dataforge dataforge-publish-blueprints --catalog-url <catalog_url> \
    --catalog-name dataforge-lamaria --output-root <dataforge output root> lamaria
```

## Results against the paper

Metric: ATE RMSE after one Sim(3) alignment per sequence, unweighted mean, as in the paper. We wrote the tolerance
before the first result: a dataset passes if |mean - paper| <= max(3 sigma, 10 % of the paper value, half a unit of
the last printed digit), and the port matches upstream if |port - upstream| <= 3 sqrt(sigma_port^2 + sigma_up^2)
(0.2 cm floor). All runs are on an RTX 5090.

### Shipped path: TensorRT `moge-2-vitl-normal`

TUM RGB-D fr1, 9 sequences, seed 1234: **4.986 cm against the paper's 4.4 cm**, 0.15 cm outside the 0.44 cm band. This is
a known and accepted trade for the surface normals (2026-11-06). The largest change against the reference path is
fr1_floor (2.52 -> 4.61 cm); the other sequences stay inside the reference path's seed spread. 7-Scenes, KITTI and LaMAria
were not re-run on this path. On fr1_xyz the run is 2.1x faster end to end (3:27.5 -> 1:40.5; 1,015 MoGe-2 forwards
instead of 1,391; tracking 20.7 fps instead of 17.7).

### Paper-match reference: ViPE's `moge-2-vitl`, torch FP32 (before the swap)

The reproduction of the paper. "Port" is this package before the MoGe-2 swap; "upstream" is unmodified ViPE `ead2e00`
with its own lietorch and MoGe-2. Seed 1234 answers the paper question; extra seeds ran only where time allowed.

| Dataset (paper table) | Paper | Upstream ViPE | Port | Port vs upstream |
| --- | --- | --- | --- | --- |
| TUM RGB-D fr1, 9 seq (Table 9) | 4.4 cm | 4.478 ± 0.074 cm, 3 seeds: PASS | 4.553 ± 0.217 cm, 3 seeds: PASS | +0.075 cm, band 0.687: PASS |
| 7-Scenes, 18 seq (Table 8) | 6.3 cm | 6.378 cm, 1 seed: PASS | 6.377 cm, 1 seed: PASS | −0.001 cm (0.2 cm floor): PASS |
| KITTI 00-10 (Tables 1, 12) | 14.05 m | 13.450 ± 0.044 m, 2 seeds: PASS | 14.089 ± 0.430 m, 2 seeds: PASS | +0.639 m, band 1.298: PASS |
| LaMAria R_01_easy, 1 of 5 seq (Table 10) | 3.09 m | not run | 3.251 m, 1 seed: PASS (band 0.309) | — |

These port runs used a temporary lietorch provider (upstream lietorch C++ re-exported as `lietorch_mojo`). One check on
lietorch-mojo: TUM fr1_xyz, seed 1234, 1.295 cm against 1.297 cm. KITTI 01 beats the paper on both sides (upstream
27.98 m against 35.94 m). Tracking speed of the port is +2 % (TUM) to +4 % (KITTI) against upstream.

## Build time

Clean `vipe-src` rebuild, `pixi install -e metrictrail-dev`, 16 jobs, niced, on a shared host with a warm package
cache (a cold download was not measured):

| What | Time |
| --- | --- |
| `pixi install -e metrictrail-dev`, recipe rebuilt | 127 s |
| rattler-build recipe step | about 2 min |
| `vipe_ext` + DPVO kernels | 50 s + 69 s (before the switch to `packages/dpvo`: the recipe now builds only `vipe_ext`) |

## Known gaps

- **TUM on the shipped path misses the band** (4.986 cm, 1 seed); accepted for the normals. Follow-up if paper-exact
  depth is needed: a `moge-2-vitl` TensorRT engine for depth, plus `vitl-normal` on keyframes only for normals.
- **Seeds.** Reference path: TUM 3 seeds per side, KITTI 2, 7-Scenes and LaMAria 1; shipped path: TUM 1.
- **fr1_360** has two Sim(3)-scale modes on the same seed in upstream too (7.86 vs 10.09 cm): GPU nondeterminism in
  calibration. Same-seed runs are not bit-reproducible.
- **Not run:** Oxford Spires, VBR, Wild-SLAM, the other four LaMAria sequences, KITTI calibrated / loop closure / stereo.
- **`metrictrail-log-run`** (post-hoc logging) has no normals; they exist in a live capture only.
- **KITTI stereo** is not available: the recipe deletes ViPE's stereo runner and its DPVO copy.
- **linux-64 only.** linux-aarch64 (DGX Spark) was not tried.
- **monopriors brings its full PyPI closure** (diffusers, pytorch-lightning, sam3, ...), although MoGe-2 needs only a
  few of those packages.
- **`metrictrail-eval`'s default `--vslamlab-dir`** is a scratch path (`/tmp/metrictrail-scratch/vslamlab/VSLAM-LAB`).
  Pass your VSLAM-LAB checkout.
- **Catalog.** The shipped path (`metrictrail`) is on the 9 TUM segments only; 7-Scenes, KITTI and LaMAria carry the
  torch reproduction (`metrictrail_paper`) and, except LaMAria, upstream ViPE.

SharedK can use `--calibration-graphs` for prepared DROID graph metadata and a CUDA graph containing
one complete projection/correlation/recurrent-update/BA step. Each shape bucket warms on its first real
update, captures on its second, then replays; no additional solver iterations are applied. Two buckets
bound retained memory. Connectivity, correlation volumes, state and all solve tables are copied on reuse.
Float64 LLT stays on the GPU for frontend and backend steps. Live
calibration hooks still see the same buffer and edge tensors. Proximity/keyframe decisions and
CUDA's exact age-tie ordering remain outside capture. This option is enabled by default after the pool/movie focal and TUM accuracy guards passed for both budgets.
Use `--no-calibration-graphs` to select the original graph implementation.

`--calibration-gru-backend tensorrt` selects the frozen DROID ConvGRU TensorRT backend.
trtkit's engine cache (`TRTKIT_TRT_CACHE`, default `~/.cache/trtkit/trt`) holds its ONNX files and engines under `droid-gru/`;
identity includes actual weights, raster, FP16 precision, edge-count profile, TensorRT version and GPU.
The dynamic profile is 1/48/128 edges (min/opt/max). CPU, training, gradients, FP32 and counts outside
that profile retain PyTorch. Outputs own their storage. A runtime lives as long as its model, so the app (which keeps its models) reuses it,
while each calibration starts with fresh hidden state. The run prints export/build/load time separately;
these costs remain included in total wall time. A Space must ship matching ONNX and engine artifacts in
this directory to avoid building them on its first calibration. This backend is enabled by default after those same guards passed; `--calibration-gru-backend torch` selects PyTorch.
