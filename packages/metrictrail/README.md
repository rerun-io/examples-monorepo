# metrictrail

NVIDIA MetricTrail is a monocular SLAM method in ViPE (`pipeline=metrictrail`). It is DPVO with two frozen priors:
MoGe-2 metric depth starts each new patch at a depth in metres and holds the bundle adjustment to that scale, and
DINOv3 ViT-S/16 features improve the patch matching. The output is camera poses and intrinsics at metric scale, plus
per-frame metric depth and, from the same MoGe-2 forward, surface normals.

This package holds our code around upstream: benchmark presets, a run CLI with a Rerun logger, the VSLAM-LAB / paper
evaluation and the catalog layer. ViPE itself comes from `recipes/vipe-src`.

## Packaging

| Part | Where it comes from | Why |
| --- | --- | --- |
| ViPE + MetricTrail | `recipes/vipe-src`: the GitHub archive of ViPE `ead2e00`, sha256-pinned | `ead2e00` is the first commit with `vipe/metrictrail`; no release has it (v1.2.0 predates it). |
| CUDA kernels | the recipe compiles `vipe_ext` (without `lietorch_ext`) at package build time (Python 3.12, conda-forge torch 2.12 cuda130, sm_86 + sm_120 + PTX). DPVO's `cuda_corr` + `cuda_ba` are `packages/dpvo`'s, built in place by the `metrictrail` feature's `_build-dpvo-cuda-kernels` task (a `depends-on` of `metrictrail-run`, `metrictrail-app`, `metrictrail-check-kernels` and `tests-integration`; for the local GPU only) | `vipe_ext` is part of `pixi install`; the DPVO kernels are the same build as the `dpvo` env's (the `.so` files in `packages/dpvo/dpvo` serve both envs). |
| lietorch | patch 0001: ViPE's vendored lietorch (`vipe/ext/lietorch`) imports `lietorch_mojo` and `build.sh` deletes the vendored code; the DPVO shim's `lietorch` re-exports it too (the DPVO copies go with ViPE's DPVO) | one lietorch for the workspace (lietorch-mojo). ViPE's copies are code-identical to upstream lietorch. |
| DINOv3 | patch 0002: `recipes/dinov3-src` (a run dependency) | the same ViT-S/16; patch tokens are bit-identical. |
| MoGe-2 (`moge2-l-normal`) | patch 0003: `packages/monoprior`'s MoGe-2 code for ViPE's own `moge2-l` (`build.sh` deletes ViPE's copy). Our port's prior is this package's `metrictrail.moge2`, registered in ViPE's `DEPTH_PROVIDERS` (patch 0004 adds only that registry and the `fuse_point_cloud` hook `metrictrail.fusion` uses for the normals) and selected by the run CLI's overrides `pipeline.slam.keyframe_depth` and `pipeline.init.calibration.depth_model`: it runs monoprior's `MoGeV2TrtPredictor` (heads `geometry`, batch 1) on `Ruicheng/moge-2-vitl-normal` @ `b135031`, one TensorRT engine per input token grid at 3600 tokens; torch FP16 on the same checkpoint when TensorRT does not import (`VIPE_MOGE2_BACKEND=auto\|tensorrt\|torch`) | one forward gives depth and normals. ViPE's FoV-conditioned post-processing is kept on the raw outputs; the forward does not depend on the FoV, so it is cached per source frame and a new focal re-runs only the post-processing. Depth differs from ViPE's `moge-2-vitl` by 1.96 % AbsRel (median +1.3 %) on 64 TUM frames; accepted. |
| DPVO | `packages/dpvo`'s `dpvo.metric.MetricDPVO` (an editable path dependency of the `metrictrail` feature). The recipe deletes ViPE's vendored DPVO, its kernels and the stereo runner, and installs re-export shims at `vipe.metrictrail.dpvo` (`recipes/vipe-src/dpvo_shim`), so ViPE's `slam.py` builds `MetricDPVO` from its yacs config unchanged; patch 0005 hands the DINO patchifier `packages/dpvo`'s patchify keywords | one DPVO for the workspace. ViPE's fork (metric-depth local BA, the metric global-BA backend, the `cuda_ba` prior kernel) is ported onto the typed core's hooks; the three presets run ViPE's operations in ViPE's order (lockstep against ViPE's copy on TUM, 7-Scenes and KITTI). MetricTrail does not use DBoW loop closure, so `packages/dpretrieval` is not used. `metrictrail.capture` observes DPVO through `dpvo.hooks.add_global_observer`. |

Weights are on the NAS (`/mnt/nas/ssd-8tb/metrictrail/weights`, `MANIFEST.md` beside them). The `metrictrail` feature
sets ViPE's checkpoint variables to these files. MoGe-2's `moge-2-vitl-normal` is the exception: monoprior loads it from
the Hugging Face cache (pinned revision) when it exports the ONNX for a new token grid or when the torch fallback runs.
ONNX files and TensorRT engines are cached under `~/.cache/monoprior` (`MONOPRIOR_TRT_CACHE`); the first run on a new
raster aspect builds its engines (about 40 s each on the RTX 5090; TUM needs two, 728x966 for tracking and 714x980 for
the SharedK calibration raster).

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
- `metrictrail-check-kernels`: run one small op of each compiled kernel (`vipe_ext`, and DPVO's through `dpvo.kernels`) on the GPU.

## App

```bash
pixi run -e metrictrail metrictrail-app
```

A Gradio app (`metrictrail/gradio_ui/metrictrail_ui.py`, shim `tools/gradio_app.py`): upload a video or pick an example
(the task downloads dpvo's example clips into `data/`), and the run streams into an embedded Rerun viewer. It runs
`metrictrail.apis.run.run_and_log` with the `video` preset and the default MoGe-2 provider, so it calibrates the camera
itself; there is no intrinsics input. Settings: the tracking stride and `max_frames` (upstream's Hydra keys). The capture
runs in live mode: DPVO's frames, patches and BA window stream in as it tracks, with depth and normals of new keyframes
only and no DINOv3 PCA. Calibration has no hook, so the viewer stays empty for those seconds. The trajectory and the
surface arrive at the end; the TSDF mesh grows in the DPVO tracking view from each new keyframe (at DPVO's current pose) and the final mesh (final poses) replaces it. Gradio prints the local URL. Open it over HTTPS (`tailscale serve`): on plain HTTP the
viewer stays empty.

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
