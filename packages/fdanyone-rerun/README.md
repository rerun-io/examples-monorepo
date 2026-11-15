# fdanyone-rerun

[4DAnyone](https://github.com/ant-research/4DAnyone) ([paper](https://arxiv.org/abs/2608.20335)) turns one casual video
of a person into synchronized videos of that person from many new cameras, for downstream 4D Gaussian splatting. GVHMR
recovers the SMPL-X motion; 4DAnyone-Turbo (Wan 2.2 5B with the Turbo-LoRA, four denoising steps) generates the views,
six at a time. This package runs it and shows the run in Rerun.

- Upstream source: `recipes/4danyone-src` (ant-research/4DAnyone `6dcf626`, import name `fdanyone`), with GVHMR from
  `recipes/gvhmr-src` and SageAttention's CUDA kernels from `recipes/sageattention-src`.
- Our code: the run core with its observers (`fdanyone_rerun.core`), the typed run-directory readers (`outputs`), the
  Rerun recording and layout (`rerun_log`), the tyro CLIs (`apis/`) and the tasks in the root `pixi.toml`.

## Quick start

```bash
pixi run -e fdanyone-rerun fdanyone-example          # six views of the example clip (the Space's policy), in Rerun
pixi run -e fdanyone-rerun fdanyone-orbit24          # the README's 24-view full orbit (dense coverage for 4DGS)
pixi run -e fdanyone-rerun fdanyone-run --run.video my.mp4 --run.output-dir data/runs/mine --run.overwrite
pixi run -e fdanyone-rerun fdanyone-log-run --output-dir data/runs/example --video data/source/pexels/2785536-uhd_2160_3840_25fps.mp4
```

The first task downloads ~21 GB of weights (`fdanyone-download`) into `$FDANYONE_MODEL_DIR` (default `data/models`;
set it in the shell to share one copy between checkouts). Add `--rr-config.headless --rr-config.save run.rrd` without a
display. Input video: one person, full or upper body, mild camera motion, at least 121 frames; 1080p+ portrait is best.

`--run.layout` sets the rig: `views-per-layer`, `layer-pitches` (degrees, -15..45), `start-yaw`, `yaw-span`. The total
view count must be a multiple of six (upstream denoises groups of six; more than six views first generate six reference
proposals).

## Recording

Timelines `time` (seconds) and `frame`. `world` is the canonical human world (Y up):

| Entity | What |
| --- | --- |
| `world/body/{mesh,keypoints,skeleton}` | SMPL-X body and the MHR70 keypoints the views are conditioned on |
| `world/source`, `world/source/video` | the source camera fitted to the body, and the input clip (`VideoStream`) |
| `world/cameras/<NN>`, `.../video` | each generated view's camera (hue by yaw) and its video (`VideoStream`) |

The default layout puts the rig and body in 3D beside the source camera, above a grid of the generated views.

## Tests

`pixi run -e fdanyone-rerun-dev gate` (lint, typecheck, deadcode, unit, integration). The integration lane runs the
example clip once on the GPU (`FDANYONE_TEST_RUN=<run dir>` reuses a finished run). `tests-golden` compares that run
with the banked reference recording (below).

## App

`pixi run -e fdanyone-rerun fdanyone-app` serves a Gradio app on the same core as `fdanyone-run`
(`fdanyone_rerun.stream.RunStreamer`). Pick the example or upload a clip, press Run, and the embedded viewer follows
the run through four layouts:

1. motion recovery (source clip and progress log);
2. the recovered body in 3D and over the source;
3. a TAEW2.2 preview of every view at each denoising step (`previews/<NN>` on `diffusion_step`);
4. the camera rig and the generated videos.

The page fits the window. One run at a time.

## Goldens

- `tests-golden` compares a fresh example run (six views, seed 42) with the banked reference recording
  `$FDANYONE_GOLDEN_RRD`, by default `data/golden/example.rrd`. The banked copy is
  `/mnt/nas/ssd-8tb/fdanyone-rerun/golden/example.rrd` (run of 2026-11-15, RTX 5090).
- The band is per-view SSIM at frames 0/30/60/90/120: mean >= 0.92 and every view >= 0.90. Two runs of the same code
  and seed scored 0.948 mean (0.943-0.955 per view). SageAttention and the GPU kernels do not give bit-identical runs,
  so the check is a band.
- Reference timing (RTX 5090, six views, SageAttention, 26.9 GB peak reserved): denoising 105 s, in line with
  upstream's 97 s on the RTX 5090 D. End to end took 576-756 s on a host whose CPU was heavily shared (load 37-108
  on 32 cores). GVHMR, skeleton loading (~14 s per view) and the decode-and-publish tail (~20 s per view) run on the
  CPU and take most of that time.
