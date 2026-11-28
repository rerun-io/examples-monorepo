# fdanyone-rerun

[4DAnyone](https://github.com/ant-research/4DAnyone) ([paper](https://arxiv.org/abs/2608.20335)) turns one casual video
of a person into synchronized videos of that person from many new cameras, for downstream 4D Gaussian splatting. GVHMR
recovers the SMPL-X motion; 4DAnyone-Turbo (Wan 2.2 5B with the Turbo-LoRA, four denoising steps) generates the views,
six at a time. This package runs it and shows the run in Rerun.

- Upstream source: `src/fdanyone`, the run-path copy of ant-research/4DAnyone `6dcf626` plus the speed levers (import name `fdanyone`), with GVHMR from
  `recipes/gvhmr-src` and SageAttention's CUDA kernels from `recipes/sageattention-src`.
- Our code: the run core with its observers (`fdanyone_rerun.core`), the typed run-directory readers (`outputs`), the
  Rerun recording and layout (`rerun_log`), the tyro CLIs (`apis/`) and the tasks in the root `pixi.toml`.

## Quick start

```bash
pixi run -e fdanyone-rerun fdanyone-example          # six views of the example clip (the Space's policy), in Rerun
pixi run -e fdanyone-rerun fdanyone-orbit24          # the README's 24-view full orbit (dense coverage for 4DGS)
pixi run -e fdanyone-rerun fdanyone-run --run.video my.mp4 --run.output-dir data/runs/mine --run.overwrite
pixi run -e fdanyone-rerun fdanyone-log-run --output-dir data/runs/example --video data/source/pexels/2785536-uhd_2160_3840_25fps.mp4
pixi run -e fdanyone-rerun fdanyone-run --train ...             # then train the 4D person on the views (below)
pixi run -e fdanyone-rerun fdanyone-log-run --output-dir data/runs/example --train   # train the person of a finished run
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
| `world/person/splats` | with `--train`: the trained 4D person as Animated SOG (`AnimatedSogSplats3D`) + a one-pose stock preview, static |
| `train/preview`, `train/loss_rgb` | with `--train`: the front view rendered every 500 steps and the RGB loss, on `train_step` |

## The 4D person

`--train` (and the app's second step) trains a 4D Gaussian model of the person on the generated views
(`fdanyone_rerun.person4d`, Open4DGS / TRiGS from `packages/open4dgs`): every view at 704x1280 with the GVHMR SMPL-X body
as the person, BiRefNet mattes in a crop around the projected body, a body-surface init, optionally the MoGe-2 prior,
then TRiGS. The model is encoded as Animated SOG and logged into the run's recording; there is no other output file.
The custom gsplat viewer (`packages/gsplat-rust-renderer`) plays it on `time`; a stock viewer shows one still pose. The
app embeds that viewer: its gradio-rerun comes from `recipes/gradio-rerun-src` (the fdanyone envs do not take the PyPI
wheel). CLI defaults: 25,000 steps, prior on (`--person.no-prior`); the app starts at 5,000 steps with the prior on (about 3 to 3.5 min on an
RTX 5090; held-out normal error 30° -> 16°). "Depth + normal prior (MoGe-2)" under Advanced settings turns it off for a faster run.
MoGe-2 runs through TensorRT when monopriors' engine cache already holds an engine for the frames' size on this machine
(built by an earlier `MoGeV2TrtPredictor` run), else through torch.

The default layout puts the rig and body in 3D beside the source camera, above a grid of the generated views.

## Catalog

`fdanyone-register-catalog` moves a finished run recording into the shared exoego:v2 rig layout and registers it as one
segment (`<clip>__v<NN>-seed<seed>`) of the dataset `fdanyone-views`, so downstream training (Open4DGS) reads the views
with the simplecv catalog readers like any other multi-camera capture. No GPU; the VideoStream packets are moved as
generated (no re-encode).

```bash
pixi run -e fdanyone-rerun fdanyone-register-catalog --rrd run.rrd --run-dir data/runs/mine   # checks the rig against cameras.json
pixi run -e fdanyone-rerun fdanyone-register-catalog --rrd run.rrd --seed 42                  # without a run directory
```

| Entity | What |
| --- | --- |
| `/world/rig_NN/cam_00` | static `cam_T_world` (`ChildFromParent`), view name, yaw and pitch |
| `/world/rig_NN/cam_00/pinhole`, `.../video` | pinhole (RDF, no distortion) and the generated view (`VideoStream`, H.264) |
| `/world/source`, `.../pinhole/video` | the fitted source camera (per frame) and the input clip |
| `/world/pred/fdanyone/people/person_00/body/mesh` | SMPL-X mesh per frame |

Timelines `frame_index` and `video_time`; properties `capture:*`, `episode:*`, `generation:*`. Each H.264 stream has one
keyframe, so a random-access decode starts at frame 0. Registering the same segment id again replaces it. The segment
file goes to `--output-dir` (a path the catalog server can read) and is registered on `--catalog-url`
(`--catalog-url None` writes the file only).

## Tests

`pixi run -e fdanyone-rerun-dev gate` (lint, typecheck, deadcode, unit, integration). The integration lane runs the
example clip once on the GPU (`FDANYONE_TEST_RUN=<run dir>` reuses a finished run). `tests-golden` compares that run
with the banked reference recording (below).

## App

`pixi run -e fdanyone-rerun fdanyone-app` serves a Gradio app on the same core as `fdanyone-run`
(`fdanyone_rerun.stream.RunStreamer`). Pick the example or upload a clip, press Run, and the embedded viewer follows
the run through five layouts:

1. motion recovery (source clip and progress log);
2. the recovered body in 3D and over the source;
3. a TAEW2.2 preview of every view at each denoising step (`previews/<NN>` on `diffusion_step`);
4. the camera rig and the generated videos;
5. with "Train the 4D person after the views" (on by default; steps under Advanced settings): the front view as it trains, the loss and the
   progress (step, it/s, ETA), then the person in 3D, playing with the views.

The page fits the window. One run at a time; Stop ends the run at the next stage, denoising step or training step.
Each run saves what the viewer showed as `data/app-runs/<run_id>.rrd` beside its run directory, and the
page offers it for download when the run has finished.

The gsplat renderer in the embedded viewer needs WebGPU. A browser without it (Linux Chrome until
`chrome://flags/#enable-vulkan` is enabled) gets Rerun's basic splats, and the page says so above the viewer; the saved
recording plays in the native canonical viewer (`rerun-canonical`, `packages/canonical-viewer`).

On a GPU shared with other jobs, pass `--gpu-lock <file>` (e.g. `/tmp/gpu-5090.lock`): the app holds that flock only
while a run is on the GPU and says "Waiting for the GPU" until it gets it. Do not wrap the whole app in `flock`: an app
idles most of its life, and that would block every other job for as long as it is up.

## Speed

The recipe builds upstream plus speed levers (fork branch `upstream-fast`, one commit each; see `recipes/README.md`).
RTX 5090, prod env (no beartype), busy host (load ~10 unless noted), whole CLI run including Rerun logging:

| build | wall | own CPU | denoise | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| default (FP8 W8A8 row-wise) | 199 s | 1,342 CPU-s | 86 s | 24.5 GB |
| default + fused DiT glue (`dit_fused_glue`, 2026-11-27, load 4-11; `FDANYONE_DIT_FUSED=0` turns it off) | 195 s | - | 47 s | - |
| `FDANYONE_DIT_FP8=0` (BF16) | 230 s | 1,389 CPU-s | 112 s | 29.2 GB |
| `FDANYONE_TURBO_KEPT_STEPS=0,2,3` (3 steps, BF16) | 184 s | 1,324 CPU-s | 72 s | 29.2 GB |
| default, 24-view orbit (RCP + 4 TCR groups, 8-view groups), busy host (load 16-38) | 767 s | 4,444 CPU-s | 85 s RCP + 389 s | 25.8 GB |

The fused glue (`fdanyone/model/fused.py`, Triton) replaces the eager FP32 RMSNorm + RoPE, LayerNorm + modulation,
gated residuals and FP8 activation quantisation of every block with one pass each; one six-view forward drops from
19.7 to 11.6 s at equal load. What is left is GEMM and attention at the hardware peak.

The 3-step Turbo schedule is opt-in: it is visibly softer (mean per-view SSIM 0.862 against the full-VAE reference,
versus 0.897 for the default).

## Goldens

- `tests-golden` compares a fresh example run (six views, seed 42) with the banked reference recording
  `$FDANYONE_GOLDEN_RRD`, by default `data/golden/example-fp8.rrd`. The banked copy of the default (FP8) build is
  `/mnt/nas/ssd-8tb/fdanyone-rerun/golden/example-fp8.rrd` (RTX 5090, 2026-11-15).
- The band is per-view SSIM at frames 0/30/60/90/120, calibrated from a same-code repeat of the default build
  (see `tests/test_golden.py`). FP8 (dynamic activation scales) and SageAttention do not give bit-identical runs.
- The earlier BF16 + full-VAE reference stays at `/mnt/nas/ssd-8tb/fdanyone-rerun/golden/example.rrd`: against it the
  default scores 0.897 mean SSIM.
