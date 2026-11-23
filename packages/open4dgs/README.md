# open4dgs

Open4DGS in the monorepo: 4D Gaussian people trained from Rerun catalog segments with the TRiGS trainer.

- The trainer is upstream code: `recipes/open4dgs-trigs-src` installs `trigs` (ASU-NEM-Labs/open4dgs-trigs at
  `ed7a84a`) with two patches: PyTorch3D's `knn_points` / `sample_farthest_points` come from
  `open4dgs.pytorch3d_ops` (plain torch, so PyTorch3D is never built), and `build_canonical_from_npz` takes the
  keyframes arrays as a Mapping, so the init cloud is passed in memory.
- This package holds our code: catalog input, the in-memory hand-off to the trainer, priors (MoGe-2 depth and
  normals through `monopriors`), held-out evaluation and Rerun output.

## Environments

`open4dgs` (prod: training runs and timings) and `open4dgs-dev` (tests, lint, typecheck), linux-64 and linux-aarch64:
gsplat is the ai-demos 1.5.3 build (CUDA 13, torch 2.12, sm_86 + sm_120) on linux-64 and 1.6.0 on linux-aarch64 (no
1.5.3 build exists there; the Spark passed a smoke test, no held-out band check). Both carry the Rerun catalog
lane (rerun-sdk 0.42 with the catalog and dataloader extras), simplecv, torchcodec and monopriors.

```bash
pixi install -e open4dgs-dev
pixi run -e open4dgs-dev --frozen tests
```

No pycolmap: the in-memory path does not read COLMAP folders. Upstream's COLMAP reader (`trigs.io.dataset`) imports
pycolmap inside a try and prints a notice when it is absent.

## In-memory training (what the trainer needs)

1. Init cloud: a dict with `positions` (N, 3) float32, `velocities` (N, 3) float32 (world units per video frame,
   `velocity_units = "world_per_video_frame"`), `colors` (N, 3) float32 in [0, 1], `times` (N, 1) float32
   (keyframe / (n_frames - 1)), plus `frame_start`, `frame_end`, `keyframe_step` scalars; optional `durations`,
   `has_velocity`, `is_static`, `source_frames`. `trigs.init.from_colmap.build_canonical_from_npz(mapping, fps=...,
   keyframe_stride=..., device="cuda")` returns `(params, sh_degree, scene_extent)`.
2. Cameras: `trigs.core.camera.CameraConfig.from_matrices(viewmats=(C, 4, 4) world-to-camera OpenCV, Ks=(C, 3, 3)
   pixels, widths=[...], heights=[...])`.
3. Frames and masks: anything indexable as `[frame][camera]`: RGB uint8 (H, W, 3) CPU tensors, masks float (H, W)
   in [0, 1] (used when `config.masked_rgb_loss`).
4. `trigs.core.trainer.TrigsTrainer("cuda").train_from_params(params=..., sh_degree=..., scene_extent=...,
   cameras=..., n_frames=..., gt_images_per_frame=..., gt_alphas_per_frame=..., config=TrigsTrainingConfig(...),
   writer=...)`; held-out cameras are `config.holdout_views` (camera positions); `writer` receives the losses and the
   eval PSNR through `add_scalar`.

## Train a person and report held-out numbers

One command (prod env, under the shared GPU lock):

```bash
cd packages/open4dgs
flock -x /tmp/gpu-5090.lock pixi run -e open4dgs --frozen open4dgs-train \
  --dataset fdanyone-views --segment-id pexels-2785536__v06-seed42 \
  --num-frames 30 --max-side 640 --num-holdout 1 --train.steps 1500 --run-name my-run
```

Steps (`open4dgs.apis.train`):

1. `read_catalog_clip` reads the window; `select_holdout(camera_names, num_holdout, seed)` picks the held-out
   cameras (`random.Random(seed).sample(sorted(names), k)`).
2. Masks: the catalog `people_matte` layer (`--mask-source layer`, `--mask-catalog-url`/`--mask-dataset` for a companion
   dataset such as `open4dgs-mamma`), or a dilated SMPL-X silhouette (`--mask-source mesh`, development only).
3. Init: `body_surface_init` (`--init body`, default): fixed area-weighted barycentric samples on each SMPL-X mesh
   (30k per person per keyframe), velocity = central difference in world units per video frame, colours from the
   training cameras only. The window's last frame is always a keyframe (its own mesh, else extrapolated from the last
   two mesh keyframes), so frames past the grid keep their Gaussians. `--init layer --init-layer init_roma` reads the
   segment's init cloud layer from the mask catalog/dataset instead; the run refuses when that layer's held-out
   cameras or window differ from the run's. Either way the result is an `InitCloud` (`open4dgs.init_cloud`), the
   trainer's init contract, also logged to the run.
4. `train_clip`: TRiGS in memory. Masked loss with the render composited on black against `gt * mask`, plus the
   alpha-outside and alpha-to-mask terms. Held-out cameras get black frames and empty masks, so their pixels never
   reach the trainer. The MoGe-2 depth + normal prior (`open4dgs.prior`) is ON by default (`--prior auto`): the
   segment's `moge2_prior` layer (masks' catalog/dataset; the local companion dataset when the source is the
   read-only Spark) is used when it matches the clip (window, cameras, sizes, ROI grid); otherwise MoGe-2 runs at
   start (monopriors' dynamic TensorRT engine, about 40 images/s at 1028 px on a 5090; the first run on a GPU builds
   the engine, about 90 s) and the maps are registered as that layer for the next run
   (replacing a layer of another grid; the file of each grid is kept). `--prior layer` requires a matching layer,
   `--prior off` trains without. Weights 0.05 / 0.05; the prior starts at 15 % of `--train.steps`, reaches full
   weight at 40 % and its normal term drops contradicting pixels from 75 %. No body mesh is needed (SelfCap trains
   with it); only the geometry metric uses the mesh. The run properties record `prior` (bool), `prior_mode` and `prior_source`.
5. `evaluate_cameras` scores held-out and training cameras (definitions below), and the held-out cameras' two floors.
   It also renders a 360-degree orbit.
6. The run recording `<runs-root>/<run-name>/run.rrd` (plus `trigs_ckpt.pt`) holds the cameras, held-out
   gt/render/error videos, per-frame metrics, the orbit video, the init cloud, the body mesh, loss curves and the
   `run`/`summary` properties. It is registered as a segment of `open4dgs-runs` on the local catalog.
   `/world` carries `ViewCoordinates` from the mean camera up. `--no-register` is the one switch that skips registration.

### What the held-out protocol shares with the held-out cameras

No held-out pixel reaches the trainer or the body-surface init. Two priors are shared, though: the SMPL-X body is
fitted by the dataset from ALL cameras (held-out ones included), and the mask model (BiRefNet, `people_matte` layer) runs
on every camera, and the held-out ground truth is `gt * mask`. So the held-out numbers measure novel-view quality
given a body and silhouettes that saw those views: a shared prior, not a pixel leak. The `init_roma` layer is built
without the held-out cameras (its `holdout` property) and `--init layer` checks that.

### Metric definitions (`open4dgs.evaluate`)

The ground truth is always `gt * mask` (soft mask; person boxes and pixel sets use mask > 0.5). The render is the
TRiGS render premultiplied over black, clamped to [0, 1].

| name | what |
|---|---|
| `crop_raw_{psnr,ssim,lpips}` | **headline**: the raw render against `gt * mask` inside the mask box padded 5 % per side; splats outside the true silhouette count |
| `crop_masked_{psnr,ssim,lpips}` | the same box, render also multiplied by the GT mask (hides silhouette errors; the old `crop`, +1.0 dB on 4DAnyone) |
| `fg_psnr` | PSNR of the raw render against the raw frame over the mask pixels only |
| `full_{psnr,ssim,lpips}` | whole frame, raw render against `gt * mask`; mostly black background, for paper comparison only |
| `alpha_iou` | IoU of rendered alpha > 0.5 with mask > 0.5 |
| floors `black`, `mean` | every metric above for an all-black render and for the mean GT colour inside the mask (alpha = mask), per held-out camera and frame (`/eval/holdout/<cam>/floor/<kind>/<name>`, summary `holdout_floor_<kind>_<name>`) |

Every score runs in full float32 (`evaluate.ieee_fp32`): PyTorch's default lets cuDNN convolutions use TF32, which
made LPIPS depend on the batch shape at the 1e-4 level. Training keeps PyTorch's defaults.

## Animated SOG export

`open4dgs-export-asog` encodes a run's checkpoint as Animated SOG into the recording `<run>/asog.rrd`
(`--asog-file` also writes an `.asog` for other players). The default preset `band-safe` keeps held-out
PSNR within 1 % of the uncompressed clip (about 12 MB); `--settings.preset small` gives a smaller file for web
streaming (upstream's encoder defaults, about 7 MB; `upstream` is an alias).

`open4dgs-export-splats` turns a run into a recording, layer of the run on `open4dgs-runs`. One `--format` flag:

- `asog` (default): the checkpoint encoded as Animated SOG (band-safe; or an `.asog` file given with `--asog`) stored
  once as `AnimatedSogSplats3D` plus a one-pose preview, `<run>/asog.rrd`, layer `asog`; the same recording
  `open4dgs-export-asog` writes (`--rrd`). The
  gsplat viewer plays it; a stock Rerun viewer shows one still pose.
- `per-frame`: one `GaussianSplats3D` per frame, `<run>/splats.rrd`, layer `splats`; plays in stock viewers.
- `trigs-motion`: the checkpoint once as `TrigsMotion3D`, `<run>/trigs_motion.rrd`, layer `trigs_motion`; the gsplat
  viewer only.
