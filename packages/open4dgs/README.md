# open4dgs

Open4DGS in the monorepo: 4D Gaussian people trained from Rerun catalog segments with the TRiGS trainer.

- The trainer is upstream code: `recipes/open4dgs-trigs-src` installs `trigs` (ASU-NEM-Labs/open4dgs-trigs at
  `ed7a84a`) with two patches: PyTorch3D's `knn_points` / `sample_farthest_points` come from
  `open4dgs.pytorch3d_ops` (plain torch, so PyTorch3D is never built), and `build_canonical_from_npz` takes the
  keyframes arrays as a Mapping, so the init cloud is passed in memory.
- This package holds our code: catalog input, the in-memory hand-off to the trainer, priors (MoGe-2 depth and
  normals through `monopriors`), held-out evaluation and Rerun output.

## Environments

`open4dgs` (prod: training runs and timings) and `open4dgs-dev` (tests, lint, typecheck), linux-64 only: the
ai-demos gsplat 1.5.3 build (CUDA 13, torch 2.12, sm_86 + sm_120) exists only there. Both carry the Rerun catalog
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
