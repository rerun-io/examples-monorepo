# Vendored: cjaverliat/sam2

- **Upstream**: https://github.com/cjaverliat/sam2 (fork of facebookresearch/sam2, 41 commits ahead)
- **Pinned commit**: `0691b7eea4514ef2216517f9902ed002f2a07653` ("Add wheel build CI", 2025-11-29, `main`)
- **Why this fork**: streaming video inference without whole-video preload
  (`SAM2GenericVideoPredictor.forward(state, frame_idx, frame)` — one frame at a
  time, per-video state externalized in `SAM2GenericVideoPredictorState`),
  pluggable memory banks (`sam2/modeling/memory.py` `ObjectMemoryBank` ABC,
  `sam2_memory.py` default, `sam2_forgetful_memory.py`
  `SAM2ForgetfulObjectMemoryBank` sliding-window forgetting — bounded VRAM for
  arbitrarily long videos), and EfficientTAM support via `build_sam2_generic*`
  builders + the ViTDet backbone (`sam2/modeling/backbones/vitdet.py`,
  `sam2/configs/efficienttam/*.yaml`).
- **Code modifications**: none.
- **Removed (repo-size only, no runtime impact)**: `demo/`, `notebooks/`,
  `sav_dataset/`, `assets/`, `training/`, `tools/`, `backend.Dockerfile`,
  `docker-compose.yaml` (~65 MB of web demo, example notebooks, and training
  code not used by the `mamma` package).
- **Install note**: `pyproject.toml` build-system requires `torch>=2.5.1` at
  build time → this package must be listed under `[pypi-options]
  no-build-isolation` in the root `pixi.toml`.
- **Known gotcha**: the EfficientTAM yaml configs declare
  `_target_: sam2.modeling.efficienttam_base.EfficientTAMBase`, which does not
  exist — EfficientTAM checkpoints only work through the `build_sam2_generic*`
  builders, which override `model._target_`.
