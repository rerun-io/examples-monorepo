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
- **Code modifications**: yes — this is a **slimmed, streaming-inference-only**
  fork, no longer byte-for-byte upstream. It still loads every model variant
  (sam2 / sam2.1 hiera t/s/b+/l, efficienttam ti/s incl. 512×512) via
  `build_sam2_generic_video_predictor`, but the non-streaming predictor types and
  their builders are gone. Divergence derived from upstream commit
  `0691b7eea4514ef2216517f9902ed002f2a07653`, verified equivalence-preserving by a
  Hypothesis differential suite (record/replay vs the pre-slim code, ≥1000
  examples/seam, CPU seams bitwise-exact + a GPU end-to-end smoke under
  `torch.allclose`).
  - Deleted whole files: `sam2_video_predictor.py`, `sam2_video_predictor_legacy.py`,
    `sam2_image_predictor.py`, `automatic_mask_generator.py`, `utils/amg.py`,
    `benchmark.py`.
  - `build_sam.py`: kept only `build_sam2_generic_video_predictor` + `_load_checkpoint`
    (removed `build_sam2`, `build_sam2_generic`, `build_sam2_video_predictor`, the
    four `*_hf` builders, `_hf_download`, the HF id→filename map).
  - Removed dead in-fork code: `SAM2GenericVideoPredictorVOS` + the `vos_optimized`
    builder branch, and the training-only `Hiera.get_layer_id`/`get_num_layers`.
  - Kept (load-bearing): all model config YAMLs, the `hiera` + `vitdet` backbones,
    the `_C` connected-components extension, and the `SAM2ObjectMemoryBank`
    default-factory import.
  - Further YAGNI polish still possible (other dead in-`misc.py`/decoder branches).
- **Removed (repo-size only, no runtime impact)**: `demo/`, `notebooks/`,
  `sav_dataset/`, `assets/`, `training/`, `tools/`, `backend.Dockerfile`,
  `docker-compose.yaml` (~65 MB of web demo, example notebooks, and training
  code not used by the `mamma` package); the upstream FB docs (`README.md`,
  `INSTALL.md`, `RELEASE_NOTES.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`) and
  the dangling `configs/sam2.1_training/` finetune YAML (training code already
  removed; never composed on the inference path).
- **Install note**: `pyproject.toml` build-system requires `torch>=2.5.1` at
  build time → this package must be listed under `[pypi-options]
  no-build-isolation` in the root `pixi.toml`.
- **Known gotcha**: the EfficientTAM yaml configs declare
  `_target_: sam2.modeling.efficienttam_base.EfficientTAMBase`, which does not
  exist — EfficientTAM checkpoints only work through the `build_sam2_generic*`
  builders, which override `model._target_`.
