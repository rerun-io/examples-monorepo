# SAM 3D Body inference source

This is a build-only vendored dependency of posekit. The public API is
`posekit.models.sam3d_body` (`Sam3dBodyConfig` -> `Sam3dBodyPredictor`, a
`TopDownBodyPose3d`); the model dependencies are the `posekit-sam3d-body` feature
and the apps run in the `posekit-body` environment.

Source: [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
at `b5c765a`, package `sam_3d_body`, as kept in `packages/sam3d-body` (deleted in this
change) with type hints added. Only the inference path is included. The module is
named `sam3d_body`, as it was in `packages/sam3d-body`.

Changes from that copy:

- DINOv3 is built from the `dinov3` package (packages/dinov3-src, DINOv3 `5aa7e93`)
  instead of `torch.hub.load("facebookresearch/dinov3", source="github")`, which read
  unpinned GitHub main over the network. The architecture arguments are the hub
  factory `dinov3_vith16plus`'s. With SAM 3D Body's weights the two builds have
  identical parameters and buffers and bitwise-identical bf16 outputs on a
  3x3x512x512 input; neither has a `mask_token` in the checkpoint, and inference does
  not read it.
- `BaseModel` is a plain `torch.nn.Module`; PyTorch Lightning only carried training
  loggers and `save_hyperparameters`. `utils/logging` drops the rank-zero gate;
  `utils/checkpoint` drops the Lightning checkpoint callback; `utils/dist` keeps only
  `recursive_to`.
- Removed (inference never imports them): `models/meta_arch/base_lightning_module.py`,
  the ViT backbones (`models/backbones/vit.py`, timm and flash-attn), the training
  optimizer code in `models/optim` except `fp16_utils` (the bf16 backbone
  conversion uses it), `data/utils/io.py`, and the duplicate top-level `transforms/`.
  Our Rerun demos, Gradio UI and multi-view optimizer moved into posekit
  (`posekit.apis.body3d*`, `posekit.body_*`).

Upstream behaviour kept on purpose (not fixed here):

- The hand decoder always runs in full mode; its `hand_logits` are computed and
  never read.
- `SAM3DBodyEstimator` hard-codes `DEFAULT_INPUT_SIZE = 512`.
- Full-mode 2D keypoints are projected about the image centre, also when the given
  intrinsics have another principal point.
- Crops are cut on the CPU with `cv2.warpAffine`, so their pixels depend on the OpenCV
  version (4 and 5 interpolate differently); the goldens record the versions they ran with.

Weights download on first use into the Hugging Face cache, at a pinned revision:
`model.ckpt` (sha256 `b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf`),
`model_config.yaml` and `assets/mhr_model.pt` from the gated repo
`facebook/sam-3d-body-dinov3`, revision `11aaa346c7204874a1cbafe3d39a979080b2c55a`.
`Sam3dBodyConfig.checkpoint_dir` overrides the download.

Licence files: `LICENSE` is upstream's SAM License (b5c765a, last updated November 19, 2025); `LICENSE-APACHE` and `LICENSE-MIT` moved with the code from `packages/sam3d-body`.

Goldens: `/mnt/nas/ssd-8tb/posekit/goldens/sam3d-body-v1` (README there);
`tests/test_body_golden.py` compares the role with them.
