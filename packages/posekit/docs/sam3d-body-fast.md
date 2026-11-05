# SAM 3D Body fast path

`Sam3dBodyFastConfig` implements the same `TopDownBodyPose3d` role and returns the
same complete `Sam3dBodyReconstruction` as `Sam3dBodyConfig`. The eager config
remains the default. Select the fast config explicitly:

```python
from posekit.models import Sam3dBodyFastConfig

body = Sam3dBodyFastConfig().setup()
people = body(frames_rgb, detections, camera)
```

The input is a decoded uint8 RGB CUDA frame batch, `BoxDetections`, and an
optional `CalibratedCamera`. Outputs stay on CUDA. The role supports person
masks, multiple people, multiple frames and both `full` and `body` modes.
`body.mhr_model` supplies differentiable MHR geometry to the multi-view fitter.

The same subcommand is available in the image/video CLI, Gradio app and
multi-view body demo:

```bash
pixi run -e posekit-body --frozen posekit-body3d --rr-config.headless --rr-config.save /tmp/body-fast.rrd sam3d-body-fast
pixi run -e posekit-body --frozen posekit-body3d-app sam3d-body-fast
pixi run -e posekit-body --frozen posekit-body3d-mv-body sam3d-body-fast --rr-config.headless --rr-config.save /tmp/body-fast-mv.rrd
pixi run -e posekit-body --frozen posekit-body-bench --output /tmp/body-fast-bench.json sam3d-body-fast
```

## Execution and artifacts

The fast path uses the shared EgoHand `CapturedGraph` owner and shared trtkit
export, QDQ and TensorRT runtimes. It prepares affine crops on CUDA, runs the
body backbone first, then batches the dependent left/right hand crops. The
learned decoder layers run in BF16 TensorRT. Intermediate MHR evaluations skin
only vertices needed by the keypoint mapping; the final evaluation reconstructs
all 18,439 vertices. Pose correctives, identity, scale, hand fusion and camera
translation are recomputed for each input.

CUDA graphs are keyed by input shape, person count, mask/camera presence and
whether stage profiling is enabled. Graph owners retain their input, output
and runtime storage. Returned estimates own their tensors, so later replays do
not overwrite earlier results. A predictor runs sequentially; its shared graph
pool is not a concurrent inference API. Batches above the configured capture
limit use the same GPU pipeline without whole-frame capture.

The first setup exports missing artifacts and builds the local TensorRT
engines. Source weights are pinned to `SAM3D_BODY_REVISION`. The quantized
backbone filename includes the measured calibration, complete per-module
precision plan, FP8 tail length and source-export version. The source preserves
the eager model's BF16 tokens, normalization and residual arithmetic. Engines
use trtkit's normal machine-local cache. Setup, export,
engine build and graph warmup are excluded from steady-state timing.

`backbone_onnx=None` keeps the original Torch backbone;
`decoder_onnx_dir=None` keeps the original learned decoder layers.
With the Torch backbone selected, `compile_torch_backbone=True` compiles it
while retaining BF16 intermediate rounding. The whole-pipeline graph remains
the sole capture owner. Ordinary fusion without those rounding boundaries
failed the OAK accuracy band in validation.
`backbone_precision="bf16"` runs the unquantized source backbone through
TensorRT, with the same decoder layers, geometry and graph execution. On the
CLI, append `--backbone-precision bf16` after `sam3d-body-fast`.
`capture_pipeline=False` retains the per-decoder capture path for comparisons.
These are config choices; callers do not need backend-specific branches.

`backbone_plan` can override individual linear or attention modules.
`hand_backbone_plan` applies additional overrides for the hand passes, which
use their own TensorRT context and artifact. A separate hand plan requires GPU
preprocessing; the CPU-crop comparison path rejects it. Module plans require
the quantized TensorRT path. To mix BF16 with quantized modules, use BF16
overrides in the plan; the wholly unquantized and Torch paths reject plans
that they cannot apply.

## Calibration and accuracy

`posekit-body-calibrate` measures the 160 linear-input and 96 post-RoPE Q/K/V
ranges on body and both hand crops at 512 pixels. The Python API also accepts
`CatalogCalibrationSource` entries for other captures/datasets and separate
local generic images or videos. Existing accuracy reports and explicit
held-out inputs can extend the excluded evaluation set.

Both source/frame identity and decoded RGB SHA256 are checked for overlap
before activation measurement. `Sam3dBodyFastConfig.calibration` selects an
explicit record; the checkpoint revision must match. Keep calibration broad
across cameras, activities and datasets. A pass on a single phone capture does
not establish accuracy on other sources.

`posekit-body-accuracy` compares complete eager and fast role outputs on the
same boxes and pixels. It reports 2D keypoints, pelvis-relative 3D keypoints,
hand-root-relative keypoints, camera translation and pelvis-relative mesh
vertices. Every real-person source must satisfy each median/p95 band. Declare
detector-false-positive stress sources before evaluation; they remain in the
report as a separate table. Repeated eager inference is available to measure
its own variation. `evaluation_report` reuses an earlier acceptance report's
exact frame indices and hashes. Use `development_frames` for a separate
selection set: the loader rejects index or decoded-pixel overlap with either
calibration or acceptance. Development reports retain that label and cannot
be replayed as acceptance reports. The evaluated model always uses the
specified calibration.

Save worst-frame comparisons with `evidence_dir`, then inspect the resulting
RRDs in Rerun. Real-person and stress outliers are saved separately. Keep
translation and full-mode hand-merge threshold flips visible when comparing
precision choices. Run accuracy again after every approximation; measure the
same end-to-end benchmark rather than inferring speed from component timings.
