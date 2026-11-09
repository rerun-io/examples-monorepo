# EgoHand fast path on DGX Spark

`EgoHandStack.catalog()` is the EgoHand detector + pose stack calibrated on five
catalog datasets. It returns the same outputs as main's default `EgoHandDetectorConfig`
+ `EgoHandConfig`: boxes, anatomical sides, 2D/3D joints, full SOMA meshes, joint
rotations, identity, translation, scale and all 26 segment scales.

- Pose: first 8 DINO blocks NVFP4, last 24 FP8, FP8 attention, BF16 heads; batch 2.
- Detector: the three encoder input projections FP8, other convolutions FP16, decoding
  FP32; batch 1; a CUDA resize that matches the reference PIL input bit for bit.
- Calibration: `assets/egohand_catalog.rrd` (pose) and `assets/egohand_detector_catalog.rrd`
  (detector), 200 frames / 243 hand crops from segments disjoint from the evaluation
  segments, checked against every evaluation frame by decoded-RGB SHA256.

```python
from posekit.models import EgoHandStack

detector_config, pose_config = EgoHandStack.catalog()
detector, pose = detector_config.setup(), pose_config.setup()
detections = detector(frames_rgb)  # decoded uint8 CUDA frames, NHWC
reconstruction = pose(frames_rgb, detections, camera)
```

On the CLI, select `egohand-catalog` for both the detector and the pose role.

## GPUs without NVFP4

`EgoHandStack.catalog(backbone_precision="bf16")` (CLI role `egohand-catalog-bf16`) is the
fast path for GPUs below Blackwell (compute capability < 10: Ampere, Ada, Hopper). It runs the
same executor (TensorRT backbone and heads, CUDA crops, batched SOMA) with an unquantized BF16
backbone and an FP16 detector, and needs no calibration assets. The choice is explicit, not
automatic: the default NVFP4 preset raises a `ValueError` naming the BF16 preset when the GPU
lacks FP8 or NVFP4.

## Results

1,500 fixed frames: 3 segments × 100 frames from each of UmeTrack, HOT3D Quest 3,
SHOW3D, GSI six-view and Gen-HumanEgo. Batch one, detection on every frame, full outputs.
Latency excludes decode and engine builds.

| | Default (main) | `EgoHandStack.catalog()` | Speed-up | Accuracy bands |
|---|---|---|---|---|
| DGX Spark (GB10) | 153.31 ms/frame | 14.98 ms/frame | **10.23x** | 45/45 |
| RTX 3060, `backbone_precision="bf16"` | 238.62 ms/frame | 39.24 ms/frame | 6.08x | 45/45 |

Per dataset on the Spark: UmeTrack 8.46x, HOT3D 10.05x, SHOW3D 10.34x, GSI 10.55x,
Gen-HumanEgo 10.71x. The 10x is the mean over all frames, not a per-dataset claim.

An accuracy band is one metric on one dataset, relative to main's default: recall, mean
IoU and side accuracy must stay ≥ 0.95×; 2D median/p95 pixel error, absolute 3D
median/p95 and wrist-relative 3D median/p95 (mm) must stay ≤ 1.05×. Nine bands × five
datasets = 45. Detector matching is one-to-one by IoU ≥ 0.3; pose errors pool the
visible joints of side-correct matches. Worst band on the Spark: SHOW3D absolute 3D
median, +2.48 %. Main's detector also misses most UmeTrack hands (recall 9 %, raw
rotated cameras); the bands only show that the fast path does not lose more.

The fixed frames guided the precision choice; there is no second untouched test set.

## Reproduce

The cache holds camera-scoped native RRDs from the catalog (video, lens calibration, rig
pose, GT, projections): `<cache>/<dataset>/eval/*.rrd`. The team copy is
`dgx-spark:~/egohand-spark-cache`.

```bash
pixi install --locked -e posekit-hands   # on aarch64 the manifest pins cholespy's build deps

# Reference: main's default classes (accuracy only is enough for the bands).
pixi run --frozen -e posekit-hands posekit-hands-cached-bench \
  --cache ~/egohand-spark-cache --output ~/egohand-results/default --stack egohand

# Fast path: latency plus the 45-band check against the reference.
pixi run --frozen -e posekit-hands posekit-hands-cached-bench \
  --cache ~/egohand-spark-cache --output ~/egohand-results/fast \
  --baseline ~/egohand-results/default/results.json
```

On a GPU without NVFP4, add `--stack egohand-catalog-bf16` to the fast run. Run timings on an otherwise idle GPU. Each dataset writes a Rerun recording with the
source image, boxes, GT vs predicted joints, per-frame errors and latencies. The
precision trials and per-stage profiles are in the
[EgoHand speed-up report](https://pablos-4800gt.ilish-ruler.ts.net:8768/egohand-3060/).
