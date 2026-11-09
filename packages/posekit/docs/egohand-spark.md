# EgoHand on Spark and Ampere

This experiment compares the full detector and pose pipeline with main
`f742736f9` on fixed native catalog frames. The reference uses
`EgoHandDetectorConfig` and `EgoHandConfig` defaults. The existing `hands_bench`
`egohand` preset uses the accelerated SelfCap stack, so it is a different
reference. Both paths receive the same decoded RGB frame on CUDA and return
boxes, anatomical sides, 2D/3D joints, full SOMA meshes, joint rotations,
identity, translation, global scale, and all 26 segment scales.

The primary device is DGX Spark GB10 (Blackwell, aarch64). RTX 3060 (Ampere)
uses the BF16 fast pose executor and supports FP16 detector inference. It has
no FP8 or NVFP4 tensor cores. `EgoHandFastConfig.backbone_precision` selects
`bf16`, `fp8`, or `nvfp4`; FP8 uses all 32 DINO blocks in FP8. NVFP4 retains
`fp8_tail_blocks` final blocks in FP8 and can protect `fp8_prefix_blocks` initial blocks. BF16 does not load quantization ranges.

## Measured presets

```python
from posekit.models import catalog_egohand_configs

# Spark: 8 NVFP4 + 24 FP8 pose blocks; 3 detector projections FP8.
detector_config, pose_config = catalog_egohand_configs()

# Ampere: FP16 detector and BF16 fast pose, without quantization assets.
detector_config, pose_config = catalog_egohand_configs(backbone_precision="bf16")

detector = detector_config.setup()
pose = pose_config.setup()
detections = detector(frames_rgb)  # one decoded uint8 CUDA frame, NHWC
reconstruction = pose(frames_rgb, detections, camera)
```

The ordinary detector/pose Tyro unions expose paired `egohand-catalog` and
`egohand-catalog-bf16` selections. Select the same preset for both roles; the
detector's nested backend is `tensorrt`. Existing `egohand` and `egohand-fast`
defaults remain unchanged. The preset uses detector batch 1 and pose batch 2,
matching the measurements. It does not skip detection or remove output fields.

## Fixed data and metrics

The read-only source is `rerun+http://pablo-dl-server:51235`. The cache contains
native camera video, lens calibration, rig transforms, GT, projections and QA
from UmeTrack, HOT3D Quest 3, SHOW3D, Gen-HumanEgo, and GSI six-view. Evaluation
uses three segments per dataset, with 100 uniformly spaced eligible frames per
segment: 1,500 frames total. One separate segment per dataset provides 40
calibration frames, yielding 200 frames and 243 detected hand crops. The
calibrator rejects segment overlap and decoded RGB SHA256 overlap with every
evaluation frame. Calibration ranges and source identities are RRD records.

A frame is eligible if a hand has a visible wrist and at least 15 valid joints.
The derived thumb-base landmark at index 1 is excluded. Selection depends only
on GT. The cache reader aligns timestamps within 2 ms and uses the original
camera model. Gen-HumanEgo requires the added CPU/CUDA double-sphere adapter;
this compatibility change is shared by the reference and fast paths. SHOW3D
supplies independent measured 2D labels, retained as supplied.

Detector matching is greedy one-to-one descending IoU, with IoU >= 0.3.
Recall counts side-correct matches with a pose. Mean IoU is the best box overlap
per GT hand, including zero for misses. Side accuracy is measured over IoU
matches of either side. Pose errors pool valid joints of side-correct matches:
median/p95 source-pixel error, absolute camera-space millimeter error, and
wrist-relative millimeter error. Both 3D forms must pass, giving nine bands per
dataset. Higher-is-better metrics must be >= 0.95 times reference; errors must
be <= 1.05 times reference. Missing or nonfinite results fail.

## Reproduce

Install with Pixi in a host terminal. On aarch64, the manifest supplies
nanobind < 3 to build cholespy 2.2 without isolated build dependencies.

```bash
pixi install --locked -e posekit-hands

# The offline cache layout is <cache>/<dataset>/{eval,calibration}/*.rrd.
# Omit --assets to use pinned Hugging Face downloads and the trtkit cache.
pixi run --frozen -e posekit-hands posekit-hands-cached-bench \
  --cache ~/egohand-spark-cache --assets ~/egohand-spark-assets \
  --output ~/egohand-spark-results/default-clean --allow-idle-residents

pixi run --frozen -e posekit-hands posekit-hands-cached-bench \
  --cache ~/egohand-spark-cache --assets ~/egohand-spark-assets \
  --output ~/egohand-spark-results/nvfp4-fp8 \
  --model catalog --precision nvfp4 --allow-idle-residents

pixi run --frozen -e posekit-hands posekit-hands-cached-calibrate \
  --cache ~/egohand-3060/cache --assets ~/egohand-3060/assets \
  --output ~/egohand-3060/new-calibration
```

Pixi tasks run from `packages/posekit`. The catalog preset loads its calibration
RRDs from the installed package; manual `--model fast` experiments can override
`--calibration`, detector scope and precision counts. Direct Python configuration uses
`catalog_egohand_configs(assets)` for the measured preset. The benchmark uses detector min/opt/max batch 1 and
pose max/opt batch 2. It performs detection on every frame.

FP8 detector experiments set `EgoHandDetectorConfig.fp8_calibration` and
`fp8_scope`. `all` quantizes 57 convolutions; `backbone` quantizes 20 residual
backbone convolutions while retaining the stem and neck in FP16.
`late_backbone` quantizes 5 final-stage convolutions. The accepted `projections`
scope quantizes 3 encoder input projections; remaining convolutions stay FP16. The source
ONNX hash must match calibration. The cache key includes calibration and
scope. The detector's CUDA resize matches PIL's two integer rounding passes
bit for bit; removing the CPU transfer preserves the model's input pixels.

## Timing and evidence

Each segment first runs an untimed accuracy/evidence pass, which warms its
camera, shapes and hand counts. Two later passes measure synchronized wall
time with internal stage profiling disabled. A separate pass profiles the
detector, crops, backbone, heads and SOMA. Decode, engine builds, calibration,
scoring and Rerun logging are outside primary latency. Any inference-time
transfers and full mesh computation remain inside it. Dataset means weight
frames equally, including zero-hand frames.

By default a timing run requires no other compute process. On the shared
Spark, `--allow-idle-residents` permits residents only below 5% total GPU
utilization before the run. It observes 30 seconds of idle per-process
activity, samples `nvidia-smi pmon` every second and device telemetry every
500 ms throughout the run, and rejects any external engine activity above
that process's measured idle envelope. It never stops another process.
`gpu-audit.json` must say `accepted: true` before using a latency result.
Raw idle/process/device samples are retained. Accuracy-only experiments use
`--no-measure-timing` and produce no latency claim.

Each dataset RRD contains exact source indices, boxes, 2D/3D GT and prediction,
sparse source images and model crops, full meshes and model parameters,
matched-hand error scalars and stage timings. View these in Rerun. The
integration evidence contract uses `EGOHAND_BENCH_EVIDENCE=/path/to/result.rrd`.
Pixel validation uses an owned headless viewer through Rerun MCP; stop that
viewer before timing on its GPU.

## Measured checkpoints

The precision comparison uses the same 1,500 frames and full outputs:

| Spark configuration | Mean ms/frame | Speed-up | Accuracy |
| --- | ---: | ---: | ---: |
| Default unquantized path | 153.308 | 1.000x | Reference |
| BF16 fast pose + FP16 detector | 19.508 | 7.859x | 45/45 bands |
| FP8 pose + FP16 detector | 15.538 | 9.867x | 45/45 bands |
| NVFP4/FP8 pose + mixed FP8/FP16 detector | 14.981 | 10.234x | 45/45 bands |



Stage means use a separate instrumented pass, in milliseconds per frame:

| Spark path | Detector | Crops | Backbone | Heads | SOMA | Other + assembly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 20.434 | 8.145 | 55.370 | 6.604 | 32.582 | 29.168 |
| BF16 fast | 3.703 | 0.380 | 13.805 | 0.791 | 0.738 | 0.201 |
| FP8 fast | 3.729 | 0.378 | 9.650 | 0.804 | 0.754 | 0.199 |
| NVFP4/FP8 fast | 3.460 | 0.371 | 9.416 | 0.796 | 0.743 | 0.196 |

The fast backbone remains the largest cost. The 3060 fast path spends
4.929 ms in the detector, 0.565 in crops, 31.981 in the backbone, 0.897 in
heads, 0.789 in SOMA, and 0.280 elsewhere. Its accepted default runs measured
only total latency; no default per-stage figure is claimed. Profiled sums
need not equal the separately measured end-to-end means.

The full evolving visual report and raw audit links are at
<https://pablos-4800gt.ilish-ruler.ts.net:8768/egohand-3060/>.

At 2026-11-09 07:36 UTC, the primary Spark target is met:

| Dataset | Default mean ms | Fast mean ms | Speed-up | Accuracy bands |
| --- | ---: | ---: | ---: | ---: |
| UmeTrack | 60.320 | 7.129 | 8.46x | 9/9 |
| HOT3D Quest 3 | 164.777 | 16.395 | 10.05x | 9/9 |
| SHOW3D | 172.389 | 16.680 | 10.34x | 9/9 |
| GSI six-view | 174.939 | 16.577 | 10.55x | 9/9 |
| Gen-HumanEgo | 194.112 | 18.123 | 10.71x | 9/9 |
| **Mean over 1,500 frames** | **153.308** | **14.981** | **10.234x** | **45/45** |

The accuracy criterion is relative to main. UmeTrack recall is only 54/595
(9.08%) for main and 55/595 (9.24%) for the accepted fast path. All frames,
including missed hands, remain in the timing sample. Passing these bands
does not establish high absolute detector accuracy.

The accepted mix uses 40 NVFP4 and 120 FP8 DINO linears, FP8 attention,
3 FP8 detector feature projections, remaining detector convolutions in FP16,
FP32 detector decode, and BF16 pose heads. The largest relative loss is
SHOW3D absolute 3D median +2.48%; the limit is +5%. The fixed evaluation frames guided precision selection; there is no second
untouched test set or whole-corpus accuracy claim. This does not claim that
every detector operation runs in FP8 or that every frame is 10x faster.

Two combined timing attempts were rejected for resident activity rises.
The final result retains the first audit-approved run per dataset: UmeTrack
attempt 2, all other datasets attempt 1. Each still uses the full fixed
300-frame sample, two isolated timing passes and a separate stage pass.
All five accepted audit files are linked in the report. Rejected attempts
remain on disk under the experiment results directory.

The secondary 3060 result is **238.619 -> 39.239 ms/frame = 6.081x**, with
all 45 bands passing. It uses the FP16 detector and BF16 fast pose. The
candidate audit accepted 849 process samples, zero spikes, and 1% pre-run
utilization. Each default dataset was independently audited; all first
attempts passed. There is no 10x requirement on the 3060.

Rejected accuracy trials include full detector FP8 (grayscale recall loss),
20- and 5-convolution residual detector scopes (UmeTrack pose p95), and
NVFP4 pose with 8, 12 or 16 final FP8 blocks or a first-4/last-8 FP8 mix.
The all-FP8 pose plus FP16 detector remains an accepted fallback at
15.538 ms/frame, 9.867x, with all bands passing.

For gates on this host, disable optional analytics for the command only:

```bash
GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1 \
  pixi run --frozen -e posekit-dev gate
GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1 \
  pixi run --frozen -e posekit-hands-dev gate
```

A faulthandler trace showed Gradio telemetry HTTP connection threads blocking
interpreter shutdown after all unit tests passed. These variables disable
analytics, not tests or runtime type checks. Final checks on pablo-ubuntu:

- `posekit-dev gate`: lint/typecheck/deadcode pass; 149 unit and 18 integration
  tests pass, with 17 and 24 skips respectively.
- `posekit-hands-dev gate`: lint/typecheck/deadcode pass; 181 unit and 39
  integration tests pass, with 15 and 13 skips respectively.
- Lock check passes; all 424 repository structure tests pass.
- Real-model public preset smoke runs pass on Spark and Ampere. Rerun MCP
  screenshots were read for all five final dataset recordings, including
  full meshes/crops and detector misses. Report layouts were checked at
  1400x1000 and 390x844; all metric, audit, image and RRD links were checked.

Optional asset skips remain explicit; existing typecheck warnings remain.
The integration fixtures select BF16 on Ampere and release each large model
before constructing the next. They retain full-output and graph-reuse checks.
The BF16 comparison also uses independently audited dataset runs;
Gen-HumanEgo needed one retry.
