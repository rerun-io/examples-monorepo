# EgoHand inference source

This is a build-only vendored dependency of posekit. The public API is
`posekit.models.egohand`, and the runtime feature is `posekit-egohand`.
Use `pixi run -e egohand egohand-selfcap --rr-config.headless --rr-config.save /tmp/egohand/selfcap.rrd`.
It reads only the fixed SelfCap episode and caches one 0–6.2 second query.
Inference defaults to 2–6 seconds. It never registers a catalog layer.

Sources:

- IsaacCapture `479063c`, `src/python/isaaccapture_egopost`.
- DINOv3 `5aa7e93ab42ae3526291f6b4e3fcd22637e03326`.

Only the inference model, camera/crop, mirror and smoothing dependencies are
included. DINO training, dataset loaders, body fitting and video exporters are
excluded. `egohand_inference` and `egohand_dinov3` are private import namespaces.
Training methods still embedded in the model class are retained for checkpoint
compatibility; this package does not expose a training command.

Changes from source: remove unused global chumpy shims; remove file exporters
and full-body wrapper methods; rename internal imports; support OpenCV 4 and 5
iterative point undistortion; add CUDA-event SOMA timing, GPU perspective sampling and prepared-crop input. The model
math and checkpoint configuration remain unchanged. The posekit adapter owns
new typed code; the vendored dependency remains an explicit source boundary.

Required local assets under `/tmp/egohand/weights`:

- `egohand/egohand_s060000.ckpt`, repo `nvidia/EgoHand-v1`, revision
  `a153616c5192e6a8b09dbd302f682093080c5b83`.
- `detector/hand_detector.onnx`, repo `nvidia/RT-DETR-Hand-Detector-v1`, revision
  `55572ddc93a5916c3e486811dc6e62bb2b7bdb9d`.
- `soma/`: `SOMAHand.npz`, `SOMA_neutral.npz`,
  `SOMA_procedural_transforms.json`, `SOMA_template_rig.usda`,
  `MHR/mhr_model_lod1.pt`, `MHR/base_body_lod1.obj`, `MHR/SOMA_wrap_lod1.obj`,
  repo `nvidia/SOMA-X`, revision `f81a01796f652260ff930d5318d6c6b845a505ed`.

Download with `hf download REPO FILE... --revision REV --local-dir DEST`.
NPZ files are read-only upstream model assets; generated predictions use Rerun.
The inference extras use the workspace PyTorch and ORT versions. Pillow <13
and python-dateutil <3 are scoped to the runtime feature, avoiding upstream
pins that conflict with Gradio and s3fs. Only basic Pillow image operations
are used. No other environment's dependency constraints change.

## Checkpoint integrity

- EgoHand checkpoint SHA256:
  `f940ecb6e3b73d5bd9e562cff232550e1c072f6c063eb4f705951122b15d8a75`
- Detector SHA256:
  `85d36428cc1ecf8832ca2d48f7aa04df0659cf94dde358352fb3cf651a319810`

`--compare-sapiens` runs full-frame Sapiens2 0.4B on every selected frame.
The report includes per-joint median/p90 errors against Quest, with capture-time
clock conversion, 48 ms hand latency, interpolated OAK pose and all 14 lens terms.
Both models use identical finite joint slots. Six of Sapiens2's 21 hand slots are
unmapped in the production adapter; these are reported as missing, never filled.
Low/high crop-sharpness thirds provide a blur proxy, not manual blur labels.

Raw RT-DETR class names are not anatomical identities. The initial port treated
them as identities and consequently mirrored the wrong crops. The adapter now
uses upstream candidate deduplication and geometry/IoU assignment before any
crop mirroring. It retains all confident candidates and preserves the selected
raw boxes: no new smoothing, gap filling, or padding changes are introduced.
Two-hand bootstrap assumes an upright, unmirrored egocentric view; subsequent
identity uses continuity. A lone bootstrap detection still relies on its class
label.

The correction is supported by an untouched upstream detector comparison:
preprocessing is bit-identical, raw boxes/classes agree, and swapping RGB/BGR
or reflecting the input does not repair identity. The supplied upstream
assignment resolves the reversed candidates; regression tests check reversed
labels, already-correct labels, and both hands under one class. The numeric
class-name table remains unchanged. Sides are resolved before selecting the
left-canonical/right-mirrored SOMA reconstruction path.

All 2D coordinates are distorted source pixels. The 3D view uses OAK camera
metres. SOMA timing is nested within forward/post-forward and is not additive.

The optional dev lane carries a four-entry pyrefly baseline for existing
Transformers `.to()` decorator-stub errors in unchanged posekit models. The
EgoHand adapter itself has no baseline entries. AnyIO <4.15 respects the shared
typing_extensions hold; no dependency constraints in other features were changed.

## Batched accelerated inference

The segment runner uses cached TensorRT engines for the released RT-DETR
ONNX detector and the DINOv3 backbone. DINO is exported with bf16 autocast;
RT-DETR keeps its fp32 graph and permits TF32 tactics. Both have dynamic batch
profiles. Pillow resizes each detector batch in parallel without changing its
pixels. Detection runs in groups of eight frames, followed by upstream
identity assignment over the ordered sequence. Reconstruction runs sixteen
frames (up to 32 hand crops) per call. `--batch-frames` controls this grouping.

The public configuration also retains the eager reference path. Select GPU
crops with `EgoHandConfig(gpu_crops=True, batch_size=32)` and supply
`backbone_onnx=Path(...)` to use TensorRT. If that artifact is absent, setup
exports it from the supplied checkpoint. Use a separate artifact path for
each checkpoint. `backbone_backend` controls the engine profile and cache.
`EgoHandDetectorConfig.backend` selects ONNX Runtime or TensorRT independently.

GPU crops retain the full 14-coefficient OAK lens model, including its physical
projection domain. CPU work computes the small box-perimeter inversion; Warp
projects the raster on CUDA, then Torch fuses uint8 sampling, mirroring and
normalization. Sampling follows the environment's OpenCV 5 float-coordinate
interpolation. The CPU crop path remains available for reference comparisons.
Run through Pixi so Triton can locate the environment's CUDA assembler. The
dev environment keeps the same sampling math eager so all local type checks
remain active; production fuses the sampling and normalization kernels.

SOMA still reconstructs **all mesh vertices**, joints, local rotations,
identity coefficients, global scale, per-finger scales and camera translation.
Batching reduces its call overhead. CUDA events measure SOMA without a
synchronization around each call. Reported SOMA time is nested in forward and
post-forward time; do not add it again. The remaining CPU output conversion
preserves the upstream camera and mirroring conventions.

Warm throughput excludes checkpoint loading, ONNX export, engine build and
first-call Torch/Warp compilation. Keep those startup costs separate. Whole
segment throughput includes RGB decoding, detection and complete mesh-producing
prediction, but excludes Rerun serialization and catalog transfer.
