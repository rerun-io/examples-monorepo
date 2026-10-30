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
iterative point undistortion; add opt-in synchronized SOMA timing. The model
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
label. `assign_identity=False` exposes raw candidates for diagnostics.

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
