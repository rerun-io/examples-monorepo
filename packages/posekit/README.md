# posekit

One model API for human-centric perception networks — detection, segmentation,
tracking, re-ID, 2D pose, dense landmarks — over **PyTorch, ONNX Runtime, and
TensorRT**, with everything (video decode included) staying on the GPU.

It replaces the five near-copies of bbox→crop math, heatmap/SimCC decoders,
and hand-rolled TensorRT runners that lived in sapiens2-pose (now `posekit.models.sapiens`),
sapiens-coco133-pose, mamma, wilor-nano (now `posekit.models.wilor`), and rtmlib's CPU path.
EgoHand (formerly `packages/egohand`) lives here too; see [Hands](#hands-wilor-and-egohand). So does SAM 3D Body
(formerly `packages/sam3d-body`); see [Whole-body 3D](#whole-body-3d-sam-3d-body).

## How it fits together

Models are grouped into **roles** — the swap points of a pipeline. Every role
consumes the same canonical batch (uint8 RGB NHWC CUDA tensors, straight from
torchcodec's CUDA decoder) and returns GPU-resident predictions:

```mermaid
flowchart LR
    V["video / cameras"] -->|torchcodec CUDA decode| F["frames_rgb<br/>uint8 NHWC CUDA"]
    F --> D["PersonDetector<br/>YOLOX · RT-DETRv2"]
    F --> VS["VideoSegmenter<br/>SAM2-streaming, stateful"]
    F --> PS["PromptableSegmenter<br/>SAM3, text-prompted"]
    D -->|BoxDetections| P["TopDownPose2d<br/>RTMPose/RTMW · ViTPose · Sapiens2"]
    VS -->|"+ masks + track_ids"| P
    VS -->|masks required| DL["TopDownDenseLandmarks2d<br/>MammaNet"]
    D -->|BoxDetections| ID["IdentityEncoder<br/>CLIP ViT-B/32"]
    P -->|Keypoints2d| C["consumers<br/>triangulation · fitting · Rerun"]
    DL -->|DenseLandmarks2d| C
    ID -->|embeddings| C
    PS -->|"BoxDetections + masks"| C
```

Any `BoxDetections` producer feeds any top-down consumer — that's what makes
detector, tracker, and pose independently swappable (see `docs/design.md` §2
for why one-stage nets like RTMO/RF-DETR get their own `InstancePose2d` role
instead).

Inside every model class the backend is orthogonal: GPU preprocessing and GPU
decode are shared torch ops, and only the middle — a pure CUDA-tensors-in /
CUDA-tensors-out function — swaps between backends. Weights arrive from three
kinds of sources:

```mermaid
flowchart LR
    subgraph SRC["weight sources"]
        Z["OpenMMLab ONNX zoo"]
        E["our one-time export<br/>fp32 dynamo"]
        H["HF transformers modules<br/>RT-DETRv2 · ViTPose · SAM3 · CLIP"]
        L["local checkpoints<br/>MammaNet · Sapiens2"]
    end
    Z --> O["ONNX artifact<br/>the interchange format"]
    E --> O
    O --> ORT["OnnxCudaRuntime<br/>CUDA EP + IOBinding on torch data_ptrs"]
    O -->|"engine build, cached per GPU<br/>~/.cache/posekit/trt"| TRT["TensorRtRuntime<br/>static batch · optional CUDA graph"]
    H --> TCH["TorchRuntime<br/>nn.Module + autocast"]
    L --> TCH
    ORT --> RT["TensorRuntime contract"]
    TRT --> RT
    TCH --> RT
```

transformers is a **weights source, never a fourth backend**: posekit loads the
`nn.Module` and keeps its own GPU pre/post (the HF image processors are
CPU/PIL-bound).

## Quickstart

Backend choice is config, not code:

```python
from posekit.models import RtmPoseConfig, YoloxDetectorConfig
from posekit.runtimes import TensorRtBackendConfig

detector = YoloxDetectorConfig().setup()
pose = RtmPoseConfig(variant="rtmw-x-coco133", backend=TensorRtBackendConfig()).setup()

detections = detector(frames_rgb)          # uint8 NHWC CUDA in
keypoints = pose(frames_rgb, detections)   # Keypoints2d out, still on GPU
```

Or grab a named preset (`posekit.zoo` — rtmlib's solutions as typed data):

```python
from posekit.zoo import preset
pair = preset("wholebody")   # body / body-performance / wholebody / body-nmsfree / wholebody-fullframe
```

CLI (tyro subcommand unions — model and backend picked per stage):

```bash
# detector backend → pose model → pose backend
pixi run -e posekit --frozen posekit-video-pose --video-path video.mp4 \
    tensorrt rtmpose --variant rtmw-x-coco133 tensorrt

# same demo, transformers-sourced models
pixi run -e posekit --frozen posekit-video-pose --video-path video.mp4 rtdetr vitpose

# detect → SAM2 track → pose → CLIP re-ID, streaming (~38 fps end to end)
pixi run -e posekit --frozen posekit-video-track --video-path video.mp4
```

## Hands (WiLoR and EgoHand)

Two roles cover hands. `HandDetector` maps frames to `HandDetections` (boxes plus a
left/right side per box). `TopDownHandPose3d` maps frames, those detections and an
optional `CalibratedCamera` (`posekit.cameras`, a static `.rrd` via `load_calibrated_camera`) to
`HandPoses3d`: `Keypoints3d` with a hand skeleton, plus per-hand mesh vertices and
per-side faces. `HandPoses3d.as_hand21()` gives every model the same HAND_21 view.

| Model | Detector | 3D hands | Skeleton | Camera | Backends |
|---|---|---|---|---|---|
| WiLoR (`wilor`) | YOLO hand detector | WiLoR ViT + MANO | HAND_21 | not used (virtual camera) | torch fp16, TensorRT |
| EgoHand (`egohand`) | RT-DETR hand detector | DINOv3 + SOMA hand head | SOMA_HAND (25) | required (perspective crops undo the lens) | torch, TensorRT backbone/detector |

Both run in the `posekit-hands` environment. EgoHand's upstream source installs from
the pinned recipe `recipes/egohand-src` (patches and licence there) with upstream module
names, because the checkpoint's Hydra targets need them. WiLoR's network lives in
`posekit/models/wilor_net/`.

```bash
# Detect + reconstruct hands on a clip, log 2D boxes/skeletons and 3D joints/meshes to Rerun
pixi run -e posekit-hands --frozen posekit-video-hands --rr-config.headless --rr-config.save /tmp/hands.rrd \
    wilor tensorrt wilor tensorrt
pixi run -e posekit-hands --frozen posekit-video-hands --rr-config.headless --rr-config.save /tmp/hands.rrd \
    egohand onnx egohand

# Ground-truth eval on a fixed HOT3D (Quest 3) + UmeTrack subset from a dataforge catalog
pixi run -e posekit-hands --frozen posekit-hands-eval --catalog-url rerun+http://127.0.0.1:<port> --output-dir /tmp/hands-eval

# Per-frame speed, decoded GPU frame to joints + full mesh, per stage
pixi run -e posekit-hands --frozen posekit-hands-bench
```

Select `egohand-fast` for the opt-in TensorRT path on an NVFP4-capable CUDA GPU:

```bash
pixi run -e posekit-hands --frozen posekit-video-hands --rr-config.headless --rr-config.save /tmp/egohand-fast.rrd \
    egohand-fast tensorrt egohand-fast
pixi run -e posekit-hands --frozen posekit-hands-bench --models egohand egohand-fast
pixi run -e posekit-hands --frozen posekit-hands-eval --models egohand egohand-fast \
    --catalog-url rerun+http://127.0.0.1:<port> --output-dir /tmp/egohand-eval
```

`fast_egohand_configs()` returns the existing `EgoHandDetectorConfig` with GPU
preprocessing and an FP16 trunk, and an `EgoHandFastConfig`: its `FastEgoHandPredictor`
runs NVFP4 DINOv3 linears with the last eight blocks in FP8, FP8 attention, BF16
TensorRT heads, captured GPU crops, and full per-frame SOMA reconstruction. Identity, all 26 segment scales, joint
rotations, and every hand vertex remain dynamic. The ordinary `egohand` selection
retains its original defaults. SelfCap's existing calibration CLI selects these
same roles with `--oak-egohand-fast`.

First setup exports and builds missing artifacts through posekit and trtkit;
subsequent calls use their cache. `EgoHandFastConfig.fp8_tail_blocks` controls the
precision trade-off (0–32); both precision and calibration affect the export cache
key. The committed calibration (`assets/egohand_nvfp4.rrd`) uses 138 crops from disjoint SelfCap frames, with
mount-fit indices excluded. Its record includes exact frame indices and RGB
SHA256 hashes. Reproduce it from the catalog with:

```bash
pixi run -e posekit-hands --frozen posekit-egohand-calibrate \
    --catalog-url rerun+http://127.0.0.1:<port> --output /tmp/egohand-calibration.rrd
```

The calibration command also rejects every decoded benchmark frame by hash.
Set `EgoHandFastConfig.calibration` to use a separate calibration record. Quantization,
head export, and engine building are library code; no preparation scripts are required.

`catalog_egohand_configs()` is the five-dataset calibrated preset (CLI role `egohand-catalog`):
DGX Spark **153.31 → 14.98 ms/frame (10.23x)** with all 45 accuracy bands passing. See
[EgoHand fast path on DGX Spark](docs/egohand-spark.md) for the results and how to reproduce them.

**Eval protocol** (`posekit.apis.hands_eval`): three HOT3D Quest 3 segments (camera
`cam_00`) and three UmeTrack test segments (`cam_01`), every 10th frame, at most 40
frames each. Ground truth is dataforge's `hand_pose` layer (UmeTrack landmarks in HAND_21
order; the derived joint 1 is not scored); a hand counts when at least 15 joints are
valid. `gt-boxes` crops 1.2x the ground-truth joint box (the paper's protocol);
`detector` uses each model's own detector and scores hands matched at IoU >= 0.3 with
the correct side. Metrics follow the EgoHand paper (arXiv 2610.30293, Sec. 6.1): MPJPE
(wrist-aligned), PA-MPJPE, RTE (root translation, metric models only) in mm and 2D EPE
in px, each as mean / p90. It writes `results.json` and `hands_eval.rrd` (frames, ground
truth next to every prediction, per-frame errors as scalars).

EgoHand, HOT3D, `gt-boxes`, mean (mm / px):

| | MPJPE | PA-MPJPE | RTE | 2D EPE |
|---|---|---|---|---|
| ours (subset above) | 20.8 | 8.1 | 33.1 | 9.3 |
| paper, Table 2 (no smoothing) | 18.9 | 9.1 | 36.2 | 16.4 |

The paper does not name its HOT3D split, device or camera, fits SOMA to MANO joints for
ground truth and runs at another image resolution; treat the comparison as a sanity
check, not a reproduction. WiLoR has HOT3D in its training data.

## Whole-body 3D (SAM 3D Body)

`TopDownBodyPose3d` maps frames, person boxes (their optional `masks` act as prompts, weighted by
`scores`) and an optional `CalibratedCamera` to `BodyPoses3d`: MHR-70 joints (`MHR_70`: body, feet, both
21-joint hands) in image pixels and camera metres, the full 18,439-vertex MHR mesh, and the box each row
came from. `Sam3dBodyConfig` (zoo `sam3d-body-dinov3`) fills it with Meta's SAM 3D Body; its
`Sam3dBodyReconstruction` adds the MHR parameters, translation, focal length and the hand crops.
`mode="full"` runs the body pass, a mirrored left-hand and a right-hand pass and merges the hands into the
body (three DINOv3 passes per person); `mode="body"` runs the body pass alone. Without a camera the model
assumes its default field of view. Any `PersonDetector` or the SAM3 `PromptableSegmenter` feeds it.

`Sam3dBodyFastConfig` selects GPU crops, captured inference, compiled MHR and
calibrated TensorRT networks through the same role. See the
[fast-path configuration and validation guide](docs/sam3d-body-fast.md).

The upstream inference source (`sam_3d_body`) installs from the pinned recipe `recipes/sam-3d-body-src`
(patches and SAM License there); its DINOv3 comes from `recipes/dinov3-src`. The checkpoint `facebook/sam-3d-body-dinov3` is gated: request access,
then `hf auth login`. The model needs the `posekit-sam3d-body` feature; the apps run in `posekit-body`
(MoGe-2 camera and depth through monopriors, Open3D).

```bash
# Images (default: the four examples) or a video; SAM3 masks + MoGe-2 camera + SAM 3D Body -> Rerun
pixi run -e posekit-body --frozen posekit-body3d --rr-config.headless --rr-config.save /tmp/body3d.rrd
pixi run -e posekit-body --frozen posekit-body3d --video-path assets/examples/hands_video.mp4 --max-frames 48

# Gradio app with an embedded Rerun viewer, http://127.0.0.1:7871
pixi run -e posekit-body --frozen posekit-body3d-app

# Multiview demos on the HO-Cap sample (downloaded on first run, ~1.7 GB)
pixi run -e posekit-body --frozen posekit-body3d-mv-video     # SAM3 video masks + TSDF fusion
pixi run -e posekit-body --frozen posekit-body3d-mv-body      # per-camera bodies fused by MHR reprojection

# Per-frame speed (decoded GPU frame + person box -> full output), body and full mode, per stage
pixi run -e posekit-body --frozen posekit-body-bench
```

| `packages/sam3d-body` task (`-e sam3d-body`) | posekit task (`-e posekit-body`) |
|---|---|
| `sam3d-body-app` | `posekit-body3d-app` |
| `sam3d-body-cli` | `posekit-body3d` (also `--video-path`) |
| `sam3d-body-download-hocap-sample` | `posekit-body-download-hocap-sample` (to `data/hocap/sample`) |
| `sam3d-body-mv-video-demo` | `posekit-body3d-mv-video` |
| `sam3d-body-mv-body-demo` | `posekit-body3d-mv-body` |
| (none; images were not downloadable) | `posekit-body-download-examples` |

`tool/demo_mv_image.py` (no task; SAM3-only multiview segmentation of one frame) and
`tool/new_gradio_sam3.py` (it launched `packages/sam3`'s own UI) were not moved.

Goldens: `tests/test_body_golden.py` checks the role against references on the NAS
(`posekit/goldens/sam3d-body-v1`, README there). Full-mode outputs depend on earlier calls in the
process (a model's first full-mode call differs from later ones by up to 0.2 mm; other shapes and inputs
move later calls by up to ~1 mm), so full-mode tolerances are a few millimetres; body mode agrees to 1e-6.

## Click-to-track app (SAM2 + Rerun)

A Gradio app for single-object video segmentation: upload a clip, click the object in
the embedded Rerun viewer, refine on any frame, and propagate the mask through the clip.

```bash
pixi run -e posekit --frozen posekit-track-app            # http://127.0.0.1:7870
pixi run -e posekit --frozen posekit-track-app --port 7870 --variant efficienttam-s-512
```

The embedded viewer needs a secure context; on the tailnet expose it with
`tailscale serve --bg --https=7870 http://127.0.0.1:7870` and open
`https://<host>.<tailnet>.ts.net:7870/` (served at `/`, so `--root-path` stays empty).
Run it in a shell without `DISPLAY`: the app never spawns a native viewer.

Using it:

- **Click** the object in the viewer at the current frame (`+ Include`); `− Exclude` and
  `✕ Remove` add negative points / delete the nearest point. Pause the viewer before
  clicking — while it plays the click's frame is unknown. Scrubbing shows a memory-conditioned
  preview of the current mask.
- **Track** propagates from the first prompted frame forward, then backward to frame 0, and
  streams masks plus per-frame confidence traces. Clicking again after a track refines it;
  Track re-runs from the kept prompts.
- **Config** tab: model (`efficienttam-s-512` default, `-ti-512` faster), SAM2 memory window,
  point-removal radius, and "re-segment" (clicks replace the object instead of refining it).
  Model and memory window reload the clip and clear the points.
- **Outputs** tab: after a track, download a self-contained `.rrd` (video, prompts, masks,
  confidence). Masks are logged at 1/4 resolution under a `Transform3D(scale=4)` on
  both `video/mask` and `video/preview`; apply it when reading them back.

H.264 input passes through unchanged. HEVC input in MP4 or MOV is transcoded to H.264,
using the GPU when available, so browsers can decode it. Other codecs depend on Rerun
`Mp4Reader` support; MPEG-4 Part 2 (`mp4v`) is not supported and fails at load.

## What runs today, and how well

Every implementation is parity-validated against its reference:

| Model | Role | Backends | Validated vs | Result |
| --- | --- | --- | --- | --- |
| YOLOX (HumanArt) | `PersonDetector` | onnx · trt | rtmlib | boxes ≤ 4 px |
| RT-DETRv2 | `PersonDetector` | torch | HF processor | ≤ 0.31 px |
| RTMPose / RTMW | `TopDownPose2d` | onnx · trt | rtmlib | 1.3 px mean |
| ViTPose | `TopDownPose2d` | torch | HF processor | 0.75 px mean |
| Sapiens2 pose | `TopDownPose2d` | trt (default); torch · onnx · trt (`bf16-1024`) | bf16-1024 (d31, 2026-10-31); pre-port goldens | PCK@0.01 body/hands 0.911/0.971 (NVFP4); port bit-identical |
| SAM3 | `PromptableSegmenter` | torch | HF processor | mask IoU ≥ 0.997 |
| SAM2/EfficientTAM | `VideoSegmenter` | torch | — | stable 30-frame track, IoU ≥ 0.97 |
| CLIP ViT-B/32 | `IdentityEncoder` | torch | PIL pipeline | cosine ≥ 0.985 |
| MammaNet (in mamma) | `TopDownDenseLandmarks2d` | torch | mamma estimator | bitwise-equal |
| SAM 3D Body | `TopDownBodyPose3d` | torch | pre-port package | bitwise in its env; 3x run-to-run noise here |

Adoption: **mv-api**'s `MultiviewBodyTracker` runs on posekit roles (batched
per tick, 0.3–0.6 px vs its old rtmlib path, roles injectable via
`detector=`/`pose=`); **mamma** provides the MammaNet role adapter
(`mamma.landmarks.posekit_role`). Per-milestone numbers:
[`docs/implementation-notes.html`](docs/implementation-notes.html); full
inventory, paradigm taxonomy, and roadmap: [`docs/design.md`](docs/design.md).

## Predictions

Flattened across the frame batch with `frame_indices`, on the inference device,
numpy only at the Rerun/serialization boundary (`xy_numpy()`):

- `BoxDetections` — `xyxy`, `scores` (+ optional `masks`, `track_ids`) — one
  type for detector, tracker, and segmenter output.
- `Keypoints2d` — `xy`, `scores`, and its `KeypointSkeleton` (+ optional
  `uncertainty`), so consumers never guess what index 17 means.
- `DenseLandmarks2d` — xy + log-variance + visibility + contact heads.
- `Keypoints3d` — sparse 3D (image xy + root-relative z), for Phase 4.

## Gotchas

- **MMDeploy detector zips** bake `TopK(k=5000)`+NMS (batch-1 only, TRT rejects
  K > 3840). `artifacts.strip_detector_nms` cuts the graph before the TopK
  cluster; thresholding + torchvision NMS run on GPU instead.
- **Sapiens2 defaults to `nvfp4-768`**, a 768×576 TensorRT 0.4B engine with
  NVFP4 backbone linears, FP8 attention and head, and a dynamic 1/4/8 batch profile.
  Scores remain raw heatmap maxima. On the RTX 5090 d31 four-rig evaluation
  (2,880 people), it took 5.643 ms/person end-to-end versus 31.85 ms for the
  original engine; keep-agreement at 0.7 was 0.9807.
  Use `SapiensPoseConfig(variant="bf16-1024", backend=...)` for the original
  1024×768 path, including Torch and ONNX backends or larger checkpoints.
  Fast export caches include the calibration hash and export version.
  Regenerate ranges from person crop images with
  `pixi run -e posekit posekit-sapiens-calibrate --crops-dir <dir> --output <rrd>`,
  then review the .rrd before replacing `src/posekit/models/sapiens_net/fast_pose_768.rrd`.
  Sapiens fp16 TensorRT overflows; the original TensorRT path uses bf16.
- **Export outside dev mode** (`-e posekit`, not `-e posekit-dev`):
  `torch.onnx.export` tracing violates beartype hints in instrumented packages.
- **Compare pose backends on confident keypoints only** (score > 0.3) —
  low-confidence argmax noise inflates pixel errors ~100×.

## Adding a network

1. **Weights** — zoo ONNX (`fetch_openmmlab_onnx`), HF `from_pretrained`
   module, or a one-time fp32 dynamo export.
2. **Crops** — a `CropSpec` (size, `udp`/`cv2` align, BGR flag, mean/std);
   `ops.crops.crop_frames` does the batched `grid_sample` on GPU.
3. **Decode** — `decode_udp_heatmaps`, `decode_classic_heatmaps`,
   `decode_simcc`, or a new GPU decoder in `ops/`.
4. **Model class** — config dataclass with `setup()` implementing the matching
   role ABC; register it in the `posekit.models` tyro union and `posekit.zoo`.
