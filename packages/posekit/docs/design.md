# posekit design: network inventory, paradigms, and role taxonomy

Status: v3 analysis (2026-07-05). Phases 1–3 are implemented (see §7 and
docs/implementation-notes.html for per-milestone validation numbers); this
document records the component inventory across the monorepo and external
references, the inference-paradigm taxonomy, the role taxonomy the abstraction
must grow into, and the phasing.

Sources: deep-dives of mv-api, mamma (+sam2-streaming, sam3, sam3d-body);
papers — AssemblyHands-X (arXiv 2509.23888), EgoExo-Hands rig (arXiv
2510.02601), Kineo (arXiv 2510.24464, liris-xr/kineo); repos — roboflow/rf-detr,
Tau-J/rtmlib (+ our fork), facebookresearch/sapiens2 (full task suite),
facebookresearch/map-anything, mmpose projects/rtmpose + projects/rtmpose3d,
and HuggingFace transformers as a model source.

## 1. Why boxes and masks belong in a "keypoint" package

Every pipeline we care about is the same shape:

```
frames ─ detect ─ (track/segment) ─ 2D keypoints ─ [associate/triangulate] ─ [3D / fit]
```

- **mv-api** = rtmlib YOLOX → RTMPose/RTMW → DLT triangulation → temporal
  extrapolation (+ optional WiLoR hand refinement). All numpy, BGR uint8,
  GPU→CPU copy at every stage, no batching across cameras.
- **mamma** = YOLO12x (sparse re-detect) → SAM2/EfficientTAM streaming masks +
  CLIP/epipolar identity → MammaNet (needs crops **and masks**) → gated weighted
  DLT → SMPL-X sliding-window fitter (CUDA graphs).
- **Kineo** = RTMDet/YOLOX → RTMPose/DWPose/NLF (explicitly swappable) →
  keypoint-driven auto-calibration (BA) + weighted DLT; SAM2 for multi-person
  re-ID (their measured bottleneck: ~2.5 h where pose takes minutes).
- **AssemblyHands-X** = DWPose (COCO-133) + SAM silhouettes → weighted
  triangulation → SMPL-X fit with differentiable-silhouette loss.
- **EgoExo-Hands** = Sapiens-308 full-frame (doubles as the hand *detector*) +
  InterNet on fisheye→virtual-pinhole crops → RANSAC triangulation → LBS IK.

The 2D keypoint net is never alone: its inputs come from a detector or a
tracker/segmenter, and (MammaNet, AssemblyHands-X) masks are first-class model
inputs, not just visualization. So the abstraction's unit is the **role**, and
detection + segmentation are roles beside pose.

**The universal boundary datatype** (identical across all five pipelines):
per-`(view, frame, instance)` keypoints as `(x, y, confidence)` in a **named
skeleton convention**, plus a camera model. Confidence is load-bearing
everywhere (pair sampling, BA weighting, triangulation weighting, gating) —
never optional. This is what posekit's `Keypoints2d` already is; it needs to
stay the center of gravity.

## 2. Inference paradigms (top-down vs bottom-up vs single-stage)

Four genuinely different contracts exist; the role signatures must not blur
them, because they differ in *what conditions the keypoints*:

**Top-down** — keypoints are conditioned on an externally supplied box. A crop
is warped from the frame (`ops.crops`), the net sees one instance, decode maps
crop coords back to image space. Cost is O(#instances) forward passes (amortized
by batching crops). Examples: RTMPose/RTMW, RTMW3D, DWPose, ViTPose, Sapiens2
pose, MammaNet, WiLoR, SAM3D-Body, InterNet. This is posekit's
`TopDownPose2d(frames, detections)` — the estimator *can consume anyone's
boxes*, which is exactly what makes detector/tracker/pose independently
swappable.

**Bottom-up** — the net predicts *all* keypoints in the full frame (typically
heatmaps), then a *grouping* step assembles them into instances (OpenPose PAFs,
associative embeddings). One forward pass regardless of person count, but
grouping is brittle and the paradigm has lost ground; notably **rtmlib exports
no bottom-up model at all**. We don't plan a role for it until a model earns
one.

**One-stage dense (RTMO)** — YOLO-style: one full-frame pass over dense anchor
grids where each location predicts box *and* keypoints together, followed by
NMS. rtmlib's `RTMO` takes only an image and returns keypoints+scores (boxes
are internal, CPU NMS in rtmlib). Cost independent of person count; accuracy
below top-down on small people.

**Query-based single-stage (RF-DETR, ED-Pose)** — DETR: learned queries decode
box + keypoints (+ per-keypoint covariance in RF-DETR) jointly, NMS-free. The
keypoints are conditioned on *internal query embeddings*, so the model
**cannot consume external boxes** — it is structurally not
`PersonDetector ∘ TopDownPose2d`.

**Full-frame dense-skeleton (Sapiens-308 as used in EgoExo-Hands, NLF)** — a
*top-down-style* net run on the whole frame (or a person net so strong it
localizes) whose keypoints can *replace the detector* via a
boxes-from-keypoints adapter. Not a separate paradigm, but proof the
`PersonDetector` role must be an interface satisfiable by pose-net adapters.

**Consequence for the API**: the swappable *pipeline slot* is
"frames → posed instances". It is satisfied either by a
(`PersonDetector`, `TopDownPose2d`) pair — maximum mix-and-match — or by a
single `InstancePose2d` model (RTMO, RF-DETR) — maximum throughput. Adapters:
`InstancePose2d → PersonDetector` (drop keypoints) is trivial; the reverse is
impossible. Kineo's `RtmlibBboxKeypointsDetectionStage` already treats the
fused stage as the boundary, and rtmlib's `Body(pose="rtmo")`/`PoseTracker`
handle both shapes behind one call — precedent that the slot, not the pair, is
the abstraction.

## 3. Role taxonomy

Roles are the swap points. Runtimes (torch/onnx/tensorrt) stay orthogonal —
that layer is done and proven.

### 2D image roles (Phase 1 — implemented)

| Role | Signature | Implementations (repo) | Candidates (external) |
| --- | --- | --- | --- |
| `PersonDetector` | frames → `BoxDetections` | YOLOX-HumanArt ✅ | RTMDet, YOLO12x (mamma's), **RT-DETRv2** (HF, NMS-free, 640², clean tensor contract — read person class from `config.id2label`, never hardcode), D-FINE |
| `TopDownPose2d` | frames + detections → `Keypoints2d` | RTMPose/RTMW ✅, Sapiens2 ✅ | DWPose (= RTMPose-arch 133-kpt checkpoints, drop-in), ViTPose/ViTPose+ (HF weights + our GPU crops/UDP decode; `dataset_index` selects skeleton), NLF, InterNet (hands) |

### 2D roles to add (Phase 2-3)

| Role | Signature | Implementations | Notes |
| --- | --- | --- | --- |
| `InstancePose2d` | frames → boxes **+** keypoints (+covariance) in one pass | **RTMO** (rtmlib ONNX zoo, one-stage, needs GPU NMS port), RF-DETR keypoint-preview (TRT-fp16 ~10 ms, COCO-17 only, query-based NMS-free) | See §2 — satisfies the "posed instances" slot; cannot take external boxes. |
| `PromptableSegmenter` | image + prompts (boxes/points/text) → instance masks | SAM3 (`packages/sam3`; HF `Sam3Model` as weights source), SAM2 image mode | Feeds mask-consuming pose nets and silhouette fitting losses. |
| `VideoSegmenter` | stateful: `step(frames, prompts?) → {track_id: mask}` | SAM2.1 / EfficientTAM-ti via **our `sam2-streaming` fork** (causal memory-bank batching) | HF `Sam2VideoModel`/`Sam3VideoModel` are NOT viable here — stateful Python sessions, per-object loops, no batched causal memory; keep the fork. Kineo's measured bottleneck → highest-value TRT target. |
| `IdentityEncoder` | crops → embeddings | CLIP ViT-B/32 (mamma re-ID) | Small but real role; pairs with epipolar scoring in consumers. |
| `TopDownDenseLandmarks2d` | frames + detections **with masks** → dense landmarks (xy + log-variance + visibility + contact heads) | MammaNet (512 pts, torch + TRT-CUDA-graph already) | Same crop math as TopDownPose2d (`cv2` align); extra mask channel and richer per-point outputs. `BoxDetections` grows optional `masks`/`track_ids` so tracker output feeds it directly. |

### Human-dense roles (Sapiens2 non-pose suite — Phase 3.5)

Sapiens2 ships far more than pose, all full-frame but **human-centric**:
29-class body-part segmentation, surface normals, pointmaps (its "depth" is
just pointmap Z — there is no standalone depth checkpoint), matting (1B only),
and albedo (configs exist, no public weights). Sizes 0.4B/0.8B/1B/5B on HF
(`facebook/sapiens2-{seg,normal,pointmap,matting}-*`). These belong in
**posekit, not monoprior**: the boundary is *the domain the model represents*,
not the tensor shape — Sapiens2 normals/pointmaps are body-surface priors
trained on human data and they share posekit's detector/crop/mask ecosystem,
whereas monoprior owns general scene geometry.

| Role | Output | Notes |
| --- | --- | --- |
| `HumanPartSegmenter` | 29-class per-pixel body/apparel parsing | Not instance/promptable segmentation — distinct from `PromptableSegmenter`. |
| `HumanSurfaceNormals` | unit normal per pixel | Full-frame, aspect-preserving pad to 1024×768. |
| `HumanPointmap` | camera-space XYZ + scale per pixel | `HumanDepth` only as a derived Z view. |
| `HumanMatting` | fg RGB + alpha | 1B checkpoint only. |

All reuse Sapiens' existing exporter/runtime path (dynamo fp32 export → bf16
TRT, same crop-free full-frame preprocessing). ⚠️ Sapiens2 has a **custom
restrictive license** (no biometrics/re-ID/surveillance...) — treat as a
legal-review dependency, unlike Apache RTM/RF-DETR-nano-large weights.

### 3D roles (Phase 4)

| Role | Signature | Implementations | Notes |
| --- | --- | --- | --- |
| `TopDownKeypoints3d` | frames + detections → sparse 3D keypoints (xy image + root-relative z, optional camera-space xyz) | **RTMW3D** (mmpose rtmpose3d; 133-kpt 3D SimCC — x/y/z classification bins, root at kpts 11/12) | No official ONNX exists; the head is plain torch ending in 3 linear classifiers → export raw `simcc_x/y/z` and do GPU decode ourselves (exactly the posekit pattern; note upstream scores ignore the z axis). Prediction needs `z_root`, optional `xyz_camera` + explicit camera convention. rtmlib already has `RTMPose3d`/`Wholebody3d` (CPU) as reference decode. |
| `TopDownPose3d` (parametric) | frames + detections → `ParametricPose3d` (model params + joints3d + verts + weak-persp cam) | WiLoR (MANO, TRT runner exists), SAM3D-Body (SMPL-X/MHR), HaMeR | Keep separate from sparse 3D keypoints; align param stacking with simplecv's `ManoStack`/`SmplxStack`. |
| `SurfacePose3d` | frames → SMPL surface keypoints + uncertainty | NLF (Kineo's best 2D/3D source) | Surface points, not a fixed sparse skeleton. |

### Scene geometry: monoprior's role family (not posekit)

**MapAnything** settles the geometry-side abstraction: it wraps MoGe, VGGT,
DUSt3R-family, and itself behind ONE input contract (list of views: image +
optional intrinsics/poses/depth priors) and ONE output dict (metric pointmaps,
depth, recovered intrinsics, poses, confidence, scale). That is monoprior's
target shape, upgraded to posekit-style GPU dataclasses:
`MonocularGeometry` / `MultiViewGeometry` roles taking
`(frames, GeometryPriors?) → GeometryPrediction` — MoGe-2 implements monocular,
VGGT implements image-only multiview, MapAnything implements both *with
priors* (`facebook/map-anything-apache*` are Apache-licensed). Kineo-style
auto-calibration and metric scale should depend on `GeometryPrediction`, never
on a specific model. posekit ↔ monoprior boundary stays: human-domain models in
posekit (including Sapiens2 human pointmaps), scene-domain in monoprior.

### Explicit non-goals (stay in consumers / simplecv)

Triangulation (weighted DLT, RANSAC, gating), cross-view association, temporal
filtering/extrapolation, auto-calibration/BA, SMPL-X fitting, metric-scale
estimation. These are *geometry glue*, not networks — Kineo, mamma, and mv-api
each need different variants, and simplecv.ops already hosts triangulate.
(Small pure-tensor helpers shared by consumers — edge-margin confidence
modulation, per-view temporal median — can live in `posekit.ops` as composable
functions on `Keypoints2d`. rtmlib's fork-only `MultiviewBodyTracker` stays in
mv-api territory.)

## 4. Model sources (where weights come from)

Three source kinds, all feeding the same three runtimes:

1. **ONNX zoos** (OpenMMLab `onnx_sdk` zips — the rtmlib tables). Adopt
   rtmlib's zoo *as typed registry data*: URL, input size, skeleton, paradigm,
   runtime support. Its solution presets (`Body`, `BodyWithFeet`→halpe26,
   `Wholebody`, `Hand` = RTMDet-nano hand detector + hand21 RTMPose,
   `Wholebody3d`) become named posekit configs. What we replace (and already
   validated parity against): `BaseTool`'s CPU numpy/session.run per-bbox
   loops, untyped tuple returns, internal-only track ids.
2. **HuggingFace transformers as a weights + `nn.Module` source** — decision:
   option (a), NOT a fourth runtime backend and NOT their processors. Every HF
   image processor is CPU/PIL/numpy-bound (ViTPose's even uses scipy in
   decode); the model *modules* however have clean tensor contracts
   (`VitPoseForPoseEstimation: pixel_values → heatmaps`,
   `RTDetrV2: pixel_values → logits [B,300,C] + boxes cxcywh`). posekit
   adapters load `from_pretrained` weights into `TorchRuntime` and keep our GPU
   pre/post; ONNX export per-module is plausible but must be validated
   per-model (wrap to return plain tensors, pin `dataset_index`). We already do
   this implicitly for Sapiens. Version churn is contained in one adapter file
   per family.
3. **Local/partner checkpoints** (MammaNet safetensors, WiLoR ckpt, Sapiens2
   HF): torch module + our exporter, as with Sapiens today.

Tracking note: rtmlib's `PoseTracker` (sparse detector refresh +
boxes-from-previous-keypoints + greedy IoU ids) is a good *baseline algorithm*
to offer as a lightweight tracker in consumers, but its contract is exactly
what not to copy — ids are internal, the reorder logic is fragile. posekit
tracking output = `BoxDetections.track_ids`.

## 5. Prediction-type upgrades motivated by the survey

- **`Keypoints2d.uncertainty` (optional)** — MammaNet per-point log-variance;
  RF-DETR full 2×2 precision-Cholesky; Kineo BA and mamma triangulation consume
  them as weights. Scalar `scores` stays mandatory.
- **`BoxDetections.masks` + `.track_ids` (optional)** — detector-only, tracker,
  and segmenter outputs become one type; MammaNet-style consumers require the
  mask field.
- **`Keypoints3d` (new, sparse)** — for RTMW3D-class models: `xy` image-space,
  `z_root` root-relative, optional `xyz_camera` with explicit convention +
  root indices. Distinct from `ParametricPose3d`.
- **Skeleton registry growth** — sapiens-308 native, halpe-26, h36m-17, body-25
  (rtmlib ViTPose), and *anonymous dense formats* (mamma-512:
  `keypoint_names` optional/generated). Format-to-format projection tables
  (coco133→coco17, sapiens308→coco133, halpe26↔coco17) as registry data.
- **Camera-model-aware cropping (later)** — EgoExo-Hands' fisheye→virtual-
  pinhole rewarp as another analytic grid in `ops.crops` (simplecv
  `Fisheye62Parameters` exists).

## 6. What adoption looks like (the long-term goal)

### mv-api (biggest win, least new code)

Today every frame does: numpy BGR → rtmlib ONNX (per view, no batching) → CPU
copies → numpy triangulation. Swap sites are exactly two constructor lines and
two call lines in `multiview_pose_estimator.py`. Plan:
1. `MultiviewBodyTracker.__init__` accepts `detector: PersonDetector` and
   `pose: TopDownPose2d` (defaults preserve today's behavior via posekit's
   yolox/rtmw on the onnx backend — same weights, validated parity).
2. Batch all exo views into one `frames_rgb` tensor per tick → one detector +
   one pose call instead of `n_views` each; stay on GPU until triangulation.
3. TRT backend flip becomes a config flag. The catalog layer reuses the same
   tracker, so it inherits the swap for free.

### mamma

Already half-way there: `LandmarkEstimator` accepts a TRT engine, and posekit's
`TensorRtRuntime(use_cuda_graph=True)` subsumes `MammaNetTrtRunner` 1:1. Plan:
`MultiViewTracker.__init__` takes `detector` + `segmenter: VideoSegmenter` +
`identity: IdentityEncoder`; `LandmarkEstimator` takes a
`TopDownDenseLandmarks2d`. Then "run MAMMA with RF-DETR proposals" or "with
SAM3 masks" is config, not a fork.

## 7. Phasing

1. **Phase 1 — done.** Runtimes (torch/onnx/trt, GPU-resident), YOLOX, RTM
   family, Sapiens2 pose, skeletons, crops/decoders, parity-validated.
2. **Phase 2 — done.** `PromptableSegmenter` (SAM3, text prompts; box/point
   prompting deferred to SAM2 image mode), `VideoSegmenter` (sam2-streaming),
   `BoxDetections` mask/track fields, `IdentityEncoder` (CLIP). TRT for the
   SAM2 image encoder deferred until a measured multi-camera bottleneck.
3. **Phase 3 — done.** mv-api swap (batched, GPU-resident, role-injectable
   detector/pose; 0.3–0.6 px vs the rtmlib path); MammaNet as
   `TopDownDenseLandmarks2d` (adapter in mamma, bitwise-equal to
   `LandmarkEstimator`); RT-DETRv2 detector; ViTPose adapter (UDP end-to-end,
   0.75 px vs HF reference); typed zoo registry + presets (`posekit.zoo`).
   Outstanding: rtmlib Hand solution (RTMDet-nano NMS-strip untested).
4. **Phase 3.5 — Sapiens2 human-dense suite.** `HumanPartSegmenter`,
   `HumanSurfaceNormals`, `HumanPointmap`, `HumanMatting` on the existing
   Sapiens exporter path (license review first).
5. **Phase 4 — 3D + single-stage.** `TopDownKeypoints3d` (RTMW3D: own export +
   GPU 3D-SimCC decode), parametric `TopDownPose3d` (WiLoR → SAM3D-Body),
   `Keypoints2d.uncertainty`, `InstancePose2d` (RTMO first — zoo ONNX exists;
   RF-DETR when a wholebody checkpoint or license clarity lands).
6. **monoprior v2 (parallel track).** `MonocularGeometry`/`MultiViewGeometry`
   with `GeometryPriors`/`GeometryPrediction` GPU dataclasses; MoGe-2, VGGT,
   MapAnything(-apache) swappable behind them.

## 8. Reference notes (condensed)

- **Kineo** is the closest external analogue: Hydra-instantiated pipeline
  stages over an annotations blackboard; 2D pose explicitly swappable;
  confidence everywhere; they forked MMPose just to get "fully GPU-resident,
  batched keypoint inference" — the gap posekit fills. Calibration BA consumes
  the same 2D streams as capture.
- **rtmlib** is the best model-zoo curation in the space (body7/body8 COCO-17,
  halpe26, DW/RTMW cocktail13/14 wholebody-133, hand21, face106, RTMO, ViTPose
  ONNX, RTMPose3d) wrapped in exactly the runtime design we're replacing (CPU
  numpy, per-bbox loops, untyped tuples). Adopt the data, replace the engine —
  parity already proven for YOLOX + RTMW.
- **RF-DETR / RTMO** motivate the `InstancePose2d` slot (§2). RF-DETR adds
  per-keypoint covariance and a first-class ONNX→`trtexec` path; RTMO is
  available today from the OpenMMLab zoo.
- **Sapiens2** is a human-centric *foundation suite*, not a pose model —
  pose/seg-29/normals/pointmaps/matting share backbones, sizes, and our
  existing export path; restrictive license.
- **MapAnything** unifies scene geometry (wraps MoGe/VGGT/DUSt3R behind one
  views-in/geometry-out contract, optional priors, Apache variants) —
  blueprint for monoprior v2, and the counter-example that keeps posekit
  human-centric.
- **RTMW3D** provides single-view 3D wholebody keypoints via 3D SimCC with no
  official deploy path — posekit's export+GPU-decode approach is precisely
  what it's missing.
- **transformers** = weights source, not runtime: modules have clean tensor
  contracts, processors are CPU-bound; RT-DETRv2 in, Sam2Video out
  (sam2-streaming keeps the `VideoSegmenter` job).
- **EgoExo-Hands / AssemblyHands-X** motivate detector-from-pose adapters,
  fisheye-aware crops, multi-source triangulation, and mask-aware fitting.
