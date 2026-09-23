# SHOW3D observations and layer mapping

## The dataset and its papers

SHOW3D is a hand-object interaction dataset from Meta Reality Labs and Yale:
Rim et al., *SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild*,
CVPR 2026 ([arXiv 2603.28760](https://arxiv.org/abs/2603.28760), project page
[show3d-dataset.github.io](https://show3d-dataset.github.io/), loader and
challenge kit at
[patrickqrim/SHOW3D-dataset-api](https://github.com/patrickqrim/SHOW3D-dataset-api),
CC BY-NC 4.0). Its predecessor is Rim et al., *Ego-Exo 3D Hand Tracking in the
Wild with a Mobile Multi-Camera Rig* (HANDS @ ICCV 2025,
[arXiv 2510.02601](https://arxiv.org/abs/2510.02601)), whose small hands-only
release is called EgoExo-Hands. Same rig, pipeline, and authors; SHOW3D is the
scaled successor with object poses and captions, not a re-release.

**Capture.** An 8 kg back-mounted rig carries eight monochrome fisheye cameras
(1024×1280, 152°×116°) in a half-dome, and a Meta Quest 3 on the head contributes
two more; all ten are hardware-synchronised at 60 Hz. Five OptiTrack cameras on
the rig track a marker tree on the headset, the HOT3D method, which is where the
headset pose comes from. The reference frame moves with the participant: the
world in these recordings is the back rig, so overlays are exact within a frame
but a static point in the room drifts across frames. The Hub ships **undistorted
pinhole** video (`DistortionModel: PinholePlane`, no coefficients) cut from those
fisheye sensors; the rectification is undocumented and our intrinsics imply
roughly 100° horizontal field of view, so quote the Hub calibration, not the
paper's 152°.

**How the labels were made.** Nothing is hand-curated. Hands: Sapiens and
InterNet detect 2D keypoints on per-hand virtual pinhole crops in all views,
RANSAC triangulation fuses them, and inverse kinematics fits a per-subject
UmeTrack skinning model built from a hand scan (`profile_umetrack.json`). The
released confidence is **per hand**, a product of a triangulation term and the
IK residual; the Hub's zero rules (hand untracked, headset pose rejected, face
blur overlap) are applied at release. Median error is 5.7–7.9 mm against a
30-camera dome and against manual clicks, with a 90th percentile up to 16 mm;
hand-object interaction is the worst case. Objects: CNOS detection, FoundPose
coarse pose and GoTrack refinement with multi-view gPnP against the HOT3D CAD
models; it is a tracker with re-initialisation, so failures are temporally
correlated and near-symmetric objects can flip. Captions on the Hub come from
Qwen3-VL 235B over the video, not from the paper's instruction paraphrases.

**What ships and what does not.** Labels exist only for the 1,689 train
recordings (32 subjects); the 448 test recordings (6 subjects) have video and
captions only. `hand_pose/v2` supersedes v1, whose landmarks were mis-scaled.
The paper's masks and contact maps are not released. Depth is described on the
Hub card as MapAnything-derived prediction and is not in the tree. MANO is
mentioned once in the paper and never fitted; the hand model is UmeTrack.

**Consequences for our layers.** Per-joint confidences in `coco133_xyz` are the
per-hand scalar broadcast to that hand's slots. Headset poses flagged
`is_synthesized` are interpolated, not tracked; we log them with the flag on
`/world/rig_01` rather than dropping them. Frames with confidence 0 still carry
`joint_angles`, and the hand layer keeps those rows. Frame `index` is the join
key across every source file, so the video re-encode must keep the frame count
exactly. A full research digest with sources and open questions is kept at
`show3d-papers-research.html` under the fleet artifacts.

## Source layout and revision

`facebook/show3d-dataset` has two `dataset_index_{train,test}.parquet` files.
Each scene is `scenes/<subject>/<scene>/`: camera MP4s,
`metadata/{recording_info,frame_info}.json`, `camera_calibration/<camera>.json`,
and `blur_info/<camera>.mp4.json`. Annotation trees use the same subject/scene
keys under `hand_pose/v2`, `object_pose/v1`, and `captions/v1`. Hand profiles
are shared per subject at `hand_pose/hand_profiles/<subject>/profile_umetrack.json`.

`download` fetches both indexes, all profiles, and the 22 mapped HOT3D BOP meshes. `discover` attaches `split`
to each typed index row, unions both splits, and orders object scenes first,
then train, then test, with subject/scene as a stable tie breaker.
`--sequences` selects scene IDs or subject/scene keys; `--split` narrows the
union. `--no-object-scenes-first` selects ordinary train/test order.

The local index snapshot has 1,689 train rows (468 object scenes) and 448 test
rows (no object scenes). Camera counts across the union are 2,088 with ten,
48 with nine, and one with eight. Seven train rows have zero frames, all from
AZH822: `none_clap-hands_a702`, `none_finger-tapping_36f9`,
`none_finger-weaving_414a`, `none_hand-roll_8a14`, `none_hand-squeeze_86c6`,
`none_typing_bcd6`, and `none_wave-hands_c6fc`. Discovery prints and skips these
rows. Once fetched, each scene's recording/frame metadata takes precedence
over the index census. A fetched scene without metadata fails with a scene-specific error.

Each recording records the Hub commit it was built from as
`property:capture:hf_revision`; the corpus run pins one with `--revision`.
Without an explicit revision, the default branch resolves once per run to a
cached commit SHA. Every SHOW3D Hub fetch uses that SHA. HOT3D BOP downloads resolve their own repository SHA.

Conversion plans the union of missing layers' inputs, then fetches absent files
with one fetch round (serial per-file downloads). Index camera and annotation flags select the
files; scene metadata validates the census at the read boundary. `--force`
rebuilds layers but never downloads raw files that are already present.
Only a call that writes base removes source MP4s after all requested layers publish successfully unless
`--keep-raw` is set. JSON, profiles, indexes, and annotations remain available
for later layer builders. Temporary encoded MP4s live beneath `<root>/work/`
and are removed even after failure. Each layer skips its own existing file unless
`--force` is set. Missing annotation layers rebuild from retained JSON without
fetching MP4s or reading the base recording. A hand-only rebuild reads metadata,
hand JSON, and the subject profile; it needs no calibration or blur sidecars.
Publication and execution order is base → hand_pose → captions → object_pose → object_mesh → hand_mesh.

## Frames, clocks, and calibration variants

The source's “World” is the back-mounted rig frame. It moves with the subject;
it is not a fixed physical-world frame. Root coordinates are right-handed Y-up.
A later SLAM layer can place a transform on `/world` without reparenting data.
All source translations are millimetres and become metres. Rotation matrices
must be finite, orthonormal, and proper (determinant +1).

Rig calibrations have one static `T_WorldFromCamera`. Headset calibrations have
`T_WorldFromCamera_by_index`, keyed by decimal string index. Legacy entries
have `index`, `agt_frame_id`, `timestamp`, `T_WorldFromCamera`, and
`is_synthesized`. New entries also carry `pose_source` and `is_pose_valid`,
with optional top-level `pose_contract_version`. Missing transforms produce
no transform row. Synthesized and validity flags are preserved independently;
optional provenance columns are emitted only where supplied. Unknown JSON
fields are allowed at these third-party boundaries.

Headset0 is the moving rig origin. Headset1 uses the mean relative transform
from paired, non-synthesized, valid poses; the mean rotation is orthonormalized.
The baseline is about 63.8 mm. Conversion rejects coordinate translation
standard deviation ≥1 mm or maximum rotation deviation ≥0.5°, naming the scene.

| Scene | Translation std, max coordinate (mm) | Maximum rotation deviation (degrees) |
| --- | ---: | ---: |
| SPI102/keyboard_toss-away_83ef | 0.0000190678396 | 0.00000615148252 |
| LYA722/birdhousetoy_shaking_8eca | 0.0000125903791 | 0.00000545803590 |

`video_time = round((timestamp - first_timestamp) * 1e9)` is a duration clock.
`frame_index` is the upstream sequence index. Every temporal BASE column has
both timelines. No stream is resampled. The capture properties preserve the
source start timestamp and frame ID to reverse the time shift.

## Video measurement and encoder decision

On 2026-09-19, the RTX 5090 host measured 20 evenly spaced frames per camera
(200 samples per scene). Values below are video-only RRD MB / median grayscale
PSNR dB / wall seconds. The builtin column is Mp4Reader's own AV1 transcode; the CQ columns are the
shipped file-input path (`transcode_mp4_gray`, one scene = ten cameras, three at a time). Source
sizes are decimal MB.

| Scene | Frames | Source MB | Builtin AV1 | CQ 28 | CQ 32 | CQ 36 |
| --- | ---: | ---: | --- | --- | --- | --- |
| keyboard_toss-away_83ef | 586 | 62.4 | 53.8 / 44.61 / 9.9 | 93.2 / 45.58 / 9.6 | 60.1 / 44.19 / 9.5 | 37.2 / 43.05 / 9.5 |
| birdhousetoy_shaking_8eca | 1002 | 127.4 | 117.5 / 44.77 / 15.0 | 188.7 / 45.43 / 12.1 | 128.0 / 44.03 / 12.1 | 87.6 / 42.58 / 12.0 |

`VIDEO_CQ = 36`: this produces the smallest tested files whose median PSNR is
at least 40 dB on both scenes, at 0.60–0.69 times source size. GOP is 60 and
B-frames are disabled. The raw report is
`data/show3d-video-measurements.json` (ignored by git).

The converter decodes each source in ffmpeg (`format=gray`) and encodes AV1 NVENC in the same
process; against the earlier PyAV-pipe prototype this is MD5-identical output at 2.2 times the
speed, and three concurrent camera encodes add another 1.6–1.7 times. A full 10-camera conversion of
these scenes takes 8–10 s wall through `dataforge-convert` on the RTX 5090. Logging and deletion of
each completed clip overlap the remaining encodes, and every sidecar is validated before the first
encode.

The comparison is a plain Python CLI; no dedicated Pixi task is added:

```bash
cd packages/dataforge
DATAFORGE_FFMPEG=/home/pablo/.pixi/bin/ffmpeg \
  pixi run -e dataforge-dev --frozen python tools/apps/show3d_measure.py
```

The tool measures only CQ 28/32/36 with the same `transcode_mp4_gray`
file-input primitive as conversion; the builtin column was measured once through `Mp4Reader` and is
kept for comparison. Reference frames are decoded
once per scene, and each RRD is streamed once for scoring. PSNR decoding is
outside the timer. Exact matches use an MSE floor of 1e-12 (168.13 dB).
Codec, GOP, and CQ are static camera AnyValues.

## Source quirks

`LYA722/birdhousetoy_shaking_8eca`, `blur_info/rig0.mp4.json`, frame 255 has
an inverted box `[454.64, 438.24, 452.30, 436.15]`. This roughly two-pixel
interpolation artefact is normalized to min/max corners at load time. Each
sidecar counts normalized boxes, and conversion reports the scene total
(13 in the birdhouse scene).

The local index has 49 nondegenerate scenes without rig0. Camera indices stay fixed: rig1 remains camera 1,
and the back-rig census counts only present exo cameras. Dataset-wide
blueprints retain all eight rig panes.

## Mapping to Rerun

| Source | Rerun destination |
| --- | --- |
| Back rig | `/world/rig_00`, name `back_rig`, kind `exo`, no rig transform |
| rig0…rig7 | `rig_00/cam_00`…`cam_07`, fixed indices even if a camera is missing |
| Quest 3 | `/world/rig_01`, name `quest3`, kind `ego`, two cameras |
| headset0 pose | Temporal `Transform3D` on `rig_01` |
| headset0 / headset1 extrinsics | Identity / constant stereo transform on `rig_01/cam_00` / `cam_01` |
| Pinhole intrinsics | Camera `/pinhole`, frustum length 0.05 m |
| MP4 | Camera `/pinhole/video`, AV1 `VideoStream` on both clocks |
| Calibration text | Static `source_calibration_json`: full rig JSON; headset intrinsics only (poses and flags have their own tracks) |
| Frame metadata | `/frames`: `source_frame_id`, `source_timestamp_s`, `missing_cameras` (typed strings, including empty lists) |
| Headset provenance | Temporal `is_synthesized`, optional `pose_source` and `is_pose_valid` on `rig_01` |
| Blur xyxy pixels | Camera `/pinhole/boxes/face`, partitioned `Boxes2D` with class id 103 (`face`) and static `source="blur_info"`; empty rows retained when supplied (schema §13: named for what the box encloses, not why it was drawn) |
| BASE census | Group `capture` (`schema=dataforge:v1`, plus `convert` group): int64 `num_frames`, `num_cameras`, `num_synthesized_headset_poses`, `source_start_frame_id`; float64 `source_start_time_s` |

The default blueprint has the prototype's 3D eye, headset L/R panes, and a
2-column × 4-row rig grid, with an instruction text pane below the ego pair. Blur boxes are excluded by default. Headset views
include `/world/**` so later 3D annotations project into them. The table card
includes only headset0 video.

### Annotation layers

| Source | Layer / destination |
| --- | --- |
| COCO-133 names and connections | `hand_pose`: static `/` AnnotationContext, one class, ID 0, "Coco Wholebody" |
| `landmarks_3d_mm` | `/world/gt/coco133_xyz`: dense 133-point Points3DWithConfidence rows in metres, class 0, static COCO keypoint IDs, per-point confidence colours; placed only where the hand's confidence is > 0.5 (the Hub README default) |
| `joint_angles` | Hand `/joint_angles`: 22 float32 values per available row |
| Wrist rotation and translation | Hand `/wrist`: world-from-wrist Transform3D, translation in metres |
| Confidence | Hand `/confidence`: Scalars on every frame, including zero |
| `landmarks_2d` | `/world/rig_01/cam_0{0,1}/pinhole/coco133_uv`: dense 133-point Points2DWithConfidence rows from shipped pixels; null points, absent hands, and hands at confidence ≤ 0.5 become NaN with zero confidence |
| Subject profile JSON | `/world/gt/hands/profile`: verbatim static TextDocument with `application/json` media type |
| Caption JSON (all ten strings) | `captions`: static Markdown TextDocument at `/task/instruction`; overall caption first, other fields as a definition list |

The root `AnnotationContext` (COCO-133 skeleton and the §13 box classes) is
written by base, so every layer resolves its classes without another one loaded.
UmeTrack landmarks use the Assembly-Hands index order and are
mapped into COCO-133, including body wrists and interpolated thumb bases. Other
body and face points remain NaN with zero confidence. Each placed hand carries
its shipped confidence; an absent hand, or one at confidence ≤ 0.5, has NaN positions
and zero confidence in the COCO stack. The Hub README sets `confidence > 0.5` as the
default threshold and calls `> 0` "low-quality frames you usually want to drop"; at 0.04
the shipped landmarks float over empty floor (`bbq_pouring-out_5d8a`, frame 138). The
per-hand `/confidence` stream keeps the shipped value on every frame, so nothing is lost
for a consumer who wants a different cut.
`landmarks_3d_mm_local` is not logged.

A rectified pinhole pane shows the viewer projection of `coco133_xyz` and hides
`coco133_uv`. A camera with distortion would show `coco133_uv` and exclude
`coco133_xyz`, because Rerun Pinhole cannot project through distortion. SHOW3D
cameras are all `PinholePlane`, so only the rectified rule applies. The world
view also excludes every camera's `coco133_uv`.

Every temporal annotation row has the base `video_time` and `frame_index`.
The hand frame census must equal scene `recording_info.num_frames`, and frame
IDs/timestamps must agree with `frame_info`. Keypoint and confidence rows are
dense on this clock. Joint angles and wrist transforms produce rows only where
present; confidence-zero records may still have joint angles. Hand JSON uses a
partial pyserde schema; profiles decode the full typed UmeTrack model, with
unknown envelope fields allowed. Profiles are stored without reserializing them.

Annotation layers use `send_properties=False` and write only their own property
groups. `hand_pose` holds string `version=v2` and float64
`coverage_left_high_conf` / `coverage_right_high_conf` (confidence > 0.5).
`captions` holds string `version=v1`, `hand` and `overall_caption`. The index facts a
catalog user filters on (`subject_id`, `split`, `object_alias`, `action`) are the
`episode` group in base: they describe the source, not a layer. The alias is the
scene ID's first token; action is everything between alias and final hash. The
base census remains in `capture`.

The keyboard and birdhouse reprojection goldens read the written hand landmarks and UV, then
project through the base sidecar camera chain (headset0 pose, fixed stereo
camera transform, and pinhole intrinsics). Each hand/camera pair must have median
finite-point error below 0.5 px. The mesh goldens also compare skinned landmarks with the shipped world landmarks (see below).

## Object and mesh layers

| Source | Layer / destination | Time and properties |
| --- | --- | --- |
| `object_pose/v1/.../object_pose.json` | `object_pose`: `/world/gt/objects/<alias>` Transform3D, translation in metres | Both clocks, only confidence > 0; `/confidence` Scalars on every frame |
| HOT3D BOP stripped GLB | `object_mesh`: object `/mesh` Asset3D + temporal `albedo_factor` and `Scalars` on `/mesh` | Static blob; alpha 1 where confidence > 0.5 (the Hub README default), alpha 0 otherwise (invisible instead of held at the last pose or shown while shaky), rows only where visibility changes; the shipped confidence repeated as Scalars on every frame; int64 `mesh_id`, string `mesh_source=bop-benchmark/hot3d` |
| Hand JSON and full subject model | `hand_mesh`: `/world/gt/hands/{left,right}/mesh` Mesh3D | Static triangles and RGBA albedo (alpha 110); one row per frame on both clocks: world vertices in metres where a wrist exists and confidence > 0.5 (the Hub README default), an empty vertex row otherwise; no properties |

Object records use a partial pyserde schema. Confidence-zero records can have
empty `R` and `t` lists; positive confidence requires finite 3×3 proper rotation
(det=+1) and 3×1 translation. `vertices_world_space` is ignored. Census and
frame identity/time must agree with the base metadata. Confidence-zero rows
remain in the confidence signal; all posed rows remain in the transform signal.

The object pose stream is sparse, so the viewer's latest-at would keep the static mesh
at its last pose through every unposed frame (about a quarter of frames per scene; 67% in
`LWA828/bbq_pouring-out_5d8a`). The `object_mesh` layer therefore logs a temporal
`Asset3D.albedo_factor` on the `/mesh` entity: opaque white where confidence > 0.5 (the same
Hub default the hands use; the object README says 0.5 "cuts most failures without throwing
away usable data"), fully transparent elsewhere. Rows exist only where visibility changes;
latest-at carries them, and the static blob plus a temporal colour on one entity is ordinary
Rerun. `Clear` cannot serve here (a cleared parent drops the static mesh at the rig origin)
and a scale of 0 warns and still draws. The parent pose stream stays exactly as shipped. The
mesh entity also repeats the shipped confidence as `Scalars` on every frame, so selecting the
mesh in the viewer, or querying its entity, shows the value behind the alpha.

The `object_pose` property group contains string `version=v1` and float64
`coverage`, `in_ego_fov_fraction`, and `palm_dist_median_m`. Coverage is the
fraction of all frames with a pose. FOV is the fraction of posed frames whose
object origin is in front of and inside either headset pinhole. Palm distance
is the median, over frames with an object pose and at least one world palm,
of the distance from the object origin to the nearest landmark 20, in metres.
Undefined metrics use float64 NaN. These are census measurements, never row filters.
Conversion reuses the base `Scene` clock and headset calibrations. For rebuilds
without base, `read_headset_calibrations` applies the same clock
agreement check, without reading videos, rig calibrations, or blur files.

Hand meshes use the full pyserde `HandModelNumpy` through the `HandProfile`
envelope (float32 geometry and int64 indices). `wrist_for_hand` mirrors the
right hand; `skin_mesh` runs in batches of 256 frames. The model and wrist
remain in millimetres until skinned vertices are converted once to metres.
Left is blue, right is peach. A trusted wrist without joint angles is an input error,
not a silently dropped row. The source ships a wrist and joint angles on many frames
it marks with confidence 0 (the tracker lost the hand; in `LWA828/bbq_pouring-out_5d8a`
the right hand carries a wrist on 377 of its 503 confidence-0 frames). `hand_pose`
keeps those rows verbatim. Low-confidence frames are worse than absent ones: at
confidence 0.04 (`bbq_pouring-out_5d8a`, frame 138) the shipped left-hand landmarks
float over empty floor in both headset images. The Hub README says to use
`confidence > 0.5` by default and calls `> 0` "low-quality frames you usually want to
drop", so landmarks are placed and the derived mesh is skinned only above 0.5; the mesh writes
an empty vertex row on every other frame, so latest-at never holds a stale mesh where no
hand is. `hand_pose` remains the only AnnotationContext owner.
The existing `/world/**` blueprint filter includes both object and hand meshes.

### Object-frame verification

The full sample recordings produce:

| Scene | Posed / all frames | In either ego FOV | Median nearest palm (m) |
| --- | ---: | ---: | ---: |
| SPI102/keyboard_toss-away_83ef | 560 / 586 | 0.0 | 1.40923017 |
| LYA722/birdhousetoy_shaking_8eca | 999 / 1002 | 1.0 | 0.11027603 |

The birdhouse mesh sits in the right hand. The keyboard track is outside both
headset images on every posed frame despite positive confidence; the wide view
shows the keyboard far from both hands. A full depth census corrects the earlier
prototype claim that it is always behind the headset: 458/560 posed centres
are behind headset0 and 469/560 behind headset1; the remaining centres still
project outside the images. The sanity properties expose this bad track without
altering source poses.

Pixel evidence from Rerun 0.37.0, headless Vulkan llvmpipe, with all seven layers merged:

- [Birdhouse, frame 801](https://pablos-4800gt.ilish-ruler.ts.net:8768/show3d/pr4/birdhouse-all-layers-f4.png)
- [Keyboard, frame 117](https://pablos-4800gt.ilish-ruler.ts.net:8768/show3d/pr4/keyboard-all-layers-f1.png)

These snapshots show the camera images, projected skeletons, translucent hand
meshes, object meshes, and instruction pane together.

The skinning golden requires <0.01 mm for both hands in both scenes. The
birdhouse palm-distance band is <0.2 m; the measured value is 0.11027603 m
over 850 object/palm pairs. Keyboard bounds and both FOV bounds are unchanged.

Storage was measured from the two full-scene `hand_mesh.rrd` files: their combined
file bytes divided by their temporal Mesh3D row count give about 9.52 KB per
posed hand-frame. The combined files are about 8× the corresponding `hand_pose`
files. Extrapolating bytes per scene-frame from these two samples to the
3,465,942 frames in 1,682 nonempty hand scenes gives about 60 GB; this is an
estimate, not a full-corpus measurement. Consumers can leave `hand_mesh`
unregistered; a later change can coarsen its clock.

### BOP names, renumbering, and texture extension

`download()` uses `transports.hf_fetch_files` for
`bop-benchmark/hot3d/object_models/models_info.json` and all 22 mapped GLBs.
The cache is `<raw root>/assets/hot3d_bop/`. The converter resolves mesh IDs by
**name** in a cached census table, never by a numeric ID from another HOT3D
release. [`OBJECTS`](../dataforge/datasets/show3d_source.py) is the alias-to-name
mapping, including the listed aliases with no mesh.

`keyboard2`, `cancoke`, `windex`, `clock`, and `mug3` have no matching mesh.
Conversion prints the missing mapping only while building `object_pose`, and
emits no `object_mesh` layer. The index token `none` denotes no object.
A mapped alias without object poses also produces no mesh layer.

Rerun 0.37 rejects `KHR_texture_transform`. Download strips that name from
`extensionsUsed`/`extensionsRequired` and from material texture infos once,
writing `stripped/obj_XXXXXX.glb` atomically and deleting the raw GLB.
UV transforms are discarded so Rerun 0.37 loads the asset. This chunk-preserving rewrite
updates JSON padding and GLB length while preserving binary chunks and other
extensions. Node scale 0.001 stays intact: the GLB scene graph already makes
the geometry metres, so the converter adds no second scale.

The exploration survey found no depth tree on the Hub (404) as of 2026-09-19,
although the README describes it. Depth remains reserved.
