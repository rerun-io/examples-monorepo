# UmeTrack

## Source

- Source: `https://github.com/facebookresearch/UmeTrack_data`, raw stacked videos and labels. Source revision property: `github.com/facebookresearch/UmeTrack_data@main (mirrored 2025-10-31)`. The mirror has no recoverable commit hash.
- Default read-only root: `paths.raw_root() / "umetrack-data/raw_data"`. `DATAFORGE_RAW_ROOT` sets the raw base directory; `--root` overrides the recording root. Tests use the same base directory and skip when an asset is absent.
- Local conversion root: the host sets `DATAFORGE_RAW_ROOT=/home/pablo/exoego-data/umetrack/raw` for the local mirror and `DATAFORGE_OUTPUT_ROOT=/home/pablo/exoego-data/umetrack/iter`. No writes under the raw root. Only the driver copies finished outputs to the NAS. `download()` verifies nonempty files against the two manifests beside `raw_data`; it does not download. With `--sequences`, only selected recordings must exist.
- One catalog dataset, `umetrack`; sample registration name `umetrack-sample`. Identity: `umetrack__<domain>__<interaction>__<split>__<user>__<recording>`. Episode properties: `domain`, `interaction`, `split`, `user`. Capture properties include source revision, source resolution, source/full frame counts, fps, and clock source.
- Real source frames are 2544×480 (four 636×480 tiles); synthetic frames are 2560×480 (four 640×480 tiles). Cameras are `cam_00`–`cam_03` in label/tile order. Their kind is `grayscale`.
- `video_time` is presentation time in nanoseconds from MP4 packet PTS × time base; `frame_index` is the second timeline. `property:capture:clock_source = "mp4_container_pts"`. The corpus has constant integer rates, mostly 30 fps, with 77 recordings at 18–29 fps. The real iteration recording is **29 fps**. Labels have no timestamps. The reader checks constant PTS spacing, zero origin, and video/label count agreement.
- Rerun cannot crop a stacked video. Each tile is transcoded with AV1 NVENC, CQ 36, GOP 60, no B-frames, at its measured source rate. Grayscale is carried in yuv420p with neutral chroma. The shared writer retains both source timelines. Four crop jobs can overlap. `--frame-limit 60` writes under `preview-first60/`, separate from full recordings.

## Raw inventory

| Shipped file or stream | Rate / clock | Layer | Entity or use |
| --- | --- | --- | --- |
| `recording_YY.mp4`, four horizontal tiles | MP4 PTS | base | `/world/rig_00/cam_0k/pinhole/video`, VideoStream |
| JSON `cameras`, list order | Static | base | Camera Transform3D, typed Pinhole and Kannala–Brandt distortion components; one AnyValues group preserves source coefficient names and crop roll |
| JSON `camera_angles` | Static degrees | base | `source_camera_angle_deg` at each pinhole; also used for measured headset up |
| JSON `camera_to_world_transforms` | One row per frame, mm | base | `/world/rig_00`, metric Transform3D and `untracked`; static camera-relative extrinsics |
| JSON `hand_model` | Static | hand_pose | `/world/gt/hands/profile`, verbatim sub-object as JSON TextDocument |
| JSON `joint_angles` | One row per frame, radians | hand_pose | `/world/gt/hands/<side>/joint_angles`, only confidence-positive rows |
| JSON `wrist_transforms` | One row per frame, mm | hand_pose | `/world/gt/hands/<side>/wrist`, metric Transform3D on confidence-positive rows |
| JSON `hand_confidences` | One row per frame | hand_pose | `/world/gt/hands/<side>/confidence`, Scalars including zeros; also per-keypoint confidence |
| `raw_data_{real,synthetic}_manifest.txt` | No clock | — | Verify-only inventory; URL/dir/out entries, no checksums or sizes supplied |
| Derived COCO-133 through each camera’s lens model | Hand-pose clock | projections | `<pinhole>/coco133_uv_projected`, Points2DWithConfidence |
| Profile rest geometry, weights, topology, joint limits, hand scale | Static | hand_pose / hand_mesh | Preserved in full profile; geometry and weights drive FK and meshes |

There are no shipped 2D points, landmarks, depth, segmentation, IMU, or object tracks. No `coco133_uv` is created.

## Layers and entities

`base`, `hand_pose`, `hand_mesh`, and `projections` share the recording id. The base owns the root annotation context and `/world` ViewCoordinates. Annotation layers do not inject RecordingInfo properties.

| Layer | Contents |
| --- | --- |
| base | Video, camera calibration, and rig poses |
| hand_pose | World COCO keypoints, confidence, and source hand parameters |
| hand_mesh | Skinned world-space hand meshes |
| projections | Derived `coco133_uv_projected` per camera, through its full FishEye62 model |

The projections layer uses the same `video_time` and `frame_index` as hand_pose and the same world COCO positions, computed once per conversion. Its properties are `property:projections:derived_from = "coco133_xyz"` and `property:projections:camera_model = "FishEye62"`. Each present projection keeps the 3D joint’s confidence. Missing inputs, untracked rig frames, z ≤ 0, rays at or beyond the first radial derivative zero (or π/2 if none), and out-of-image pixels become NaN with confidence 0. No coordinates are clamped.

`world_T_rig` is cam0's source pose, with translation multiplied by 0.001. `rig_T_cam` is computed from the first tracked source frame and is static. An all-zero camera frame becomes NaN translation and quaternion on that timestamp, with `untracked=true`. No pose is carried forward or replaced by identity.

The shared UmeTrack model computes 21 landmarks from the profile, angles and wrist, in one call per present hand. `hands.coco133_from_hands` maps them to `/world/gt/coco133_xyz` as Points3DWithConfidence, in metres. Confidence is the shipped per-hand value. Missing hands and uncovered COCO slots are NaN with confidence 0. The Assembly-Hands mapping supplies wrist copies (body slots 9/10 and hand roots), a thumb-base midpoint, and hand slots 91–132; palm landmark 20 is not a COCO slot. These keypoints are derived FK, not shipped landmark observations.

Joint angles and wrist rows are sparse and only logged at confidence > 0. Confidence Scalars remain dense. The profile TextDocument preserves the original `hand_model` object's text, including whitespace. Mesh topology and albedo are static; world vertices are in metres. Every confidence-zero mesh frame has an empty vertex row so latest-at cannot display a stale mesh. SHOW3D and UmeTrack share handedness and the batched mesh-writing core.

### World axes

No gravity sensor is shipped. For each tracked frame and each camera, undo its crop roll about the optical axis: image-up in world coordinates is `sin(roll) * R[:,0] - cos(roll) * R[:,1]`. Average over all four cameras and all tracked source frames, then normalize. This measurement uses the full recording even for a preview.

Log `WORLD_UP_VIEW_COORDINATES["+y"]` (RUB) at `/world`. Capture properties `world_up_axis`, `headset_up`, and `headset_up_spread_deg` record the choice, mean direction, and maximum angle from a camera’s average up vector to that mean. The four cameras are pitched about 50 degrees (synthetic: about 37 degrees) in two symmetric pairs. Their mean is the bisector. The +Y choice still holds: all four camera up vectors have positive y-components, and the hands are below the head. The spread uses each camera’s roll-corrected up averaged over all tracked source frames, even for a preview. The iteration examples measured approximately:

| Recording | Headset up (world) |
| --- | --- |
| real/hand_hand/training/user_03/recording_05 | (−0.06, 1.00, 0.01) |
| synthetic/separate_hand/testing/user_19/recording_02 | (0.09, 0.99, 0.08) |

### Camera panes

Pinholes use shipped fx/fy/cx/cy over the distorted source image. The canonical `Fisheye62Parameters` path logs typed `DistortionModel` and the `DistortionCoefficients` component with all eight values; **the viewer does not apply this distortion**. Synthetic cameras ship `p3/p4` instead of `k5/k6`. UmeTrack_data issue #4 reads them as the fifth and sixth radial terms, but they do not fit the images: with `p3/p4` as `k5/k6` (values such as 56.7 and −70.6) the projected hands miss the rendered hands by hundreds of pixels, while `k5 = k6 = 0` puts them on the hands in all four cameras (checked on `synthetic/separate_hand/testing/user_19/recording_02` frame 224 and `synthetic/hand_hand/training/user_03/recording_11` frame 150; evidence `umetrack-synthetic-lens-p3p4-vs-k0-{a,b}.png`). So the typed synthetic lens has `k5 = k6 = 0`, and `p3/p4` stay only as source data. One small AnyValues group keeps the original coefficient names and `source_camera_angle_deg`. A camera must supply exactly one complete pair, `k5/k6` or `p3/p4`.

Camera panes show only their video and derived lens-model projections. They exclude 3D hands and meshes because Rerun 0.38.1’s distortion-free Pinhole would misplace that content on the fisheye image. Each camera uses its own full FishEye62 calibration. The default blueprint retains the 3D orbital view and four ego panes; table cards use the same pane contents for cam0 beside the hands.

Real and synthetic projections were checked against the video pixels (see the lane evidence). Synthetic projections use `k5 = k6 = 0`, as explained above.

## Differences from simplecv

| Difference | Raw evidence / audit defect |
| --- | --- |
| Read raw stacked MP4, encode CQ 36 | Old loader reads CQ 35 split re-encodes; raw source is the agreed input |
| Indexed camera names, grayscale kind | Source cameras have no names and neutral chroma; old TL/BL/BR/TR names were guesses, kind was rgb |
| Rig dropout is NaN | All four source matrices are zero; simplecv carried the previous pose (real iteration frame 430) |
| Missing keypoints are NaN | Confidence zero exactly matches zero wrist; simplecv carried 42 finite hand points at frame 430 |
| Joint angles, wrists, profile, meshes, camera roll, tracking flag, per-hand confidence retained | These shipped fields were previously omitted |
| Synthetic p3/p4 retained as source data, not projected | Old camera construction dropped them too; pixel checks show that the images follow k5 = k6 = 0, not issue #4's reading |
| RUB at `/world` | Measured +Y up and +X forward resolve contradictory BUL/RUB loader conventions |
| Derived UV has a separate name | We write `coco133_uv_projected` through each camera’s own calibration and full FishEye62 model. Simplecv wrote derived UV under the shipped name `coco133_uv`, which remains reserved for source 2D |

Golden gate: compare confidence-positive hand slots against the local simplecv real reference RRD at identical video_time, with absolute tolerance 1e-4 m. Compare rig translations on tracked rows at the same tolerance. Assert frame 430 is missing in ours. A separate FK comparison uses `landmarks_from_hand_pose`. There is no synthetic reference RRD in the corpus.

SHOW3D fixture refactor comparison: 35 hand-pose component tracks and 6 mesh tracks equal at zero tolerance. This compares serialized component values and clocks, excluding generated store/chunk ids; it does not claim the entire RRD files have identical binary hashes.

Integration, golden, and viewer pixel evidence are driver-run gates. Integration converts the first 60 frames of both iteration recordings with the shared NVENC fixture. Driver visual checks must include changing frames in all four panes, world geometry, hand disappearance, and real frame 430 rig disappearance. No pixel validation has been claimed by the sandbox implementation.

## Parity

Checked on 2026-09-25 against simplecv `main` @ 34ee7f4c. The reference is simplecv's own rrds for the three real recordings that have one: `real/hand_hand/training/user_03/recording_05`, `real/hand_hand/training/user_01/recording_14` and `real/hand_hand/testing/user_09/recording_04`. There is no synthetic reference. Rows are matched on `video_time`, which is identical on every row.

| Recording | coco133_xyz joints in both | Max joint distance | Rig translation / rotation (tracked) | Other differences |
| --- | --- | --- | --- | --- |
| user_03/recording_05 | 18,920 | 6.0e-8 m | 6.0e-8 m / 4.3e-6° | frame 430: simplecv carries 44 keypoints and the rig pose forward; ours is NaN + `untracked` |
| user_01/recording_14 | 19,844 | 3.1e-8 m | 6.0e-8 m / 4.0e-6° | none beyond the table above |
| user_09/recording_04 | 11,000 | 4.5e-8 m | 6.0e-8 m / 4.9e-6° | none beyond the table above |

Intrinsics and all eight lens coefficients are equal. The static camera extrinsics agree to 4e-8 m. simplecv's reprojected 2D agrees with `coco133_uv_projected` to 4e-4 px. Every other difference is in "Differences from simplecv". Missing joints have NaN confidence in simplecv and 0.0 here.

## Timing

Measured on pablo-dl-server (RTX 5090, NVENC) on 2026-09-25, in the prod environments, from local copies. The numbers are seconds of processing per minute of capture.

| Recording | Capture | dataforge `convert` | simplecv split + convert | dataforge s/min | simplecv s/min |
| --- | --- | --- | --- | --- | --- |
| real/hand_hand/training/user_03/recording_05 | 14.86 s (431 frames, 29 fps) | 2.64 s (transcode 2.43) | 4.03 s + 0.18 s | 10.7 | 17.0 |
| synthetic/separate_hand/testing/user_19/recording_02 | 14.93 s (448 frames, 30 fps) | 2.80 s (transcode 2.45) | 4.98 s + 0.17 s | 11.3 | 20.7 |

dataforge times come from its own timers (`convert.jsonl`: fetch, hands, transcode, write per layer). The four crops encode in parallel, and the base write includes the transcode. simplecv times are its own timers: `split_umetrack_video.py` encodes the four crops one after the other at preset p7, and `batch_raw_to_rrd.py` then remuxes the split files. Neither figure includes Python and pixi startup, which is about 2.7 s per process on both sides. Output sizes for the real recording: base 5.6 MB, hand_pose 0.51 MB, hand_mesh 8.2 MB, projections 0.41 MB. dataforge is faster per capture-minute, so the soft speed gate passes. Registration time is added when the sample is registered.

The eight sample recordings convert at 8.6–15.8 s per capture-minute. The one exception is `synthetic/hand_hand/testing/user_09/recording_01` (68 frames, 2.3 s) at 33.6 s/min, where fixed per-recording costs dominate.
