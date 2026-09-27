# UmeTrack

## Source

- Source: `https://github.com/facebookresearch/UmeTrack_data` at commit `950bda2ab602d0ca5591476e04c993ec6f3cac4f` (2023-05-03, the repository's only commit and still `main`), raw stacked videos and labels. Source revision property: `github.com/facebookresearch/UmeTrack_data@950bda2ab602d0ca5591476e04c993ec6f3cac4f`. The earlier local mirror is byte-identical to this commit (checked on `real/hand_hand/training/user_03/recording_05`).
- One catalog dataset, `umetrack`; sample registration name `umetrack-sample`. Identity: `umetrack__<domain>__<interaction>__<split>__<user>__<recording>`. Episode properties: `domain`, `interaction`, `split`, `user`. Capture properties include source revision, source resolution (`cam_0k:<W>x<H>` per tile), source/full frame counts, fps, and clock source. Each camera node carries the shared camera-source keys: tile `source_width`/`source_height`, `video_codec`, `cq`, `gop`.
- Real source frames are 2544×480 (four 636×480 tiles); synthetic frames are 2560×480 (four 640×480 tiles). Cameras are `cam_00`–`cam_03` in label/tile order. Their kind is `grayscale`.

## Get the raw data

- Raw root: `paths.raw_root() / "umetrack/raw_data"` (the source repository's `raw_data/` inside the dataset directory), by default `packages/dataforge/data/raw/umetrack/raw_data` (the pixi tasks run from `packages/dataforge`); set `DATAFORGE_RAW_ROOT` or pass `--root`. No login and no URL file: everything is public on GitHub (labels as git blobs, videos as git-LFS objects).
- Get the data (2,410 recordings, 168.7 GB, ~70 MB each; ~229 MB/s with 8 parallel transfers, so ~12 min for the corpus on a fast link):

  ```bash
  pixi run -e dataforge dataforge-download --list-remote umetrack          # JSON lines: key, size_bytes, files
  pixi run -e dataforge dataforge-download umetrack                        # everything
  pixi run -e dataforge dataforge-download umetrack --sequences real/hand_hand/training/user_03/recording_05
  pixi run -e dataforge dataforge-convert umetrack --sequences real/hand_hand/training/user_03/recording_05
  ```

  The first call (listing or download) creates the raw root and writes the index `umetrack_index_950bda2ab602.json` (770 kB, the only shared file) there from one GitHub tree request plus the 2,410 LFS pointers (~5 s), and later calls reuse it. The GitHub API allows 60 unauthenticated requests per hour; one is needed per fresh raw root. `download()` fetches every file at the pinned commit to `<name>.partial`, resumes that on the next run, and renames it only after the size and hash match (sha256 from the LFS pointer for MP4s, git's blob sha1 for JSONs); a `.partial` of the wrong size or hash is removed and restarted on the next run. Every failed file is listed at the end of the run. A second download on the same raw root is refused while one runs (a lock on the directory itself, no lock file). Retry messages go to stderr, so `--list-remote` stdout is always JSON lines. A cached index with malformed rows, unpaired recordings or the wrong count is refused (delete it to rebuild). A file under its final name counts as complete (size check only); a final file of the wrong size stops the run and is never overwritten. `download()` deletes only its own proven-wrong `.partial` files. `discover()` lists every recording whose JSON and MP4 are both present, so recordings pruned after conversion are ignored; blueprints and registration need no raw files.
- Local conversion root: the host sets `DATAFORGE_OUTPUT_ROOT`; conversion never writes under the raw root.

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
| GitHub tree + LFS pointers at the pinned commit | No clock | — | `umetrack_index_<rev>.json`: path, size and hash of every raw file, for `download()` and `--list-remote` |
| Derived COCO-133 through each camera’s lens model | Hand-pose clock | projections | `<pinhole>/coco133_uv_projected`, Points2DWithConfidence |
| Profile rest geometry, weights, topology, joint limits, hand scale | Static | hand_pose / hand_mesh | Preserved in full profile; geometry and weights drive FK and meshes |

There are no shipped 2D points, landmarks, depth, segmentation, IMU, or object tracks. No `coco133_uv` is created.

## Clocks

`video_time` is presentation time in nanoseconds from MP4 packet PTS × time base; `frame_index` is the second timeline. `property:capture:clock_source = "mp4_container_pts"`. The corpus has constant integer rates, mostly 30 fps, with 77 recordings at 18–29 fps. The real iteration recording is **29 fps**. Labels have no timestamps. The reader checks constant PTS spacing, zero origin, and video/label count agreement.

## Layers and entities

`base`, `hand_pose`, `hand_mesh`, and `projections` share the recording id. The base owns the root annotation context and `/world` ViewCoordinates. Annotation layers do not inject RecordingInfo properties.

Rerun cannot crop a stacked video. Each tile is transcoded with AV1 NVENC, CQ 36, GOP 60, no B-frames, at its measured source rate. Grayscale is carried in yuv420p with neutral chroma. The shared writer retains both source timelines. Four crop jobs can overlap. `--frame-limit 60` writes under `preview-first60/`, separate from full recordings.

| Layer | Contents |
| --- | --- |
| base | Video, camera calibration, and rig poses |
| hand_pose | World COCO keypoints, confidence, and source hand parameters |
| hand_mesh | Skinned world-space hand meshes |
| projections | Derived `coco133_uv_projected` per camera, through its full FishEye62 model |

2D is projected only: UmeTrack ships no 2D, so there is no `coco133_uv`. The projections layer uses the same `video_time` and `frame_index` as hand_pose and the same world COCO positions, computed once per conversion. Its properties are `property:projections:derived_from = "coco133_xyz"` and `property:projections:camera_model = "FishEye62"`. Each present projection keeps the 3D joint’s confidence. Missing inputs, untracked rig frames, z ≤ 0, rays at or beyond the first radial derivative zero (or π/2 if none), and out-of-image pixels become NaN with confidence 0. No coordinates are clamped.

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

Real and synthetic projections were checked against the video pixels: real `hand_hand/training/user_01/recording_14` frame 225 ([cam01 zoom](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/umetrack-sample-real-u01r14-f225-cam01-projections-zoom.png)) and synthetic `hand_hand/training/user_03/recording_11` frame 168 ([cam02 zoom](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/umetrack-sample-synthetic-u03r11-f168-cam02-projections-zoom.png)). The projections layer has converted bit-identically since. Synthetic projections use `k5 = k6 = 0`, as explained above.

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

Integration, golden, and viewer pixel checks run on a host with the raw data and NVENC. Integration converts the first 60 frames of both iteration recordings with the shared NVENC fixture. Visual checks must include changing frames in all four panes, world geometry, hand disappearance, and real frame 430 rig disappearance.

## Parity with simplecv

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

## Known gaps

- There is no synthetic simplecv reference rrd, so parity covers real recordings only.
- Rerun 0.38.1 does not apply the FishEye62 distortion: camera panes show no 3D hands or meshes.
- Synthetic `p3/p4` are kept as source data only; the typed lens uses `k5 = k6 = 0`.
