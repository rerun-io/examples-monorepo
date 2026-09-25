# HOT3D Aria and Quest 3

## Source

HOT3D CDN release `v4.0.0`, verified against
`Hot3D{Aria,Quest}_download_urls-20260924.json`. `download()` verifies local files;
it does not fetch data. A complete sequence needs `recording.vrs` at the listed
byte size and all extracted ground-truth and hand-data files. Download markers,
MPS files, preview videos and `_simplecv` are not required. Partial folders are
skipped with a printed reason naming the incomplete file. Truncated or invalid
sequence metadata skips that sequence; a malformed URL manifest remains fatal.
Discovery carries the parsed metadata into target selection and conversion,
so each sequence metadata file is decoded once per discovery pass. No-GT metadata selects base only and requires
VRS, metadata and camera calibration, without annotation files.

The default is `paths.raw_root() / "hot3d"`. Set `DATAFORGE_RAW_ROOT` to the
**parent** of `hot3d`, or set the dataset's `root` directly. For the staged data:

```bash
export DATAFORGE_RAW_ROOT=/home/pablo/exoego-data/hot3d/rawroot
export DATAFORGE_OUTPUT_ROOT=/home/pablo/exoego-data/hot3d/iter
```

This resolves to `rawroot/hot3d/{aria,quest3,assets,...}`. The host NAS layout is
`/mnt/nas/datasets/hot3d`. Raw roots are read-only. The converter refuses an
output or temporary-work root beneath the raw root. It creates no caches there.
All conversion output stays on local disk; only the orchestrator copies finished
files to the NAS. No extraction, permission changes, bulk writes or downloads
run on the NAS.

Catalog names are `hot3d-aria` and `hot3d-quest3`; recording identities are
`hot3d-<device>__<sequence>`. Samples are registered by the driver as
`hot3d-aria-sample` and `hot3d-quest3-sample`.
`property:capture:source_resolution` preserves each native stream's resolution,
`source_revision` records the release and URL-list date, and `url_list_date`
records `20260924`. Episode properties include participant, native object IDs
and `has_gt`. Table cards decode only the primary camera.

`video_time` uses unshifted device nanoseconds. Each camera keeps its own native
capture stamps. Aria labels join `timestamp_ns` to `timecode_ns` exactly, row by
row, then use the shipped `devicetime_ns`. These are usually **1 ns before** the
VRS capture stamp; that difference is preserved. The offset changes during the
capture, so a constant time shift is incorrect. Quest capture timestamps are a
30 Hz grid as shipped, with no rebasing. `property:capture:clock_source` states
these conventions. `frame_index` is the secondary timeline: native camera index
for video, nearest primary frameset index for annotations only. This index does
not resample or alter their true timestamps. Ties go to the earlier primary
frame. The scene computes this index once and all annotation writers reuse it. IMU retains its native time axis;
no artificial camera frame index is assigned to IMU samples.

The label census is fixed before any annotation is read: Aria with GT uses
all mapping device stamps; Quest and every no-GT sequence use an identity
mapping over the full primary camera clock. Preview mode then cuts the census
at its stop stamp. Every nonzero label stamp must belong to that census,
including labels beyond a preview cutoff. Off-grid labels raise with the stamp;
readers never extend the census. UmeTrack and MANO rows leave the reader keyed
by device time; quality arrays also use device time. Ordering must be strictly
increasing, per stream for quality flags and per object for poses. Duplicate
and decreasing stamps raise. Missing pose rows stay missing.

Aria has three ~30 Hz cameras (1408×1408 RGB and two 640×480 gray SLAM streams),
and two IMUs at about 1000 and 800 Hz. Quest has two 1280×1024 gray cameras at
30 Hz and no IMU. Images from both devices rotate 90° clockwise for display.
The Quest decision was checked on a decoded native left-camera frame from
`P0002_5a9cfa51`: the unrotated table and monitor are sideways. This source-frame
check was confirmed in the viewer: in the sample screenshots (below) the rotated Quest panes are upright.
Output sizes are 1408×1408 / 480×640 for Aria and 1024×1280 for Quest.

Both devices use **`camera_models.json`**, the calibration solved with the GT.
Aria's JSON differs from its VRS factory calibration and from simplecv's MPS
online calibration. Aria has 15 parameters with a single focal length; Quest
has 16 with separate fx/fy fields, which must be equal (otherwise conversion
raises with both values). Both layouts use the same projectaria-tools
`CameraCalibration` and `rotate_camera_calib_cw90deg` path, including
`max_solid_angle`. Rotation updates dimensions, principal point, distortion and
camera basis together. Each camera stream retains the rotated full FISHEYE624
calibration for the derived projections layer. The
`Fisheye62Parameters` path drops the four thin-prism terms from FISHEYE624;
the logged pinhole is an approximation, and the videos retain their fisheye
appearance. The projections layer uses the full lens model; video is not undistorted.

Aria world coordinates are right-handed Z-up; Quest follows the headset's
right-handed Y-up world. The moving rig is the GT device frame, not a camera
or IMU origin. Aria IMU extrinsics come from VRS factory calibration, bridged
into the GT device frame through the common physical SLAM-left camera:
`gt_device_T_factory_device = gt_device_T_left @ inverse(factory_device_T_left)`.
Raw gyro/accel values are retained without rectification.

## Raw inventory

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `recording.vrs`, Aria `214-1` RGB | Native ~30 Hz, device | base | `/world/rig_00/cam_00/pinhole/video` |
| VRS Aria `1201-1`, `1201-2` SLAM | Native ~30 Hz, independent stamps | base | `cam_01`, `cam_02` pinhole/video |
| VRS Quest `1201-1`, `1201-2` | 30 Hz device grid | base | `cam_00`, `cam_01` pinhole/video |
| VRS Aria `1202-1`, `1202-2` | Native ~1000 / 800 Hz device | base | `/world/rig_00/imu_00`, `imu_01`, gyro rad/s and accel m/s² |
| VRS Aria time streams `285-*`, inactive `286-1` | UTC/timecode | — | Not ingested: explicit shipped label mapping supplies the required clock join |
| VRS factory calibration | Static | base | Aria IMU geometry only; camera geometry uses GT JSON |
| `camera_models.json` | Static | base | Rig camera extrinsics and pinholes; rotation and approximation above |
| `headset_trajectory.csv` | All native label stamps | base | `/world/rig_00`, world-from-device in metres |
| `metadata.json` | Static | base | `property:episode:{participant_id,object_ids,has_gt}` |
| `timecode_devicetime_mapping.csv` (Aria) | Every label stamp | all | Exact timecode→device join; no mapping by row count or nearest index |
| `umetrack_hand_pose_trajectory.jsonl` | All native rows, usually three per Aria frameset | hand_pose | `/world/gt/coco133_xyz`, `/world/gt/hands/<side>/{joint_angles,wrist,confidence}` |
| `umetrack_hand_user_profile.json` | Per sequence | hand_pose, hand_mesh | Verbatim profile at `/world/gt/hands/profile`; typed model drives FK/skinning |
| `mano_hand_pose_trajectory.jsonl` | Its own timestamps | hand_pose | `/world/gt/hands/<side>/mano`: PCA 15, betas 10, wrist translation/quaternion; not drawn |
| `masks/mask_good_exposure.csv` | Per stream / native label stamps | hand_pose | `/world/gt/quality/rig_00/cam_<index>/good_exposure`, 0/1 scalar |
| `masks/mask_hand_pose_available.csv` | Same | hand_pose | `.../hand_pose_available` |
| `masks/mask_hand_visible.csv` | Same | hand_pose | `.../hand_visible` |
| `masks/mask_headset_pose_available.csv` | Same | hand_pose | `.../headset_pose_available` |
| `masks/mask_object_pose_available.csv` | Same | hand_pose | `.../object_pose_available` |
| `masks/mask_object_visible.csv` | Same | hand_pose | `.../object_visible` |
| `masks/mask_qa_pass.csv` | Same | hand_pose | `.../qa_pass`; quality does not rewrite source confidence |
| `dynamic_objects.csv` | Native label stamps, sparse per UID | object_pose | `/world/gt/objects/<uid>` and confidence; metres, scalar-last logged quaternion |
| `assets/instance.json` | Static | object_pose | Native instance ID and human name; BOP IDs are not used |
| `assets/<uid>.glb` | Static geometry; temporal visibility | object_mesh | `/world/gt/objects/<uid>/mesh`, Asset3D in native object coordinates |
| Download URL lists | Release metadata | base / discovery | Size verification, extracted-file inventory, revision/date properties; URLs are not logged |
| `box2d_hands.csv`, `box2d_objects.csv` | Per-camera labels | — | Not ingested: 2D boxes deferred |
| Segmentation masks, if supplied | Pixel labels | — | Not ingested: segmentation deferred; CSV quality masks above are included |
| `mps/slam/{closed_loop_trajectory,open_loop_trajectory}.csv` | Up to 1 kHz | — | Not ingested: dense MPS trajectories deferred; GT headset trajectory owns rig pose |
| `mps/slam/online_calibration.jsonl`, `summary.json`, MPS artifacts | Calibration / diagnostics | — | Not ingested: GT camera JSON owns calibration; MPS processing deferred |
| `mps/slam/semidense_{points,observations}.csv.gz` | Sparse map | — | Not ingested: semi-dense reconstruction deferred |
| `mps/eye_gaze/{general,personalized}_eye_gaze.csv`, `mps/hand_tracking/` | MPS outputs | — | Not ingested: gaze deferred; shipped benchmark hands are the source |
| `*_preview_rgb.mp4` | Derived preview | — | Not read: VRS is the image source |
| `_simplecv/` | Prior derived outputs | — | Never read or written |
| CDN-named duplicate VRS, `.vrs.part`, `.download/*.done` | Download artifacts | — | Ignored: verify `recording.vrs` size and required sidecars instead |
| `license.txt` | Document | — | Not ingested: outside recording content |

## Layers and entities

Device facts live in one `DeviceSpec` table. Aria and Quest configs share
`Hot3dConfig` and keep their existing CLI and catalog names.

Six layers share the recording ID. Base carries capture properties and the
default blueprint; projections carries its derivation properties. The layout
has a 3D world view, orbital eye, and an ego-camera column. Each fisheye pane
shows only its video and `coco133_uv_projected`. It hides 3D hands, meshes and
objects because the viewer's pinhole cannot project them through a fisheye lens.
`coco133_uv` remains reserved for shipped 2D keypoints.

| Layer | Content |
| --- | --- |
| base | Videos, cameras, headset poses and Aria IMUs |
| hand_pose | 3D keypoints, model parameters and quality flags |
| hand_mesh | Skinned UmeTrack hands |
| projections | Derived full-lens 2D keypoints per camera |
| object_pose | Native object transforms |
| object_mesh | Native object geometry |

- **base:** native-rate videos encoded through the shared AV1 NVENC writer
  (CQ 36, GOP 60, no B-frames, RGB retained), GT rig poses and Aria IMUs.
  Every HOT3D camera stream stores JPEG. `dataforge.vrs.VrsImageReader` reads the raw
  JPEG block of each image record straight from the VRS container (no projectaria-tools
  decode, no pyvrs: it has no linux-aarch64 wheel); each record's DataLayout
  `capture_timestamp_ns` must equal the projectaria-tools stamp of the same index, and the
  record count must equal the stream's. `video_encoding.decode_jpeg_frames` decodes them with
  TurboJPEG to the JPEG's own planes (gray, or YUV 4:2:0 for RGB) on 8 threads per camera;
  ffmpeg rotates them, maps the JPEG's full-range YUV to limited range and encodes.
  Gray streams are byte-identical to the old RGB/gray path; RGB skips the old
  YUV→RGB→YUV round trip (chroma within 19 levels, mean 0.6).
  `parallel_clips` runs camera jobs with a separate reader per job and
  yields clips for logging in camera order. Frame counts are checked.
  Quest's missing headset poses get NaN transforms, never nearest-neighbour
  interpolation, head clamping or carry-forward.
- **hand_pose:** `Points3DWithConfidence` from UmeTrack FK, plus the source
  model parameters and quality scalars. HOT3D ships no measured landmarks.
  Profile coordinates are millimetres; wrist inputs and output geometry are
  metres. Right-hand mirroring uses the shared SHOW3D UmeTrack math.
  Present hands retain shipped confidence (currently 1.0 in the inspected
  corpus); absent hands have NaN positions and confidence 0.0. Uncovered COCO
  slots are also NaN/0. The Assembly-Hands mapping derives thumb-base slots
  92/113 as wrist–thumb-CMC midpoints and copies wrist slots 91/112 into 9/10.
  UmeTrack palm landmark 20 is unused. Missing parameters are explicit NaNs.
  MANO stays 15 PCA values, not a padded 45-value vector. Stamp-zero MANO and
  quality rows are dropped and counted in stdout. MANO is dense on the census,
  with NaNs where its own stamped rows are missing. Quality entities use the
  scene camera index and retain the VRS stream ID as a static field. Duplicate quality stamps
  from distinct cameras remain distinct stream rows.
- **hand_mesh:** UmeTrack skinning in bounded batches; missing/untrusted hands
  get empty vertex rows so no old hand mesh remains visible. MANO is not drawn.
  The dispatcher loads the hand profile and evaluates FK once for both hand layers and projections.
- **projections:** the same dense world `coco133_xyz` rows projected with each
  rotated camera's full FISHEYE624 model, including thin-prism terms, through
  the exact GT headset pose at each label stamp. Pixels live at
  `<pinhole>/coco133_uv_projected` on `video_time` and `frame_index`.
  Missing joints or headset poses, points behind the camera, and rejected
  projections are NaN with confidence 0.0; accepted joints keep 3D confidence.
  No clamping or interpolation occurs. Static properties are
  `property:projections:derived_from="coco133_xyz"` and
  `property:projections:camera_model="FISHEYE624"`. No-GT sequences remain base only.
- **object_pose:** exact native-ID transforms; a present row has confidence
  1.0 because the source has no confidence column. A missing row has confidence
  0.0 and an invalid transform. No temporal pose interpolation occurs.
  Rig, wrist and object poses use the shared dense pose writer. Object
  invalidation is opt-in; the default shared behavior for SHOW3D and HO-Cap
  remains sparse. Object pose and mesh layers have separate writers.
- **object_mesh:** native HOT3D GLBs, with `KHR_texture_transform` stripped
  in memory using the shared utility. Other GLB chunks and node transforms
  remain intact. This layer owns no object transforms. Albedo alpha is zero
  when a pose is missing/untrusted.

`frame_limit=N` keeps the first N frames of each camera and all label stamps
through the latest Nth camera stamp. It writes into
`<output_root>/preview-firstN/`, apart from full recordings. Full conversion
retains IMU samples before and after the image span; previews cut only the end.

## Differences from simplecv

| Difference | Raw evidence / corrected audit defect |
| --- | --- |
| Keep every Aria hand row | 5476 rows for 1827 framesets in `P0001_4bf4e21a`: RGB, right SLAM, left SLAM (12 ns later); occasional shared SLAM stamp. Simplecv kept only nearest RGB rows. |
| Exact timecode join | Mapping offsets vary by 0.93 ms; labels must match mapping keys, not just row counts. |
| Missing hands are NaN/0 | Absent hand keys and empty rows are explicit; simplecv carried landmarks forward. |
| MANO uses its own stamps | Quest adds an invalid stamp-zero row and sometimes omits another row; UmeTrack row indices cannot select MANO. Hand keys stay 0=left, 1=right. |
| Sparse headset gaps stay missing | Quest iteration sequence has 13 frame gaps; simplecv nearest-neighbour/head clamping fabricated poses. |
| Objects, quality, IMU and model parameters retained | These files/streams were omitted by simplecv's recording writer. |
| Native unshifted video clock | Simplecv used re-encoded CFR MP4 PTS and rebased to zero. Container PTS here only locates encoded samples; Rerun sample times come from VRS. |
| Full-lens derived 2D | Simplecv projected with Fisheye62 (thin-prism dropped) into `coco133_uv`. We use each camera's full FISHEYE624 at the separate derived entity `coco133_uv_projected`; fisheye panes hide 3D content. |
| GT camera calibration | Both devices use shipped `camera_models.json`; Aria no longer reads MPS online-calibration row zero. |

A later rectified-video layer could let object meshes appear correctly in camera panes.

Parity sequences are Aria `P0001_4bf4e21a` and Quest `P0002_5a9cfa51`.
Golden tests read the simplecv reference RRDs under
`/mnt/nas/datasets/exoego-forge-catalog-rig/hot3d-<device>/`. Reference times
are shifted by the first primary VRS capture stamp (Quest
`46201933333333` ns); Aria matching allows the documented 1 ns difference.
Raw hand rows determine expected joint presence before coordinate comparison.
Missing or extra joints fail unless reference-only joints belong to a hand
absent in the raw row (the documented simplecv carry-forward defect). The gate
prints unmatched reference and census stamp counts, plus carried joint counts.
The gate is maximum error ≤0.0001 m, with >90% of primary frames compared.
Extra Aria rows are expected.
A second test appends exact rest landmarks as mesh probes with their source
skinning weights, checking mesh and FK world coordinates; landmarks are not
assumed to be surface vertices. Object placement checks logged transforms on
the native mesh's POSITION bounds and checks preserved GLB bytes.

Integration tests encode all layers for 60 frames on both devices, verify native
video stamps and layer ownership. A 30-frame preview checks projection entities,
properties, finite rows and rotated image bounds. Tests also check rotated gray/RGB output and no
B-frames. Missing assets produce skips naming the path. The driver runs these tests and the golden
tests on the staged captures (`dataforge-dev gate`).

Pixel evidence (headless Rerun 0.38.1, all six layers, default blueprint), from the local sample files:

- Aria `P0001_550ea2ac` frame 2093: [full](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hot3d-aria-sample-P0001_550ea2ac-f2093-default-blueprint.png),
  [RGB pane zoom](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hot3d-aria-sample-P0001_550ea2ac-f2093-rgb-projections-zoom.png),
  [SLAM-left zoom](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hot3d-aria-sample-P0001_550ea2ac-f2093-slam-left-zoom.png).
- Quest 3 `P0002_273c2819` frame 1722: [full](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hot3d-quest3-sample-P0002_273c2819-f1722-default-blueprint.png),
  [left pane zoom](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hot3d-quest3-sample-P0002_273c2819-f1722-left-projections-zoom.png),
  [3D zoom](https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/hot3d-quest3-sample-P0002_273c2819-f1722-3d-zoom.png).

In every camera pane `coco133_uv_projected` sits on both hands.

## Parity

Measured 2026-09-25 on pablo-dl-server: simplecv `main` @ 34ee7f4c against this converter
(`1+fb2576abcf9d`, prod env) on Aria `P0001_4bf4e21a` and Quest 3 `P0002_5a9cfa51`. The reference is
simplecv's own rrds; a fresh simplecv run on local copies reproduces them (identical stamps and
NaN pattern, keypoints within 2.4e-7 m). Full report and scripts:
`/tmp/fleet-artifacts/exoego-migration/runs/hot3d-parity.md`.

| recording | matched rows | joints in both | max joint distance | joints only in one side | camera centres (max) | orientation beyond the 90° display turn |
| --- | --- | --- | --- | --- | --- | --- |
| hot3d-aria__P0001_4bf4e21a | 1827 of 1827 | 80,124 | 2.4e-7 m | 0 | 13 µm | ≤5.6e-6° |
| hot3d-quest3__P0002_5a9cfa51 | 3650 of 3650 | 160,204 | 3.6e-7 m | 0 | 0.26 µm (3637 frames) | ≤4.9e-6° |

The gate is 1e-4 m. Every other difference is a corrected simplecv defect or a documented choice
(table above): the dropped Aria SLAM-stamp rows, NaN confidences, the rebased CFR clock, the 13
fabricated Quest poses, MPS vs GT calibration (≤0.03 px, ≤13 µm), the cw90 image rotation, MANO
drawn vs kept as data, and reprojected `coco133_uv` vs the full-lens `coco133_uv_projected`.
Informative 2D check: Quest projections agree with simplecv's to 5e-4 px. Aria differs by 0.3–2.5 px
because simplecv drops the thin-prism terms. Points outside the lens model's valid field of view are
NaN here; simplecv projects them anyway.

## Timing

Conversion uses `dataforge.timing` fetch, transcode and per-layer write stages; the convert CLI
appends `<output_root>/timing/convert.jsonl`. Registration uses
`<output_root>/timing/register.jsonl`. Read with `load_records` and `ConvertRecord` /
`RegisterRecord`.

### Baseline against simplecv (2026-09-25, pablo-dl-server, AV1 NVENC)

Both run in their prod envs on local copies of the same raw files, one at a time under the shared
GPU lock. simplecv = `preprocess_hot3d.py` (VRS → AV1 mp4) + `batch_raw_to_rrd.py hot3d`, wall
time including two pixi start-ups; its internal timers give 6.1 s (Aria) and 8.8 s (Quest).
dataforge = the convert record's `total_s` for all six layers.

| recording | capture s | simplecv s | simplecv s/capture-min | dataforge s | dataforge s/capture-min | dataforge transcode s | output (simplecv / dataforge) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hot3d-aria__P0001_4bf4e21a | 60.9 | 9.4 | 9.3 | 8.4 | 8.3 | 3.6 | 86 / 165 MB |
| hot3d-quest3__P0002_5a9cfa51 | 121.6 | 12.1 | 6.0 | 7.3 | 3.6 | 4.3 | 140 / 191 MB |

**Soft gate: met** after the TurboJPEG decode path (dataforge rows above; all six layers,
prod env, pablo-dl-server, 2026-09-25). Before it, dataforge took 17.9 s (17.7 s/capture-min,
transcode 12.9 s) on Aria and 17.4 s (8.6, transcode 14.3 s) on Quest 3: every JPEG went through
projectaria-tools to an RGB/gray array (Aria RGB alone: 7.3 s for 1827 frames; one Quest camera:
5.5 s for 3650 frames) before ffmpeg rotated it. Now the raw JPEG blocks come straight from the
VRS records and TurboJPEG decodes them to planes on 8 threads per camera (Aria RGB 275 → 1,724
frames/s; Quest camera 630 → 4,700 frames/s, CPU only). ffmpeg still rotates: rotating the planes in
NumPy measured slower (0.21 vs 0.17 s per 100 RGB frames). dataforge also writes more than simplecv:
hand meshes (0.7–1.5 s), projections (1.0–2.1 s), IMU, objects.

### Sample conversions (2026-09-25, prod env, converter `1+fb2576abcf9d`)

| recording | capture s | total s | s/capture-min | bytes (6 layers) |
| --- | --- | --- | --- | --- |
| hot3d-aria__P0001_550ea2ac | 130.2 | 34.6 | 16.0 | 354.2 MB |
| hot3d-aria__P0001_624f2ba9 | 124.7 | 35.6 | 17.1 | 323.1 MB |
| hot3d-aria__P0001_8d136980 | 127.5 | 34.5 | 16.2 | 306.1 MB |
| hot3d-aria__P0001_9c030609 | 126.1 | 36.3 | 17.3 | 344.5 MB |
| hot3d-aria__P0001_a68492d5 | 124.6 | 36.0 | 17.3 | 348.6 MB |
| hot3d-aria__P0001_a9d6c83d | 128.8 | 34.1 | 15.9 | 303.6 MB |
| hot3d-aria__P0002_2ea9af5b | 121.3 | 32.0 | 15.8 | 297.9 MB |
| hot3d-aria__P0002_65085bfc | 119.6 | 31.4 | 15.8 | 290.2 MB |
| hot3d-quest3__P0002_1464cbdc | 132.7 | 24.9 | 11.3 | 304.7 MB |
| hot3d-quest3__P0002_273c2819 | 121.3 | 17.4 | 8.6 | 170.6 MB |
| hot3d-quest3__P0002_45904c71 | 123.7 | 17.6 | 8.5 | 180.8 MB |
| hot3d-quest3__P0002_75103f48 | 121.4 | 16.9 | 8.3 | 188.1 MB |
| hot3d-quest3__P0002_a2f1b530 | 56.4 | 8.4 | 8.9 | 88.8 MB |
| hot3d-quest3__P0002_af0d3d4a | 101.7 | 14.8 | 8.7 | 158.2 MB |
| hot3d-quest3__P0002_c3aec89e | 137.5 | 18.7 | 8.2 | 218.0 MB |
| hot3d-quest3__P0003_3fb19e29 | 62.1 | 10.1 | 9.7 | 97.9 MB |

The Aria samples run at 15.8–17.3 s per capture-minute, the Quest 3 samples at 8.2–11.3.
`P0002_1464cbdc` (11.3) has a 2.85 GB VRS (the others 1.5–1.6 GB) and a 183 MB base layer
(the others ~70 MB): its frames carry more detail, so transcode takes 21.6 s instead of ~14 s. Registration time is added when the samples are registered.
