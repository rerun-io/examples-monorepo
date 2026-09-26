# Aria Gen2 Pilot

## Source

Aria Gen2 Pilot Dataset release `v1.0` (projectaria.com), 12 sequences:
`clean_0, cook_0, eat_0..3, play_0..3, walk_0, walk_1`. The CDN URLs in
`AriaGen2PilotDataset_download_urls.json` have expired; `download()` never fetches.
It verifies each discovered `video.vrs` against the manifest's `main_vrs`
size and SHA-1 and prints the count.

The default raw root is `/mnt/nas/datasets/aria-gen2-pilot`; `DATAFORGE_RAW_ROOT`
(or the dataset's `root`) points at a directory with the same layout. Over NFS the
VRS files read as mode 000 (Synology ACL), so the lane staged its subset locally:

```bash
export DATAFORGE_RAW_ROOT=/home/pablo/exoego-data/aria_gen2_pilot/raw
export DATAFORGE_OUTPUT_ROOT=/home/pablo/exoego-data/aria_gen2_pilot/iter
```

The converter reads only `<seq>/video.vrs` and `<seq>/mps/`. Raw roots are
read-only: the converter refuses an output or work root beneath the raw root
or the source directory, and writes no cache there.
It never reads `_simplecv/` (simplecv's preprocessing output).

Catalog dataset `aria_gen2_pilot` (sample `aria_gen2_pilot-sample`); recording
identity `aria_gen2_pilot__<seq>`. Properties: `property:capture:source_resolution`
(`<stream id>:<w>x<h>` per camera), `source_revision` `AriaGen2PilotDataset v1.0`,
`dataset_version` `v1.0`, `calibration_source`, `clock_source`, `num_frames`
(RGB frames), `source_num_frames`; `property:episode:sequence`.

**Clock.** `video_time` is the unshifted device clock in nanoseconds (nanoseconds
since device boot). Every camera and IMU keeps its own VRS `capture_timestamp_ns`
(the VRS record timestamp is the same value). MPS rows carry
`tracking_timestamp_us`, logged as `× 1000` ns. MPS hand rows are the SLAM camera
stamps floored to the microsecond, up to 999 ns before the matching frame; the
shipped value is kept. `frame_index` is the second timeline: the nearest `slam-front-left` (cam_01, 30 Hz) frame for every camera
video, rig pose, quality, hand and projection row (ties go to the earlier
frame). RGB therefore lands on about every third index; native sample order
and `video_time` retain each camera's own clock. No IMU sample gets a frame index.
`property:capture:clock_source` states these rules.

Native rates (measured on all 10 staged sequences): RGB 10.001 Hz; the four SLAM
cameras 29.997–30.000 Hz, each on its own clock; `imu-left` ~800–803 Hz,
`imu-right` ~793–806 Hz; MPS closed-loop trajectory 1 kHz (cook_0, clean_0) or
~802 Hz (the other files); MPS hand tracking 30 Hz.

## Raw inventory

| Shipped file or stream | Rate / clock | Layer | Entity or not ingested: reason |
| --- | --- | --- | --- |
| `video.vrs` 214-1 `camera-rgb` (H.265 all-intra, 2560×1920 yuv420p) | 10 Hz, capture_timestamp_ns | base | `/world/rig_00/cam_00/pinhole/video` (AV1) |
| `video.vrs` 1201-1..4 `slam-front-left/right`, `slam-side-left/right` (H.265 all-intra, 512×512 gray) | 30 Hz each, own clock | base | `/world/rig_00/cam_01..04/pinhole/video` (AV1) |
| `video.vrs` file tag `calib_json` (factory calibration) | static | base | pinholes + `cam_0N` transforms; full FISHEYE624 in `projection_params` |
| `video.vrs` 1202-1 `imu-left`, 1202-2 `imu-right` | ~800 Hz, own clocks | base | `/world/rig_00/imu_0N/{gyro,accel}` |
| `video.vrs` 211-1/-2 eye-tracking cameras (H.265 200×200, 5 Hz) | 5 Hz | — | not ingested: eye gaze is a later layer |
| `video.vrs` 1203-1 `mag0`, 247-1 `baro0`, 246-1 temperature, 248-1 PPG, 500-1 ALS, 240-1 battery | 1–129 Hz | — | not ingested: out of scope (IMU only) |
| `video.vrs` 231-1 `mic` (opus 8 ch 16 kHz) | 50 Hz blocks | — | not ingested: no audio layer |
| `video.vrs` 281-*, 282-1, 283-1 GPS / Wi-Fi / Bluetooth | ~1 Hz | — | not ingested: location/beacon data out of scope |
| `video.vrs` 285-* time-domain mapping | 0–1 records | — | not ingested: device clock needs no mapping |
| `video.vrs` 371-1 on-device hand tracking, 371-2/-3 on-device VIO, 373-1 on-device eye gaze | 30 / 10 / 800 / 30 Hz | — | not ingested: MPS products are the ground truth; on-device gaze is a later layer |
| `mps/slam/closed_loop_trajectory.csv` | 1 kHz or ~802 Hz | base | `/world/rig_00` transform (every row), `/world/rig_00/quality` (`quality_score`) |
| `mps/slam/closed_loop_trajectory.csv` velocity, gravity, ECEF, `utc_timestamp_ns`, `graph_uid` columns | same | — | not ingested: gravity is constant (0, 0, −9.81) = Z-up world; one graph per file |
| `mps/slam/open_loop_trajectory.csv` | ~800 Hz | — | not ingested: closed loop is the product |
| `mps/slam/online_calibration.jsonl` | 30 Hz | — | not ingested: static factory calibration chosen (differs ≤ 0.75 mm, 0.05°, 0.14 px); time-varying calibration is a later layer |
| `mps/hand_tracking/hand_tracking_results.csv` landmarks + confidence | 30 Hz | hand_pose | `/world/gt/coco133_xyz`, `/world/gt/hands/<side>/confidence` |
| `mps/hand_tracking/hand_tracking_results.csv` wrist pose, palm/wrist normals | 30 Hz | hand_pose | `/world/gt/hands/<side>/wrist`, `/world/gt/hands/<side>/{palm_normal,wrist_normal}` |
| `mps/hand_tracking/summary.json` | static | — | not ingested: aggregate statistics of the CSV |
| `<seq>_main_recording.vrs` | — | — | not ingested: byte duplicate of `video.vrs` (sizes equal, sampled windows and SHA-1 equal) |
| `<seq>_preview_rgb.mp4` (cook_0, walk_1) | — | — | not ingested: preview re-encode of the RGB stream |
| depth, scene, heart_rate, diarization, hand_object_interaction, mps_slam_points, mps_slam_summary, mps_artifacts | — | — | not ingested: not downloaded (URLs expired) |
| `_simplecv/` | — | — | not ingested: simplecv preprocessing output, never read |

## Layers and entities

**base.** `/` carries `ViewCoordinates.RIGHT_HAND_Z_UP` and the root
`AnnotationContext`. One moving ego rig `/world/rig_00` ("Aria Gen2 device"): its
`Transform3D` is `world_T_device` for every closed-loop row. MPS starts ~0.8 s
after the first RGB frame in 8 of 10 sequences. Before the first row, after the
last row and across any gap over 2 ms (1 ns after each run's last row), the rig
logs NaN translation and NaN `mat3x3`, which hide the rig subtree while
leaving the 2D panes intact. The first hide marker uses the earliest logged
camera or kept IMU stamp; later markers occur 1 ns after each run ends.
Markers are kept only where `Trajectory.at` returns a missing pose.
Rerun 0.38.1 treats a NaN quaternion as an invalid transform and falls back
to identity (drawing at the parent origin), so dense tracks use rotation
matrices instead. The pose is never held or clamped. A row with a non-finite
value or a quaternion more than 1e-5 off unit norm is missing too, never
renormalised (shipped trajectory quaternions sit ~1e-9 off; hand wrists, printed
with six decimals, up to 1.2e-6). `quality_score` (1.0 / 0.5 / 0.0) is
logged as a scalar at `/world/rig_00/quality`; the pose stays as shipped.

Cameras `cam_00` = `camera-rgb`, `cam_01` = `slam-front-left`, `cam_02` =
`slam-front-right`, `cam_03` = `slam-side-left`, `cam_04` = `slam-side-right`. Each
has its factory `T_device_camera` and a pinhole approximation of FISHEYE624, so the
thin-prism terms are dropped there. Its `AnyValues` hold the stream id, the codec
settings and the full 15 FISHEYE624 parameters. The factory `camera-rgb` model
is for the full 4032×3024 sensor, while the stream is the 2560×1920
`pov_downscaled` image. The converter reads the stream size from the VRS
configuration record and rescales the model uniformly (`CameraCalibration.rescale`).
The result equals MPS's online calibration (f 1112.73 vs 1112.73 px on clean_0).
projectaria-tools' VRS provider rescales by about 0.635 and gives 0.14–0.16 px
more. Video is the VRS H.265 access units, piped unchanged into ffmpeg (software
`hevc` decode), then AV1 NVENC at CQ 36, GOP 60, no B-frames. RGB keeps colour;
SLAM gray becomes yuv420p with neutral chroma. Sample times are the VRS capture
stamps. IMU: raw `gyroscope` (rad/s) and `accelerometer` (m/s²) samples read
through projectaria-tools (`aria.read_imu`, the package's one IMU reader); a
record whose accel or gyro valid flag is false is dropped from both channels
(none in the ten local sequences). Each IMU has its factory `T_device_imu`.

**hand_pose.** Every MPS hand row at 30 Hz. The 21 device-frame landmarks go to
world with `world_T_device` at the row's own timestamp: SE(3) interpolation
(slerp + lerp) between the two bracketing trajectory rows, only when both are
valid and ≤ 2 ms apart; otherwise the hand is missing. The landmarks map to
COCO-133 via `dataforge.hands` (the Assembly-Hands/UmeTrack order: the thumb-base
slot is a derived midpoint and the wrist slot a copy, as in the shared mapping).
Confidence: `*_tracking_confidence == -1` means the hand is absent: NaN positions,
confidence 0.0. A present hand keeps its shipped confidence, including exactly
0.0 (cook_0: 27 rows; walk_1: 397 rows), and its positions. The shipped per-hand
score, −1 included, is at `/world/gt/hands/<side>/confidence`. The wrist pose
(`world_T_wrist`) is at `/world/gt/hands/<side>/wrist`; palm and wrist normals,
rotated to world, are at `/world/gt/hands/<side>/{palm_normal,wrist_normal}`.
There is no shipped 2D.

**projections** (fisheye exception, derived). `coco133_xyz` is projected through
each camera's full FISHEYE624 model (thin-prism included) with the hot3d
`project_keypoints` path (projectaria-tools `CameraCalibration.project`), using
the hand row's own `world_T_device`. Results go to
`/world/rig_00/cam_0N/pinhole/coco133_uv_projected` on the hand clock, with the
joint's confidence. NaN + 0.0 when missing, behind the camera or outside the
valid field of view. A recording property marks the layer as derived
(`derived_from=coco133_xyz`, `camera_model=FISHEYE624`, `calibration_source=VRS factory`).

**Blueprint.** `exoego_blueprint`: an ego column with RGB and the four SLAM panes.
Each pane shows its video and `coco133_uv_projected` only, never the 3D content.
The 3D view is `/world/**` with an orbital eye fitted to the trajectory: the target
is the centre of the 5th–95th percentile box of the valid rig positions, lowered by
0.3 m to where the hands are. The eye sits 1.1 × that box's diagonal away (at least
1 m) along (1, −1.5, 1). Table cards decode the RGB camera
only (`table_blueprint`); fields: sequence, version, RGB frames.

No meshes and no objects are shipped. Eye gaze is not ingested now (a later layer).

## Differences from simplecv

simplecv (`packages/simplecv/simplecv/data/exoego/aria_gen2_pilot.py`,
`data/ego/aria_gen2_pilot_ego.py`, `apis/preprocess_aria_gen2_pilot.py`) against
this port (audit `/tmp/fleet-artifacts/exoego-audit/synthesis.md` §3):

| Difference | Evidence | Audit defect fixed |
| --- | --- | --- |
| Hands at 30 Hz: every MPS row | simplecv keeps one row per RGB frame (`aria_gen2_pilot.py:180-211`) | (1) 30 → 10 Hz downsample |
| SLAM cameras and rig pose on their own clocks | simplecv samples every camera pose at RGB stamps (`aria_gen2_pilot_ego.py:103-123`) | (1), (3) RGB-as-canonical clock |
| No clamp at the clip head: before MPS starts, pose and hands are missing | simplecv clamps both searches (`aria_gen2_pilot.py:200-205`, `hot3d_utils.py:451-452`): 8–9 RGB frames get the first sample held backwards up to 866 ms | (2) head clamp |
| Singular / non-finite pose → missing, never held | simplecv reuses the previous pose, starting from identity (`aria_gen2_pilot_ego.py:150-175`); no such row exists in the shipped data, so a unit test writes one into a CSV | (2) held pose |
| Hand-row pose interpolated at the hand's own time | simplecv takes the nearest-previous trajectory row (`hot3d_utils.py:451`) | raw parity 0.64–0.80 mm, see below |
| Present hands with confidence 0.0 kept (positions, confidence 0.0) | simplecv drops `conf <= 0` (`aria_gen2_pilot.py:175-178`) | binding brief: present joint = shipped confidence |
| Full FISHEYE624 (thin prism) from the VRS factory calibration, rescaled to the stream | simplecv uses the first online-calibration line and drops s0–s3 (`preprocess_aria_gen2_pilot.py:204`, `aria_gen2_pilot_ego.py:143-147`) | fisheye exception |
| Projections through the full lens at `coco133_uv_projected`; no `coco133_uv` | simplecv writes reprojected 2D at `coco133_uv` | binding brief: `coco133_uv` is shipped-only |
| Native capture stamps | simplecv writes CFR mp4s at an integer fps and rebases labels to RGB frame 0 (`preprocess_aria_gen2_pilot.py:110-115`, `aria_gen2_pilot.py:272`) | clock rules |
| IMU ingested | simplecv extracts only the 5 image streams (`preprocess_aria_gen2_pilot.py:51`) | (4) |
| RGB colour kept at CQ 36 | simplecv's docstring calls RGB "monochrome Rext" (`preprocess_aria_gen2_pilot.py:3-6`); RGB is 4:2:0 Main Still Picture | transcode rule |

**Parity** (`tests/test_aria_gen2_pilot_real.py::test_simplecv_parity_after_documented_pose_correction`,
golden). For each of simplecv's rows at RGB stamps, the test takes the hand row
simplecv picked and skips simplecv's clamped head rows and its `conf <= 0` hands.
cook_0: 3,341 rows, 3,280 compared, raw max difference 0.803 mm. clean_0: 3,302
rows, 3,022 compared, raw max 0.640 mm. The whole raw difference is simplecv's
nearest-previous pose: after applying the SE(3) correction
`ours_pose @ inv(simplecv_pose)` to simplecv's points, the max error is 3.1e-7 m
(cook_0) and 4.3e-7 m (clean_0), inside the 1e-4 m gate. The missing mask is
identical row by row. Projections vs projectaria-tools' own `project` on every
third hand row, all joints, all 5 cameras: ≤ 1e-6 px (golden).

Pixel evidence (headless viewer 0.38.1, all three layers streamed into one recording,
default blueprint; the hand projections sit on the hands):

- cook_0, t = 2085.5 s, default layout: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-cook_0-mid-default.png
- cook_0 camera-rgb, full size: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-cook_0-mid-camera-rgb.png
- cook_0 slam-front-right, full size: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-cook_0-mid-slam-front-right.png
- clean_0, t = 1364.9 s, default layout: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-clean_0-mid-default.png
- clean_0 slam-front-right, full size: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-clean_0-mid-slam-front-right.png

## Timing

Iteration conversions, 2026-09-25 on pablo-dl-server, **dev env** (beartype on),
converter `1+8a04cfb42688` plus this branch's uncommitted changes, all three layers,
`--force`, read from local disk (cook_0 over NFS), under the shared GPU lock.
From `<output_root>/timing/convert.jsonl`:

| sequence | capture | fetch | transcode | write:base | write:hand_pose | write:projections | total | s per capture-minute | bytes base / hand_pose / projections |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cook_0 | 334.2 s | 14.3 s | 22.6 s | 51.3 s | 0.5 s | 5.4 s | 71.5 s | 12.8 | 273.3 MB / 8.6 MB / 13.6 MB |
| clean_0 | 330.3 s | 16.2 s | 28.8 s | 34.2 s | 0.1 s | 4.7 s | 55.2 s | 10.0 | 307.5 MB / 7.9 MB / 11.8 MB |

The first run (same code apart from the 3D eye) took 47.1 s (cook_0) and 50.7 s
(clean_0); the difference is host load. Transcode overlaps write:base (the stage
timers nest), so the stages sum to more than the total. The simplecv baseline
(preprocess + conversion, prod env, same sequences) and the prod-env timing are
still to be measured for the benchmark report.
