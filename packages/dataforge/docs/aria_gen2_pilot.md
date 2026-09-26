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
| `mps/slam/online_calibration.jsonl` | 30 Hz | — | not ingested: static factory calibration chosen (line 0 differs by 0.55–1.19 mm, 0.05–0.11°, principal point 0.8–1.2 px on the SLAM cameras; camera-rgb equal); time-varying calibration is a later layer |
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
| Hand-row pose interpolated at the hand's own time | simplecv takes the nearest trajectory row (`hot3d_utils.py:441-461`) | raw parity 0.64–0.80 mm, see below |
| Present hands with confidence 0.0 kept (positions, confidence 0.0) | simplecv drops `conf <= 0` (`aria_gen2_pilot.py:175-178`) | binding brief: present joint = shipped confidence |
| Full FISHEYE624 (thin prism) from the VRS factory calibration, rescaled to the stream | simplecv uses the first online-calibration line and drops s0–s3 (`preprocess_aria_gen2_pilot.py:204`, `aria_gen2_pilot_ego.py:143-147`) | fisheye exception |
| Projections through the full lens at `coco133_uv_projected`; no `coco133_uv` | simplecv writes reprojected 2D at `coco133_uv` | binding brief: `coco133_uv` is shipped-only |
| Native capture stamps | simplecv writes CFR mp4s at an integer fps and rebases labels to RGB frame 0 (`preprocess_aria_gen2_pilot.py:110-115`, `aria_gen2_pilot.py:272`) | clock rules |
| IMU ingested | simplecv extracts only the 5 image streams (`preprocess_aria_gen2_pilot.py:51`) | (4) |
| RGB colour kept at CQ 36 | simplecv's docstring calls RGB "monochrome Rext" (`preprocess_aria_gen2_pilot.py:3-6`); RGB is 4:2:0 Main Still Picture | transcode rule |

## Parity

Checked 2026-09-25 on cook_0 and clean_0, rrd against rrd: simplecv's own
recordings (`exoego-forge-catalog-rig/aria-gen2/`, and a fresh simplecv run on
local copies, which gives identical numbers) against this converter's prod-env
output at 6f45e78e. Full report with every difference mapped to an audit defect:
`/tmp/fleet-artifacts/exoego-migration/runs/aria_gen2_pilot-parity.md`.

- **`coco133_xyz`.** simplecv's keypoint rows sit on the real RGB stamps (within
  1 ns once its rebase is undone). Each is compared with the hand row simplecv
  picked (nearest to the RGB stamp). cook_0: 3,280 rows compared, raw max
  0.803 mm; clean_0: 3,022 rows, raw max 0.640 mm. The whole raw difference is
  simplecv's nearest-row pose: after the SE(3) correction
  `ours_pose @ inv(simplecv_pose)` the max error is 3.1e-7 m (cook_0) and
  3.8e-7 m (clean_0), inside the 1e-4 m gate. Present confidences are equal.
  No joint is finite in simplecv only. The joints finite in dataforge only
  (242 and 572) all belong to hands with shipped confidence 0.0, which simplecv
  drops. The same check runs as the golden test
  `tests/test_aria_gen2_pilot_real.py::test_simplecv_parity_after_documented_pose_correction`.
- **Camera poses.** simplecv's rig rows sit on a 10 Hz CFR grid that drifts up
  to 20.0 ms (cook_0) / 23.3 ms (clean_0) from the real RGB stamps, so they are
  matched by RGB frame index. In world coordinates `camera-rgb` agrees to
  3.9e-7 m and 7e-6°. The SLAM cameras differ by a constant 0.55–1.19 mm and
  0.05–0.11°; this equals the difference between simplecv's calibration (MPS
  online calibration line 0) and the VRS factory calibration used here, camera by
  camera. The 9 RGB frames before MPS starts have a held pose in simplecv and a
  NaN pose here.
- **Projections** vs projectaria-tools' own `project` on every third hand row,
  all joints, all 5 cameras: ≤ 1e-6 px (golden).

Pixel evidence (headless viewer 0.38.1, the three layer files of one sample
recording streamed into one recording, prod-env output at 6f45e78e). The hand
projections sit on the hands in the RGB and SLAM panes:

- play_0, t = 5197.662 s, default layout: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-play_0-sample-default.png
- play_0 camera-rgb pane: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-play_0-sample-camera-rgb.png
- play_0 slam-front-left pane: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-play_0-sample-slam-front-left.png
- eat_2, t = 2815.018 s, default layout: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-eat_2-sample-default.png
- eat_2 camera-rgb pane: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-eat_2-sample-camera-rgb.png
- eat_2 slam-front-left pane: https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-eat_2-sample-slam-front-left.png
- clean_0 before MPS starts (1200.2 s) and after the last pose (1530.12 s), rig hidden:
  https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-clean_0-fix2-before-tracking.png,
  https://pablos-4800gt.ilish-ruler.ts.net:8768/exoego-migration/evidence/aria_gen2_pilot-clean_0-fix2-after-tracking.png

## Timing

Baseline, 2026-09-25 on pablo-dl-server. Both sides ran in their prod envs on
local copies of the same raw files, one at a time under the shared GPU lock.
simplecv = `preprocess_aria_gen2_pilot.py` (VRS → AV1 mp4) + `batch_raw_to_rrd.py
aria-gen2`, wall time including two pixi start-ups. dataforge = the convert
record's `total_s` for all three layers (`convert.jsonl`, converter
`1+6f45e78e8e91`, `--force`).

| recording | capture s | simplecv preprocess + convert s | simplecv s/capture-min | dataforge s | dataforge s/capture-min | dataforge transcode s | dataforge write:projections s | output (simplecv / dataforge) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aria_gen2_pilot__cook_0 | 334.2 | 24.1 + 4.6 = 28.7 | 5.16 | 34.1 | 6.12 | 22.5 | 5.4 | 164 / 303 MB |
| aria_gen2_pilot__clean_0 | 330.3 | 30.5 + 4.5 = 35.0 | 6.35 | 40.0 | 7.26 | 30.0 | 4.7 | 162 / 334 MB |

**Soft gate: missed by 14–19 %.** The video work is equal: dataforge's
transcode (five streams, CQ 36) takes 22.5 s and 30.0 s, simplecv's preprocess
23.4 s and 29.7 s by its own log (SLAM at CQ 40). The gap (5.4 s and 5.0 s) is
the derived `projections` layer: 30 Hz × 5 cameras × 133 joints through
projectaria-tools' per-point `CameraCalibration.project` (5.4 s and 4.7 s).
simplecv reprojects at 10 Hz with its own Kannala-Brandt code and without
thin-prism terms. Without that layer dataforge is on par (28.7 s vs 28.7 s,
35.3 s vs 35.0 s). dataforge's other base work (every 1 kHz trajectory row, two
~800 Hz IMUs, hands at 30 Hz) takes 3.9 s inside `write:base`, about what
simplecv's whole convert step takes. The lever is a vectorised FISHEYE624
projection (checked against `project` to 1e-6 px); not done in this stage.

Earlier dev-env runs (beartype on) took 50–72 s for the same sequences.
The stage timers nest (transcode runs inside `write:base`), so the stages sum
to more than the total. Registration time is added by the orchestrator.

Sample conversions (prod env, same converter, GPU lock):

| recording | capture s | transcode s | write:base s | write:projections s | total s | s/capture-min | bytes base / hand_pose / projections |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aria_gen2_pilot__eat_1 | 324.3 | 23.2 | 26.6 | 4.5 | 34.8 | 6.43 | 277.1 / 8.3 / 10.7 MB |
| aria_gen2_pilot__eat_2 | 344.3 | 25.4 | 29.3 | 5.6 | 38.2 | 6.65 | 292.0 / 9.6 / 14.1 MB |
| aria_gen2_pilot__eat_3 | 336.6 | 23.9 | 27.3 | 4.4 | 35.5 | 6.32 | 299.0 / 8.2 / 10.5 MB |
| aria_gen2_pilot__play_0 | 341.3 | 26.9 | 30.8 | 2.3 | 36.7 | 6.45 | 307.0 / 4.8 / 5.7 MB |
| aria_gen2_pilot__play_1 | 340.0 | 28.2 | 31.5 | 3.6 | 39.1 | 6.90 | 311.6 / 6.9 / 8.5 MB |
| aria_gen2_pilot__play_3 | 342.4 | 27.9 | 31.5 | 2.9 | 38.2 | 6.70 | 321.1 / 5.9 / 6.5 MB |
| aria_gen2_pilot__walk_0 | 299.5 | 23.0 | 26.1 | 2.2 | 33.4 | 6.69 | 308.7 / 4.7 / 5.1 MB |
| aria_gen2_pilot__walk_1 | 300.7 | 23.5 | 26.6 | 3.1 | 34.6 | 6.90 | 334.8 / 6.1 / 6.7 MB |

Every layer file reads back (one recording per file, chunks and rows > 0).
