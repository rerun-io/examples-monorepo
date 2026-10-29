# Gen-HumanWholeBody (GenRobot DAS-Ego)

## Source

- Hugging Face dataset `genrobot2025/Gen-HumanWholeBody`, pinned at commit
  `22a037591598f07992b2488e5eacc62b06a1036d` (`genrobot_source.REVISION`). The repository
  changed on the day of the port, so `--revision` accepts only a full commit sha. The publisher later rewrote its history:
  the first pin, `79fa81bf`, no longer exists. The 48 pilot rrds still name it in `hf_revision`; checked 2026-10-26, the 46
  pilot MCAPs fetched at it have the same SHA-256 at `22a03759`, and the release has the same 15,943 files and 8.79 TB.
- 15,943 MCAP files, 8.79 TB, one per episode, under two path layouts:
  `data/<a>/<b>/<shard>/<uuid>.mcap` (17 top-level domains) and
  `data/<a>/<b>/<c>/<d>/<shard>/<uuid>.mcap` (6). The shard is the uuid's first two characters.
  There are no calibration or annotation sidecars: everything is inside the MCAP.
- Catalog name `genrobot`; review subsets register with `--catalog-name dataforge-genrobot-sample`.
  The recording id is `genrobot__<uuid>`. The domain directories are properties, so a later
  change to the folder tree does not change an id.
- `property:capture:*`: `hf_revision`, `clock_source`, `source_resolution`,
  `source_start_time_ns` (the `video_time` origin), `reference_camera`, `num_reference_frames`,
  `duration_s`, `sps_constraint_set3_cleared`, `accelerometer_source_unit`.
- `property:episode:*` (null when the episode lacks the topic it comes from: older firmware ships
  `system_info` without `uuid`, for example): `task` (the shipped instruction), `sst` (the domain path as GenRobot
  spells it), `domain_0` … `domain_3` (null below the depth of a short path), `video_id` (the
  device recording uuid), `num_segments`, `num_sub_segments`, `is_success` (every sub-segment
  succeeded; null without sub-segments), `firmware`, `hardware_version`, `device` (for example
  `DAS Ego V7`), `device_name`, `product_model_id`, `recording_start_trigger` and `abnormal_reason` (firmware 2.2 and
  later), `calibration` (the reference camera's calibration name and version), and the invalid
  stretches as `invalid_codes`, `invalid_start_s`, `invalid_end_s`, `invalid_messages`.
- Every camera node carries `source_width` 1600, `source_height` 1300, `video_codec` `h264` and
  `source_num_frames` (the cameras differ by a frame or two), through `log_camera_source`. There
  is no `cq` or `gop`: the source bitstream is remuxed. Beside it the node carries the calibration
  as shipped: `camera_model` `ds`, `ds_coefficients` `[fx, fy, cx, cy, xi, alpha]`,
  `calib_T_cam` (`T_b_c`), `calibration_frame_id`, and `skipped_leading_samples`.

## Get the raw data

```bash
pixi run -e dataforge --frozen dataforge-download genrobot      # writes data/raw/genrobot/manifest.json
export DATAFORGE_OUTPUT_ROOT=/home/pablo/exoego-data/genrobot-pilot/rrd   # local disk, never the NAS
pixi run -e dataforge --frozen dataforge-convert genrobot --per-domain 2 --sequences <uuid> ...
pixi run -e dataforge --frozen dataforge-register --catalog-name dataforge-genrobot-sample genrobot
```

- `download` reads only the pinned listing and writes `manifest.json` (path, size and LFS
  sha256 of every episode). It fetches no MCAP.
- `convert` fetches one MCAP at a time into the raw root, the next one on a background thread,
  checks each against its LFS sha256 (`transports.hf_fetch_verified`), converts it, and deletes
  it once its base layer is published. `--keep-raw` keeps it. The raw root is scratch; at most
  two MCAPs (up to 2.3 GB each) are on it at once.
- `--per-domain N` takes the first N episodes of every top-level domain in uuid order (uuids are
  random hex, so this is a stratified random sample). `--sequences` adds named episodes.
- Write the rrds to local disk and copy them to the NAS afterwards, paced, as one step. Do not
  convert straight onto the NAS while catalogs read from it.

## Raw inventory

Every episode has the same 26 topics. All of them share one clock (see *Clocks*).

| Shipped topic | Rate | Layer | Entity, or not ingested: reason |
| --- | --- | --- | --- |
| `/robot0/sensor/camera{0..5}/compressed` (`foxglove.CompressedImage`, H.264, 1600×1300) | 30 Hz | base | `rig_00/cam_0N/pinhole/video` (`VideoStream`); exposure fields at `rig_00/cam_0N/exposure` |
| `/robot0/sensor/camera{0..5}/camera_info` (double sphere, plus `T_b_c`) | once | base, sidecar | `rig_00/cam_0N`: `rig_T_cam`, a frustum `Pinhole`, the coefficients as shipped |
| `/robot0/sensor/camera{2,3}/camera_info_resize` | once | — | not ingested: describes a 960×780 image the file does not contain |
| `/robot0/vio/eef_pose` (`foxglove.PoseInFrame`) | 30 Hz | base, sidecar | `rig_00` `Transform3D` = `world_T_rig` |
| `/robot0/vio/relative_eef_pose` | 30 Hz | — | not ingested: identity on every frame |
| `robotx/motioncapture/human_pose_fuse_smooth_concat` (`foxglove.PosesInFrame`, SMPL-X body22) | 30 Hz | pose (sidecar) | `world/gt/body/smplx22`, COCO-133 body slots; rotations not ingested (see *Layers*) |
| `/robot0/handtracking/{left,right}` (21 bones, MANO, flags) | 30 Hz | pose, hand_mesh (sidecar) | COCO-133 hand slots, `world/gt/hands/<side>/{wrist,mano,quality,confidence,mesh}` |
| `/robot0/sensor/imu` | 200 Hz | base, sidecar | `rig_00/imu_00/{gyro,accel}` |
| `/robot0/sensor/audio` (16 kHz stereo PCM16, 64 ms chunks) | 15.6 Hz | sidecar | not in the recording: exoego:v2 has no audio slot; newer Rerun releases log audio, so a later layer can add it from the sidecar |
| `/robot0/annotation_v2` | once | base, actions (sidecar) | `task/instruction`, `task/actions/{coarse,fine}`, `property:actions:*`, `property:episode:*` |
| `/robot0/frame_validity` | 30 Hz | base, sidecar | `/frames` |
| `/robot0/time_range_validity` | once | base, sidecar | `property:episode:invalid_*` |
| `/robot0/sim/robot_info` | 30 Hz | — | `robot_name` → `property:episode:device_name`; its `base_link` is the VIO pose, bit for bit |
| `/robot0/system_info` | ~1 Hz | base, sidecar | `property:episode:*` from the first message; only the CPU load varies |

The sidecar (`sidecars/<recording_id>/source.rrd`, about a tenth of the MCAP) keeps every
ingested non-camera topic protobuf-decoded, with each topic's schema and the MCAP's
statistics. The derived layers are built from it alone.

## Clocks

- `log_time`, `publish_time` and `header.timestamp` are equal on every message of every topic.
- `video_time` = `log_time` minus the MCAP's first message (`McapStatistics:message_start_time`,
  kept in the sidecar), in nanoseconds; the origin is `property:capture:source_start_time_ns`.
- `frame_index` is the shipped `frame_validity.frame_index` of camera2, the reference camera
  `frame_validity` names. Every other stamp takes the number of the nearest reference frame plus the
  whole frames, at 30 Hz, between them: a stamp on a reference frame gets its number (the six
  cameras fire within 0.16 ms of each other), a stamp in a gap of `frame_validity` gets the number
  the gap skipped, and a camera that starts a frame early gets `first - 1`. Body, hands and
  `frame_validity` are on camera2's stamps exactly; the VIO pose is within 0.4 µs and often misses
  the first and last frame. Action bounds are frame numbers on the same clock.
- The IMU keeps its own 200 Hz stamps and gets no `frame_index`.
- `/frames:source_timestamp_s` is the schema's float column; the exact stamp is beside it as
  `source_timestamp_ns`.

## Layers and entities

| Layer | Entities |
| --- | --- |
| `base` | Root `ViewCoordinates` (`RIGHT_HAND_Z_UP`) and the COCO-133 annotation context; `rig_00` (`das_ego`, `reference = imu_00`) with temporal `world_T_rig`; six cameras; `imu_00`; `/frames`; `task/instruction` |
| `pose` | `world/gt/coco133_xyz`; `world/gt/body/smplx22` with its own annotation context; per hand `wrist`, `confidence`, `mano` (`betas`, `global_orient_rotmat`, `hand_pose_rotmat`, `global_transl`), `quality` (`is_valid`, `reproj_loss`, `bbox_touches_edge`) |
| `actions` | `task/actions/coarse` (segments) and `task/actions/fine` (sub-segments) as `StateChange` tracks: one state per lane, an empty state where no label is active; every field as `property:actions:*` |
| `projections` | per camera `pinhole/coco133_uv_projected` and `pinhole/smplx22_uv_projected` |
| `hand_mesh` | per hand `world/gt/hands/<side>/mesh`, MANO skinned per frame |

A derived layer with nothing to carry is not published: `actions` without annotated segments,
`pose` without body or hand data, `hand_mesh` when no hand passes its MANO check. The next
convert checks again. The reference frames and the six calibrations are the only required topics;
an episode without them fails and is reported in `timing/convert.jsonl`, as does one whose camera
has no keyframe at all. A camera that starts late is read from its first keyframe, however late.

Conventions solved from the data (no GenRobot document describes them):

- Quaternions are `xyzw`. The VIO body frame is forward-left-up; the world is Z up, metres, and
  starts at the device's first pose with the wearer's first heading.
- `world_T_cam = world_T_rig · RIG_FROM_CALIB · T_b_c`, with `RIG_FROM_CALIB =
  [[0,0,-1],[1,0,0],[0,-1,0]]`. The cameras fan out at +42, +23, 0, 0, −22 and −43 degrees;
  camera2 and camera3 are a front stereo pair 6 cm apart. A test checks the fan on both samples.
- The IMU frame is the calibration body frame: on devices V3 to V7 (six pilot episodes) the gyro
  matches the VIO angular rate under `RIG_FROM_CALIB` within 0.7 degrees with no time offset, and
  the accelerometer reads gravity within 0.6 degrees of +z. So `rig_T_imu` is `RIG_FROM_CALIB` at the rig
  origin, and the rig node names `imu_00` as its reference. The accelerometer ships in g; dataforge
  multiplies by 9.80665.
- The camera bitstream is passed through. Two source faults are handled: the SPS
  `constraint_set3_flag` toggles between keyframes (the viewer rejects a changing codec
  string), so the flag is cleared; and some streams start before their first keyframe, so those
  samples, which no decoder can show, are skipped and counted.
- Body and hand rows may be null or empty, and a few frames ship a hand's bones with null MANO
  lists (seen on 34 of the 48 pilot episodes). Such a frame is absent for what it lacks: the
  bones still fill COCO-133, and the mesh skips the frame. Each frame is skinned with its own
  shipped betas.

Body: the track is an SMPL-X-shaped skeleton, not SMPL-X parameters. The joint positions are
good. The rotations are not: no forward-kinematics convention keeps a constant rest skeleton
(bones drift 4 cm median, 15 cm worst), and there are no betas or translation. The layers
therefore carry the positions only; the rotations stay in the sidecar. The body's wrists are the
hand tracker's wrists exactly.

COCO-133: body22 fills slots 5–16 (shoulders to ankles) through simplecv's
`smplx_joints_to_coco133_xyzc`; the face (0–4) and feet (17–22) slots stay empty. Each hand's
21 bones fill its 21 hand slots (the bones ship in COCO hand order). Confidence is 1.0 where the
source has a point: no confidence is shipped. Per-hand `confidence` is 1.0 on frames with the
hand's message. The shipped `is_valid` is False on every frame seen and `reproj_loss` is 0, so
they are kept verbatim under `quality` and not used.

MANO: full-pose MANO (no PCA) with `global_orient`, the 15 finger rotations and `global_transl`
lands on the shipped bones. The left hand needs the known MANO left-model shape fix (the first
shape direction's x negated: simplecv's `MANOLayerNP(..., fix_left_shapedirs=True)`) and its finger rotations
mirrored (`M·R·M`, `M = diag(-1, 1, 1)`). EPFL and HO-Cap pose their left hands without the fix; whether
their fits need it is not checked.
`hand_mesh` checks each hand of each episode and writes its mesh only when the median distance
from the MANO joints to the shipped bones is under 2 mm; `property:hand_mesh:*` records both
distances. On the 48 pilot episodes the right hand is 0.00 mm on every episode; the left median is
0.07 to 2.99 mm, so one left mesh fails the gate and is not written. On both hands the
fingertips are 1 to 4 mm off.

2D: the source ships no 2D keypoints, so there is no `coco133_uv`. The `projections` layer
projects `coco133_xyz` and the body22 skeleton through each camera's double-sphere model
(`property:projections:camera_model = "ds"`); points outside the image are dropped. The cameras
are fisheye, so their panes show the projected 2D and leave out `coco133_xyz`, which a Rerun
`Pinhole` would draw through the wrong projection.

Default blueprint: the 3D view on the world, with an upright three-quarter eye on the wearer, over
the task and a state timeline of the coarse and fine actions (hover a segment for its full label and
frames); the six cameras on the right in a two-column grid, in yaw order. Table card: the wearer in 3D without video, beside camera2.

## Differences from simplecv

simplecv has no loader for this dataset, so there is nothing to differ from.

## Parity with simplecv

Not applicable (see above). Pixel evidence for the conventions is in the PR.

## Timing

Read `<output_root>/timing/convert.jsonl` with `load_records(path, ConvertRecord)`.
On pablo-dl-server, prod environment, 2026-10-23:

| Episode | Capture | MCAP | Fetch | Convert (all layers) | Output |
| --- | --- | --- | --- | --- | --- |
| barber shop (8 s) | 8.1 s | 26 MB | local | 2.3 s | 31 MB |
| ping-pong (3 min) | 179 s | 1.47 GB | 74 s | 12.5 s | 1.57 GB |
| pilot, 48 episodes | 1.91 h | 27.0 GB | overlapped | 2.7 s per capture minute | 30 GB |

The Hub fetch (about 20 MB/s) dominates; the next episode is fetched while the current one
converts. The base layer is collected for the object-store optimization, which peaks at about
14 GB of memory for a 3-minute episode; run long jobs under a memory cap.

Pilot output by layer: base 25 GB (about 225 MB per capture minute), hand_mesh 3.6 GB,
projections 0.7 GB, pose 0.6 GB, actions under 1 MB; the sidecars add 2.0 GB.

## Known gaps

- Audio is in the sidecar only.
- No SMPL-X body fit, so no body mesh; the face and feet COCO slots are empty.
- The hand flags (`is_valid`, `reproj_loss`, `bbox_touches_edge`) look unpopulated; revisit if a
  later release fills them.
- The full corpus is 8.8 TB raw and about the same converted (the video is passed through); only
  a subset fits the NAS.
