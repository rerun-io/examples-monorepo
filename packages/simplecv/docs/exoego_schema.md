# Exo/Ego Rerun Logging Schema (`exoego:v2`, COLMAP-style rigs)

This document is the canonical entity layout for combined exocentric/egocentric
(exo/ego) Rerun recordings. **v2** replaces the flat `/world/{exo,ego}/{name}`
layout with a COLMAP-style **rig** layout: every camera belongs to a **rig** (a
set of sensors with fixed relative poses), and when a rig moves, all of its
cameras move rigidly with it. This is the same model used by
[`live-rerun`](../../live-rerun/docs/rig_schema.md) and
[`slam-evals`](../../slam-evals/docs/schema.md), and it mirrors COLMAP's rig/sensor
concept (<https://colmap.github.io/concepts.html#rigs>).

The shared rig schema types live in `simplecv/rig.py` (re-exported by
`live_rerun.rig`); the logger is `simplecv/rerun_rig_logger.py`; the dataset-agnostic
builder is `BaseExoEgoSequence.build_rig_layout`.

All arrays follow jaxtyping notation and use metres for positions, radians for
axis-angle vectors, and seconds/nanoseconds for timestamps.

## 1. One rig per physical device

- **Exo cameras** — each independent exo camera is its **own static,
  world-anchored rig** (`rig_00`, `rig_01`, …, one camera each). The rig frame
  coincides with world (`world_T_rig` is implicit identity — **no transform on the
  rig node**), so each camera's `rig_T_cam` equals its `world_T_cam`. A future
  multi-sensor exo unit (e.g. a RealSense or OAK with several sensors) simply adds
  more cameras under one exo rig — no schema change.
- **Ego device** — the worn device (Aria, HoloLens, Quest3, HOT3D, UmeTrack, …)
  is **one moving rig** whose `world_T_rig(t)` is the reference camera's
  trajectory. Its cameras are fixed `rig_T_cam` offsets from the reference
  camera (the rig origin, identity `rig_T_cam`). When the loader exposes cameras
  whose relative pose is **not** constant (not rigidly factorable), each ego
  camera falls back to its own single-camera moving rig.
- **Rig indices** — exo rigs take `rig_00..rig_(E-1)` (E = number of exo
  cameras); the ego rig follows at `rig_E`. With no exo cameras the ego rig is
  `rig_00`.

## 2. Transform notation

Right-to-left composition, matching the project-wide rule:

```
cam_points   = cam_T_world @ world_points       # world  → camera
world_T_cam  = world_T_rig @ rig_T_cam           # composes along the entity tree
```

- The **rig node** `/world/rig_NN` carries `world_T_rig` — for a moving rig this
  is a *temporal* `Transform3D` (logged without `from_parent`, so the stored
  value is `world_T_rig`); a static world-anchored rig carries **no** transform.
- The **camera node** `/world/rig_NN/cam_MM` carries the static `rig_T_cam`,
  logged by `simplecv.rerun_log_utils.log_pinhole` as
  `Transform3D(..., from_parent=True)` (the stored parent→child step is
  `rig_T_cam`; the reference camera's is identity).
- A **tracking dropout** is encoded as a **NaN** `world_T_rig` on the rig node for
  that frame; the whole rig — and every child frustum — disappears for the gap.

## 3. Entity tree

```
/                               ViewCoordinates (RDF, static, at the root)
/world
  /rig_00                       static exo rig: AnyValues{schema_version, reference, num_cameras};
                                NO transform (implicit identity)
    /cam_00                     Transform3D = rig_T_cam = world_T_cam (static) + AnyValues{name, kind}
      /pinhole                  Pinhole / PinholeWithDistortion (static)
        /video                  VideoStream (encoded samples, video_time timeline)
        /coco133_uv             Points2DWithConfidence (projected 2D keypoints)
        /depth                  DepthImage (optional)
  /rig_01                       moving ego rig: AnyValues{...} + Transform3D = world_T_rig(t) (temporal)
    /cam_00                     reference camera: Transform3D = identity rig_T_cam (static)
      /pinhole/video, /pinhole/coco133_uv
    /cam_01                     fixed rig_T_cam offset (multi-camera ego devices)
      /pinhole/video, /pinhole/coco133_uv
    /imu_00                     peer sensor (IMU — see §8; emitted by dataforge)
    /mag_00                     peer sensor (magnetometer — see §9; emitted by dataforge)
  /gt                           ground-truth annotations (UNCHANGED from v1, see §5)
```

- Entity ids are **zero-padded to two digits** (`rig_00`, `cam_00`) so they sort
  lexicographically in numeric order.
- All entities use metres and the OpenCV **RDF** (Right-Down-Forward) camera
  convention, logged as `ViewCoordinates.RDF` at `/`.

## 4. Per-rig metadata

`simplecv.rerun_rig_logger.log_rig_static` logs, as static `rr.AnyValues` on each
`/world/rig_NN`:

- `schema_version` = `"exoego:v2"`,
- `reference` = the reference camera's id (e.g. `"cam_00"`),
- `num_cameras`.

Those three keys are the **required** set. A writer may add the two optional
rig-level keys `name` (human device label, e.g. `"robocap"`, `"oak"`, an iPhone's
advertised name) and `kind` (device role: `"exo"` / `"ego"` / `"quest"`) — dataforge
emits both, because a capture with several unlike rigs is unreadable without them
and blueprints cannot select entities by their `AnyValues`. Readers must treat them
as optional. Note also that `reference` names a **sensor child**, not necessarily a
camera: a rig whose extrinsics are all expressed in its inertial frame states
`reference = "imu_00"` (dataforge's RoboCap rig does), and a single-camera rig
trivially states `"cam_00"`.

Per camera, on `/world/rig_NN/cam_MM`: `name` (human stream label) and `kind`
(`"rgb"` / `"grayscale"`, a best-effort content hint). The reference camera of a
**multi-camera** rig gets a green frustum tint; single-camera rigs are untinted.

## 5. Ground-truth annotations (paths unchanged from v1)

GT lives under `/world/gt/...`, independent of the rig layout:

```
/world/gt/coco133_xyz                  Points3D + KeypointConfidence3D  (Float "n_frames 133 3/…")
/world/gt/mano/{left,right}/mesh       Mesh3D (verts metres, shared faces)
/world/gt/mano/{left,right}/...        global_orient / hand_pose / betas / mp_21
/world/gt/env_mesh                     Mesh3D (static environment)
```

### Projected 2D keypoints (per camera, derived)

```
/world/rig_NN/cam_MM/pinhole/coco133_uv   Points2DWithConfidence  (Float "n_frames 133 2")
```

Each camera stores its own 2D projections beneath its `pinhole` entity. Missing
points are `NaN` with confidence `0.0`. A parallel prediction layout under
`/world/pred/...` and `/world/rig_NN/cam_MM/pinhole/pred/coco133_uv` is
**reserved but not emitted by the current writer**.

## 6. Validation rules

When ingesting a recording:

1. Read `schema_version` (`"exoego:v2"`) on any `/world/rig_*` node; refuse older
   revisions (the v1 flat reader, `rrd_exoego.py`, is **deprecated** and cannot
   read v2).
2. Walk `/world/rig_*/cam_*`. Each calibrated camera has a static `Pinhole`/
   `PinholeWithDistortion` and a static `rig_T_cam` `Transform3D`.
3. A **moving** rig has a temporal `world_T_rig` on its rig node; a **static** rig
   has none (implicit identity). Treat a temporal transform on an exo rig as an
   error.
4. GT tensors resolve under `/world/gt/...` when `config.load_labels` is true.
5. Timeline is `video_time` everywhere.
6. Every non-camera peer sensor (`/world/rig_*/imu_*`, `/world/rig_*/mag_*`) has a
   static `Transform3D` (`rig_T_imu` / `rig_T_mag`) and a static `kind`
   (`"imu"` / `"mag"`); a sensor node without its transform is an error, because a
   reader then cannot place its samples in the rig frame. A magnetometer's `field`
   is in the sensor's native units, which are only known when it carries a `unit`
   AnyValue — treat an absent `unit` as uncalibrated counts, never as tesla. The
   optional `heading` child is derived, so a reader may ignore it entirely.

The read side of these rules is `simplecv/catalog_rig_layout.py`: `parse_rig_layout`
turns a catalog schema back into typed cameras (video stream, moving rig, rig `kind`,
calibration presence, camera-node markers). Consumers add only their selection policy
on top of it instead of parsing entity paths themselves.

## 7. Dataset author checklist

Datasets need **no** per-dataset rig code — `build_rig_layout` derives everything
from the normalized `exo_cam_list` / `exo_video_names` and `ego_cam_dict` /
`ego_video_names`. To add or regenerate a dataset:

- [ ] Provide calibrated `PinholeParameters` / `Fisheye62Parameters` per camera
      (ego per-frame `world_T_cam`; exo static `world_T_cam`). Invalid ego frames
      carry NaN extrinsics.
- [ ] Emit GT (COCO-133, MANO) in metres under `/world/gt/...`.
- [ ] Regenerate via the catalog generator (`batch_raw_to_rrd` → `visualize_exo_ego`).
- [ ] Verify: `rerun rrd print <file>.rrd` shows `/world/rig_NN/cam_MM`, a
      temporal `world_T_rig` on the ego rig, and `schema_version=exoego:v2`; no
      `/world/{exo,ego}/*` remain.

## 8. IMU *(emitted — first writer: dataforge / RoboCap dev0)*

The rig model treats **every sensor as a peer child of the rig**, so a non-camera
sensor slots in alongside the cameras without nesting under one. The first such
sensor is the **IMU**: ego devices like **Project Aria** (RGB + SLAM cameras *and* IMUs) and
the **RoboCap** capture rigs carry inertial data, and `SensorKind` in
`simplecv/rig.py` already reserves `"imu"` for exactly this.

**Layout** (mirrors `slam-evals`' `-vi` inertial modalities). `dataforge`'s RoboCap
converter (`packages/dataforge/dataforge/datasets/robocap.py`) is the first writer that
actually emits it; simplecv's own exo/ego writer still does not.

```
/world/rig_NN/imu_MM        Transform3D = rig_T_imu (static) + AnyValues{name, kind="imu"}
  /gyro                     Scalars (3-component, rad/s)  — angular velocity, video_time
  /accel                    Scalars (3-component, m/s²)   — linear acceleration, video_time
```

- The IMU is a **peer of the cameras** (`/world/rig_NN/imu_MM`), **not** nested under
  a camera, carrying its own static `rig_T_imu` offset in the rig frame — exactly the
  `slam-evals` convention (its `imu_0` sits beside `cam_0`). `imu_MM` is zero-padded
  via `entity_id("imu", j)` (`INDEX_WIDTH=2`) and peer-indexed independently of the
  cameras.
- A rig may carry **several** IMUs (`imu_00`, `imu_01`, …) — e.g. Aria's two IMUs —
  just as it carries several cameras.
- On a **moving** ego rig the IMU rides the rig's `world_T_rig(t)` like every other
  sensor; its `gyro`/`accel` samples are logged on the shared `video_time` timeline,
  as `slam-evals` does.
- Samples are logged **raw, at their native rate, without interpolation or
  resampling**, as columnar `rr.Scalars` batches (three components per row:
  x/y/z) on `video_time`. Readers must not assume IMU rows line up with video
  frames.
- Adding IMUs is **mechanical** — a new peer entity plus the already-reserved `"imu"`
  kind, no new vocabulary. Emitting IMU data did not change any existing path, so
  the schema version stays `exoego:v2`.
- **RoboCap status / TODO:** dataforge v1 emits the middle IMU (`dev0`) only, as
  `imu_00`; `dev1`/`dev2` and a multi-IMU blueprint layout (one gyro/accel pane pair
  per IMU) are still TODO. RoboCap's IMU and camera clocks differ by a fixed
  14,902,432 ns offset (basalt's `kCameraToImuOffsetNs`); dataforge picks the raw
  **camera** clock for `video_time` and subtracts the offset from IMU timestamps.

## 9. Magnetometer *(emitted — first writer: dataforge / Monado SLAM Datasets)*

The magnetometer is the second peer sensor, and it needed no new vocabulary beyond
the `"mag"` kind: it is an IMU-shaped stream (timestamps plus xyz) that happens to
measure a field rather than motion. Headsets carry one next to the IMU — the
**Monado SLAM Datasets** ship one per sequence for the Reverb G2 and the Odyssey+ —
and it is the only sensor that observes an absolute heading, so a downstream
yaw-drift evaluation wants it beside the video and the inertial data.

**Layout** (the §8 shape, one entity down):

```
/world/rig_NN/mag_MM        Transform3D = rig_T_mag (static) + AnyValues{name, kind="mag", unit?}
  /field                    Scalars (3-component, sensor's native units) — video_time
  /heading                  Arrows3D (unit field direction × a fixed length) — video_time, derived
```

- The magnetometer is a **peer of the cameras and the IMU** (`/world/rig_NN/mag_MM`),
  with its own mandatory static `rig_T_mag`, exactly as §8 requires of `imu_MM`.
  `mag_MM` is zero-padded via `entity_id("mag", j)` and peer-indexed independently.
- `field` is logged **raw, at its native rate, without interpolation, in the sensor's
  own units**. Consumer headsets ship unlabelled counts, and inventing a calibration
  would be worse than saying so; the optional `unit` AnyValue records the units when a
  dataset actually documents them. MSD's Reverb G2 / Odyssey+ files are unlabelled
  50 Hz xyz whose total field sits around 300 — consistent with milligauss, which is
  an inference and not a claim the files make, so dataforge writes no `unit` for them.
- `heading` is a **derived visualization aid**, not data: the same samples normalized
  and scaled to a fixed length (0.15 m by default) so the field direction is legible
  in the 3D view while riding the rig's `world_T_rig(t)`. Rows whose field norm is 0
  (a dropout) get no arrow rather than a NaN direction. A reader that wants the field
  reads `field`; `heading` may be dropped or regenerated at will.
- Emitting a magnetometer did not change any existing path, so the schema version
  stays `exoego:v2`.
- **Status / TODO:** `SensorKind` in `simplecv/rig.py` now lists `"mag"` beside
  `"imu"`, but simplecv's own exo/ego writer still emits neither; dataforge
  (`packages/dataforge/dataforge/logging_toolkit.py`, `log_magnetometer`) is the only
  writer.

Any change to the layout should increment the schema version and update this doc.
