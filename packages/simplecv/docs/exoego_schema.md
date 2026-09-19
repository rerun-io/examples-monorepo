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
  world-anchored rig** (`rig_00`, `rig_01`, …). The rig frame
  coincides with world (`world_T_rig` is implicit identity — **no transform on the
  rig node**), so each camera's `rig_T_cam` equals its `world_T_cam`. A multi-sensor
  exo unit adds more cameras under one exo rig — no schema change.
- **Ego device** — the worn device (Aria, HoloLens, Quest3, HOT3D, UmeTrack, …)
  is **one moving rig** whose `world_T_rig(t)` is the reference camera's
  trajectory. Its cameras are fixed `rig_T_cam` offsets from the reference
  camera (the rig origin, identity `rig_T_cam`). When the loader exposes cameras
  whose relative pose is **not** constant (not rigidly factorable), each ego
  camera falls back to its own single-camera moving rig.
- **Device-anchored world** — world may be a moving device frame such as a
  back-mounted rig; that rig is a static multi-camera exo rig in it. Physical
  motion, if ever recovered, lives on `/world`.
- **Rig indices** — exo rigs take `rig_00..rig_(E-1)` (E = number of exo
  rigs); the ego rig follows at `rig_E`. With no exo cameras the ego rig is
  `rig_00`.

## 2. Transform notation

Right-to-left composition, matching the project-wide rule:

```
cam_points   = cam_T_world @ world_points       # world  → camera
world_T_cam  = world_T_rig @ rig_T_cam           # composes along the entity tree
```

All stored transforms are relative to `/world`; a temporal `Transform3D` a
later layer writes on `/world` is applied by the viewer to every descendant
and is not part of `world_T_cam`.

- The **rig node** `/world/rig_NN` carries `world_T_rig` — for a moving rig this
  is a *temporal* `Transform3D` (logged without `from_parent`, so the stored
  value is `world_T_rig`); a static world-anchored rig carries **no** transform.
- The **camera node** `/world/rig_NN/cam_MM` carries the static `rig_T_cam`,
  logged by `simplecv.rerun_log_utils.log_pinhole` as
  `Transform3D(..., from_parent=True)` (the stored parent→child step is
  `rig_T_cam`; the reference camera's is identity).
- A **tracking dropout** is encoded as a **NaN** `world_T_rig` on the rig node for
  that frame; the whole rig — and every child frustum — disappears for the gap.
  A source-side pose a writer *repaired* to identity is a different thing and must
  not be emitted as one: it keeps its translation and stays visible, and the count
  belongs in that layer's properties (dataforge's msd reports `num_sanitized`),
  because "the source wrote a degenerate rotation here" and "there was no pose
  here" are claims a consumer has to be able to tell apart.

## 3. Entity tree

```
/                               ViewCoordinates (static, at the root — the recording's world convention)
/task                           recording-level text (§12)
/frames                         per-frame source provenance (§14)
/world
  /rig_00                       static exo rig: AnyValues{schema_version, reference?, num_cameras};
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
  /gt                           ground-truth annotations (v1 paths retained, see §5; additions §10–11)
```

- Entity ids are **zero-padded to two digits** (`rig_00`, `cam_00`) so they sort
  lexicographically in numeric order.
- Positions use metres. Cameras use the OpenCV **RDF** (Right-Down-Forward)
  convention. The root declares the dataset's world convention: `RDF` when
  the world frame is a camera frame. Dataforge's msd/lamaria write
  `RIGHT_HAND_Y_UP`/`RIGHT_HAND_Z_UP` via
  `packages/dataforge/dataforge/world_up.py`; cameras stay RDF.
- See §2 for re-framing the recording with a transform on `/world`.

## 4. Per-rig metadata

`simplecv.rerun_rig_logger.log_rig_static` logs, as static `rr.AnyValues` on each
`/world/rig_NN`:

- `schema_version` = `"exoego:v2"`,
- `reference` = the reference camera's id (e.g. `"cam_00"`),
- `num_cameras`.

`schema_version` and `num_cameras` are required; `reference` is omitted for
a static world-anchored rig whose origin is not a sensor. Readers place its
children by `rig_T_cam` alone. `log_rig_static` still always writes a
reference for single-camera rigs. A writer may add the two optional
rig-level keys `name` (human device label, e.g. `"robocap"`, `"oak"`, an iPhone's
advertised name) and `kind` (device role: `"exo"` / `"ego"` / `"quest"`) — dataforge
emits both, because a capture with several unlike rigs is unreadable without them
and blueprints cannot select entities by their `AnyValues`. Readers must treat them
as optional. Note also that `reference` names a **sensor child**, not necessarily a
camera: a rig whose extrinsics are all expressed in its inertial frame states
`reference = "imu_00"` (dataforge's RoboCap rig does), and a single-camera rig
trivially states `"cam_00"`.

A moving headset rig MAY also carry temporal `AnyValues` on `/world/rig_NN`:
`is_synthesized` (boolean), `pose_source` (the source's string tag), and
`is_pose_valid` (boolean). Copy these per-frame flags only when shipped by the
source; absence means unknown. They describe the rig pose on the same timelines
as that pose and do not replace its `Transform3D` or the dropout rule in §2.

Per camera, on `/world/rig_NN/cam_MM`: `name` (human stream label) and `kind`
(`"rgb"` / `"grayscale"`, a best-effort content hint). Readers must also treat as
optional the three further per-camera keys `camera_model`,
`distortion_valid_radius` and `image_rotation_cw_deg`, which dataforge writes for
the Monado SLAM Datasets.
`camera_model` is the **dataset's own model tag**, copied through uninterpreted
(e.g. `"kb4"` / `"pinhole-radtan8"`, basalt's names): this schema fixes no
vocabulary for it, and a reader that does not recognise a tag falls back to the
distortion component, which is authoritative. `distortion_valid_radius` is the
radius in normalized image coordinates past which that model stops holding; a
writer whose source states a non-positive radius must **omit the key** rather
than emit it, because the formats that carry one (basalt's `rpmax`, whose
non-positive value disables the check) mean "no limit" by it, and a reader
seeing `0.0` would conclude the model holds nowhere.
`image_rotation_cw_deg` is a **clockwise rotation of 90, 180 or 270 degrees that
the writer already applied to this camera's encoded frames**, for a sensor
mounted rolled: the intrinsics, the distortion and the `rig_T_cam` on the same
node describe the rotated image, so a reader that only projects needs nothing
from this key — it is there for one that relates the video back to the raw sensor
readout. A writer that applied no rotation must **omit the key** rather than emit
`0`, which would state a decision where none was made. The reference camera of a
**multi-camera** rig gets a green frustum tint; single-camera rigs are untinted.

## 5. Ground-truth annotations (paths unchanged from v1)

GT lives under `/world/gt/...`, independent of the rig layout:

```
/world/gt/coco133_xyz                  Points3D + KeypointConfidence3D  (Float "n_frames 133 3/…")
/world/gt/mano/{left,right}/mesh       Mesh3D (verts metres, shared faces)
/world/gt/mano/{left,right}/...        global_orient / hand_pose / betas / mp_21
/world/gt/env_mesh                     Mesh3D (static environment)
```

Skeleton class IDs share the root `AnnotationContext` (§6):

| Class ID | Layout | Writers |
|---|---|---|
| 0 | COCO-wholebody 133 | Existing exoego writers |
| 1 | UmeTrack 21-landmark hand | SHOW3D (§10) |

### Projected 2D keypoints (per camera, derived)

```
/world/rig_NN/cam_MM/pinhole/coco133_uv   Points2DWithConfidence  (Float "n_frames 133 2")
```

Each camera stores its own 2D projections beneath its `pinhole` entity. Missing
points are `NaN` with confidence `0.0`. A parallel prediction layout under
`/world/pred/...` and `/world/rig_NN/cam_MM/pinhole/pred/coco133_uv` is
**reserved but not emitted by the current writer**.

### Surveyed control points *(emitted — first writer: dataforge / LaMAria)*

A **surveyed control point** is a point of the world whose coordinates a survey
measured, independently of any capture: LaMAria's `R_11`-onwards sequences ship 5
to 15 of them, tags photographed along the walk and levelled in Switzerland's
LV95/LN02 grid. It is ground truth about the *world*, not about a body in it, so
it sits under `/world/gt/` beside the §5 annotations above, and its per-camera
detections sit under that camera's `pinhole` exactly as `coco133_uv` does:

```
/world/gt/control_points               Points3D + labels (static; positions metres, world frame)
/world/rig_NN/cam_MM/pinhole/cp_uv     Points2D + labels ("n_detections 2", video_time)
```

- The 3D points are **static**: a survey is a property of the world, not of a
  moment. The 2D detections are temporal, one row at the timestamp of the frame
  the tag was detected in, so a detection lands on its own frame.
- Positions are in the recording's own world frame, i.e. after whatever origin
  translation that frame carries (LaMAria subtracts a fixed LV95/LN02 origin so
  metres stay small). A reader treats them as metres like any other position.
- A point the survey **never levelled** has no height. Its `z` is a placeholder,
  so it is drawn in a distinct colour and its label says so (`OB1881 (no
  height)`), and its unknown height uncertainty never reaches Rerun — a `NaN`
  radius is not a radius.
- Radii are a **marker size**, not a measurement: survey uncertainties are
  centimetres, which is invisible against a kilometre of walking, so the writer
  floors the radius and only lets a genuinely uncertain point grow past it.
- A camera whose detector found nothing (LaMAria runs its tag detector on the
  SLAM pair only, never on the RGB camera) gets **no** `cp_uv` entity rather than
  an empty one.
- Emitting control points did not change any existing path, so the schema version
  stays `exoego:v2` — the same additive precedent as §8 and §9.

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
5. Use `video_time` everywhere; a frame-indexed source additionally stamps
   `frame_index` (sequence) on frame-aligned rows (§14). Dataforge exposes it
   as `schema.FRAME_INDEX` (landing in the same PR stack). Native-rate sensors
   keep their own sample times (§8).
6. Every non-camera peer sensor (`/world/rig_*/imu_*`, `/world/rig_*/mag_*`) has a
   static `Transform3D` (`rig_T_imu` / `rig_T_mag`) and a static `kind`
   (`"imu"` / `"mag"`). This is a **writer-side** rule for now — dataforge's
   `logging_toolkit._log_sensor_node` is the only thing that enforces it, and no
   reader rejects a recording that breaks it — but a sensor node without its
   transform is still wrong, because a reader then cannot place its samples in
   the rig frame. A magnetometer's `field`
   is in the sensor's native units, which are only known when it carries a `unit`
   AnyValue — treat an absent `unit` as uncalibrated counts, never as tesla. The
   optional `heading` child is derived, so a reader may ignore it entirely.

7. A reader that walks the transform tree to compute `world_T_cam` stops at
   `/world`: it must not include that node's optional `root_T_world`
   transform (§2).

8. Log one static `AnnotationContext` at `/`, as all existing exoego writers
   do. Merge the hand skeleton class (§10) and all other annotation classes
   into that context, using the class IDs in §5.

The read side of these rules is `simplecv/catalog_rig_layout.py`: `parse_rig_layout`
turns a catalog schema back into typed cameras (video stream, moving rig, rig `kind`,
calibration presence, camera-node markers). Consumers add only their selection policy
on top of it instead of parsing entity paths themselves.

## 7. Dataset author checklist

`BaseExoEgoSequence.build_rig_layout` produces single-camera exo rigs;
dataforge writes multi-camera exo rigs directly.

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
  per IMU) are still TODO. Legacy RoboCap ingestion picks raw **camera** time for
  `video_time` and subtracts 14,902,432 ns from IMU timestamps. This inherited
  Basalt approximation matches the median of Cap A's four coverage-camera
  factory offsets; it is not independently validated for every camera or device.

### Optional static calibration

Log `simplecv.imu_calibration.ImuCalibration` on the IMU entity with
`static=True`. Each field is independently optional: omit unknown values;
zero represents a known zero. Existing recordings without these fields remain
valid. VIO consumers may require a complete noise model and should name missing
fields rather than substitute another device's calibration.

| Field | Meaning / units |
|---|---|
| `gyro_noise_density` | Isotropic continuous-time white noise, rad/s/√Hz |
| `accel_noise_density` | Isotropic continuous-time white noise, m/s²/√Hz |
| `gyro_bias_random_walk` | Bias random walk, rad/s²/√Hz |
| `accel_bias_random_walk` | Bias random walk, m/s³/√Hz |
| `rate_hz` | Nominal sensor sample rate; not measured stream cadence |
| `source` | Calibration source and any assumption or placeholder status |

The catalog column is, for example,
`/world/rig_00/imu_00:simplecv.ImuCalibration:gyro_noise_density`.
`ImuCalibration.from_catalog(table, imu_entity)` reads a static query into the
same typed record. The archetype describes fixed sensor parameters, not the
time-varying bias estimated by SLAM. It does not encode spatial extrinsics;
those remain the existing `Transform3D`.

### Calibration timing versus applied correction

These are separate static `AnyValues` fields:

| Entity | Fields | Meaning |
|---|---|---|
| Camera | `camera_imu_time_offset_ns`, `time_offset_reference`, `time_offset_source` | Factory estimate: `t_imu = t_camera + offset`; reference is the full IMU entity path, source identifies the calibration entry. |
| IMU | `applied_time_shift_ns`, `time_shift_source` | Signed relative alignment shift already added to the source IMU timestamps during ingestion, in ns; applies to gyro/accel children. |

The applied correction describes a relative sensor alignment adjustment, not a
shared timeline-origin change such as subtracting capture start from all sensors.
For legacy RoboCap it is **-14,902,432 ns**. Log an explicit `0` where ingestion
applied no relative correction; absence means unknown. These fields describe
existing timestamps and must not trigger a second relative correction. Kalibr's
floating-point seconds are rounded to integer nanoseconds at ingestion.

Calibration and timing provenance belong to the recording. A static-only
`sensor_metadata` layer can add them to an existing catalog segment without
rewriting its videos, samples, or poses. No existing entity paths change, so this
optional extension remains `exoego:v2`.

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

Specified here; writers land in dataforge's SHOW3D PRs (base, hand_pose, object_pose, captions layers).

## 10. Hands

UmeTrack hand annotations preserve the source's 21-landmark layout and subject
model. §5 MANO and §10 UmeTrack may coexist; a reader picks by presence,
and neither is derived from the other.

**Layout:**

```
/world/gt/hands/profile                    TextDocument (static, media type application/json)
/world/gt/hands/{left,right}
  /landmarks                              Points3D (21, world frame, metres)
  /joint_angles                           AnyValues{joint_angles} (22 radians)
  /wrist                                  Transform3D = world_T_wrist (temporal)
  /confidence                             Scalars (one value, every frame)
  /mesh                                   Mesh3D (optional derived layer)
/world/rig_NN/cam_MM/pinhole/hands/{left,right}/uv   Points2D (21, source projections)
```

- `landmarks` and `wrist` follow the sparse-pose convention below.
  `joint_angles` has rows only where the source supplies them. Wrist-frame
  landmarks are never logged as their own entity: every entity under `/world`
  is read in the world frame, so wrist-local coordinates would draw a hand at
  the rig origin. They are the skinning of `joint_angles` with the profile,
  and a consumer that needs them applies `world_T_wrist` to `landmarks`. The
  sibling `wrist` entity does not transform `landmarks`.
- For hands and objects (§11), pose rows are sparse: emit them only where
  the source has a pose. `confidence` has one row on **every frame**, with `0`
  when no pose exists. Consumers use confidence to identify gaps; they must
  not carry the last pose forward as valid.
- Use the UmeTrack 21-landmark layout, ids `0..20`, and edges
  `UME_HAND_CONNECTIONS`; simplecv's `umetrack_temp` module is the reference.
  Both hands use skeleton class id `1` (§5), with names from `LANDMARK`.
  Log static `class_ids` and `keypoint_ids` on each landmark/UV entity.
  COCO-133 mapping is a consumer concern.
- `uv` holds the dataset's shipped projections in encoded-image pixels, only
  for cameras with those annotations; out-of-view points are `NaN`. Do not
  substitute newly computed projections for shipped values.
- `profile` holds the per-subject hand model as a static `TextDocument` whose
  media type is `application/json`; the text is the verbatim source JSON. Logged
  geometry and wrist translations are metres.
- A separate derived layer MAY add `mesh`: static `triangle_indices` and
  temporal `vertex_positions` in metres, world frame, only where posed. The
  sibling `wrist` does not transform this mesh.

## 11. Objects

Tracked rigid objects have one entity per object, independent of the cameras.

**Layout:**

```
/world/gt/objects/<alias>                  Transform3D = world_T_object (temporal)
  /confidence                             Scalars (one value, every frame)
  /mesh                                   Asset3D or Mesh3D (static, optional layer)
```

- `<alias>` is the dataset's own object name. Translations and mesh coordinates
  use metres; a mesh is in the object frame and inherits `world_T_object`.
- Pose and confidence rows follow the sparse-pose and every-frame-confidence
  convention in §10.
- A separate layer MAY add a static `Asset3D` or `Mesh3D` at `mesh`; absence of a
  mesh does not remove the object's pose or confidence.

## 12. Captions/text

Recording-level captions and instructions live outside the spatial tree.

**Layout:**

```
/task/instruction                         TextDocument (static, text/markdown)
```

- Store recording-level text as a static markdown `TextDocument` at
  `/task/instruction`. Also store the instruction/caption as a segment property,
  following `rerun-io/rrd-datasets`, so catalog queries can find it without
  loading the text entity. SHOW3D uses `episode.overall_caption` for its overall
  caption; the document may include the source's structured caption fields.

## 13. Face-blur boxes

Face-blur regions describe the encoded camera image.

**Layout:**

```
/world/rig_NN/cam_MM/pinhole/blur_boxes     Boxes2D (temporal)
```

- Log a row per annotated frame in encoded-image pixel coordinates, after any
  image rotation or resize. A frame with no boxes has an empty batch, so boxes
  from a previous frame do not persist.
- Blueprints hide these entities by default. The boxes describe source blur
  regions; logging them does not apply a blur to video pixels.

## 14. Per-frame source provenance

Frame provenance preserves source identity and missing-camera information.

**Layout:**

```
/frames                                   AnyValues (one row per source frame)
  source_frame_id                          integer source frame identifier
  source_timestamp_s                      float64 source timestamp, seconds
  missing_cameras                         list of source camera names (strings)
```

- These are columns on `/frames`, not child entities. Preserve source values;
  `missing_cameras` is an empty list when every camera is present. A missing
  camera keeps its assigned `cam_MM` index; do not renumber later cameras.
- When the source has a frame index, use **two timelines** on frame-aligned
  rows: `video_time` (duration) and `frame_index` (sequence, the source index).
  SHOW3D sets `video_time = source_timestamp_s - first_source_timestamp_s` and
  keeps the original timestamp in the provenance column. The source frame ID
  and index may differ.
- Native-rate sensors retain their own `video_time` samples (§8–9); this rule
  does not resample them onto camera frames.

These additive sections retain `exoego:v2`. An incompatible change to existing
paths or meanings must increment the schema version and update this document.
