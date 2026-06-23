# live-rerun rig schema

`live-rerun` logs a multi-sensor **rig** into Rerun using a generic,
system-agnostic entity layout. The schema is deliberately the same shape as the
monorepo's SLAM-oriented [`slam-evals`](../../slam-evals/docs/schema.md) package
(COLMAP rig/sensor model) so a recording made here drops into the same mental
model and could be ingested by the same tooling later. This is the *entity
schema* only — `live-rerun` does **not** implement SLAM; it just logs a layout a
future SLAM pass can drive.

Nothing in the logging core (`rerun_video_logger.py`, `blueprint.py`,
`rig.py`) knows the word "oak". Vendor specifics live under `sources/` and in
`calibration.py`, which translate a device's calibration into the generic
[`RigCalibration`](../src/live_rerun/rig.py) the core consumes.

## Transform notation

Right-to-left composition, matching `slam-evals` and the project-wide rule:

```
cam_points     = cam_T_world @ world_points        # world  → camera
world_points   = world_T_cam @ cam_points          # camera → world
world_T_cam    = world_T_rig @ rig_T_cam           # composes along the entity tree
```

A `Transform3D` logged at `/a/b` is the parent-to-child step Rerun multiplies
when walking the entity path. `simplecv.rerun_log_utils.log_pinhole` logs the
camera extrinsic as `Transform3D(..., from_parent=True)`, so the value stored at
`/world/rig_00/cam_<NN>` is `rig_T_cam` (here the rig frame *is* the reference
camera's frame, see below).

## Entity tree

```
/world                          ViewCoordinates.RDF (static)
  /rig_00                       AnyValues{schema_version, reference, num_cameras}
                                NO transform -> implicit identity now; a future
                                SLAM pass logs world_T_rig here over the timeline
    /cam_00                     Transform3D = rig_T_cam_00 (static)
                                AnyValues{name, kind}
      /pinhole                  Pinhole / PinholeWithDistortion (static)
        /video                  VideoStream (encoded H.265/H.264 samples,
                                device_time timeline)
    /cam_01
      /pinhole/video
    /cam_02
      /pinhole/video
    /imu_<NN>                   (reserved — see "Reserved sensors")
```

Entity ids are **zero-padded to two digits** (`cam_00`, `rig_00`, `imu_00`) so
they sort lexicographically in numeric order — `cam_02` sorts before `cam_10`,
whereas a single digit would put `cam_10` first. (This is the one deliberate
divergence from `slam-evals`, which uses single-digit `rig_0`/`cam_0`.)

- The rig is stationary today, so the per-sensor `Transform3D` and `Pinhole` are
  logged once as **static**, and the rig node itself carries **no transform**
  (implicit identity). The per-frame `VideoStream` samples are the only
  time-varying data, on a shared `device_time` timeline. A future SLAM pass logs
  `world_T_rig` **temporally** on `/world/rig_00`, moving every sensor rigidly —
  logging a *static* identity there would shadow that temporal pose and trip
  Rerun's "static + temporal" conflict, so we deliberately omit it.
- All entities use metres and the OpenCV `RDF` (Right-Down-Forward) camera
  convention, logged as `ViewCoordinates.RDF` at `/world`.

## Naming conventions

- **`rig_<NN>`** — zero-padded rig root. A single device is `rig_00`; the index
  anticipates multi-rig setups (matches `slam-evals`, modulo the padding).
- **`cam_<NN>`** — zero-padded sensors, peers under the rig. The path index is
  system-agnostic; the human role (`left`/`rgb`/`right`) rides in metadata, not
  the path, so any multicam system maps onto the same tree.
- **Reference sensor = `cam_00`** — the sensor whose `rig_T_cam` is identity (the
  rig origin). Its frustum is tinted green in the viewer. For the OAK backend the
  reference is the **left** camera (CAM_B): SLAM is run around the left camera, so
  the whole rig moves rigidly with it. `rgb`/`right` are expressed relative to it.

### OAK backend mapping

| entity   | OAK socket   | `name`  | `kind`      | role               |
|----------|--------------|---------|-------------|--------------------|
| `cam_00` | CAM_B (mono) | `left`  | `grayscale` | reference (origin) |
| `cam_01` | CAM_A (color)| `rgb`   | `rgb`       | —                  |
| `cam_02` | CAM_C (mono) | `right` | `grayscale` | —                  |

## Metadata

Metadata is logged as **static `rr.AnyValues`** on the relevant entity (no
recording-property plumbing required), so it is discoverable by selecting the
entity in the viewer:

- per sensor, on `/world/rig_00/cam_<NN>`: `name` (role label) and `kind` (one of
  the reserved sensor kinds below).
- once, on `/world/rig_00`: `schema_version` (`"live-rerun-rig:v1"`),
  `reference` (e.g. `"cam_00"`), and `num_cameras`.

A loader can read `schema_version` to validate the layout and `reference` to
find the rig origin.

## Reserved sensors

`live_rerun.rig.SensorKind` reserves `depth` and `imu` alongside the emitted
`rgb`/`grayscale`. They are **not logged today**:

- **IMU** would land at `/world/rig_00/imu_00` as a peer of the cameras (a static
  `rig_T_imu` transform plus `gyro`/`accel` scalars), exactly as in
  `slam-evals`. There is **no IMU backend**: on the target OAK-D-W unit even a
  minimal IMU stream crashes the device (`INTERNAL_ERROR_CORE` / `IMUHalTask`,
  see the README). Do not enable it without deliberately testing that failure.
- **depth** would hang off a camera's pinhole (`…/cam_<NN>/pinhole/depth`).

Adding either is mechanical (a new peer entity + `kind`) and needs no change to
the existing schema.

## References

- COLMAP rig/sensor model: <https://colmap.github.io/concepts.html#rigs>
- slam-evals schema (the SLAM-side counterpart): [`../../slam-evals/docs/schema.md`](../../slam-evals/docs/schema.md)
- simplecv exo/ego schema (related, many-cams-as-many-rigs): [`../../simplecv/docs/exoego_schema.md`](../../simplecv/docs/exoego_schema.md)
