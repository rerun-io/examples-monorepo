# live-rerun rig schema

`live-rerun` logs a stationary multi-sensor **rig** (today a single OAK-D-W unit)
into Rerun using the generic COLMAP-style entity layout shared across the monorepo.
It does **not** implement SLAM; it just logs a layout a future SLAM pass can drive.

The generic rig schema is **owned by `simplecv`** — both the types
([`simplecv/rig.py`](../../simplecv/simplecv/rig.py): `SensorKind`, `CameraSensor`,
`RigCalibration`, `Rig`, `RigPoseStream`, `entity_id`, `INDEX_WIDTH`) and the
convention itself: right-to-left transform notation, the `/world/rig_<NN>/cam_<NN>`
entity tree, zero-padding (`INDEX_WIDTH=2`), RDF + metres, the static `rr.AnyValues`
metadata set (`name`/`kind` per sensor; `schema_version`/`reference`/`num_cameras`
per rig), the reference-is-identity/green-tint rule, and the reserved
IMU-as-peer / depth-under-pinhole layout. That convention is documented once in
`simplecv/rig.py` and the sibling
[`exoego_schema.md`](../../simplecv/docs/exoego_schema.md) (moving, many-cameras-as-many-rigs)
and [`slam-evals`](../../slam-evals/docs/schema.md) (the SLAM-side counterpart).
`live_rerun.rig` is a thin re-export of those types. **This doc covers only the
live-rerun / OAK specifics that live nowhere else.**

Nothing in the logging core (`rerun_video_logger.py`, `blueprint.py`, `rig.py`)
knows the word "oak". Vendor specifics live under `sources/` and in `calibration.py`,
which translate the device's calibration into the generic `RigCalibration` the core
consumes.

## Entity tree (the live-rerun OAK instantiation)

```
/world                  ViewCoordinates.RDF (static)
  /rig_00               AnyValues{schema_version="live-rerun-rig:v1", reference, num_cameras}
                        NO transform -> implicit identity (see below)
    /cam_00             left  (CAM_B, grayscale) — reference, identity rig_T_cam
      /pinhole/video    VideoStream (device_time timeline)
    /cam_01             rgb   (CAM_A, color)
      /pinhole/video
    /cam_02             right (CAM_C, grayscale)
      /pinhole/video
    /imu_<NN>           (reserved — see below)
```

The rig's `schema_version` is `"live-rerun-rig:v1"` (a loader reads it to validate
the layout; `reference` names the rig origin) — distinct from the exoego writer's
`exoego:v2`. Entity ids use `INDEX_WIDTH=2` zero-padding (`cam_00`, owned by
`simplecv/rig.py`); `slam-evals` uses single-digit `rig_0`/`cam_0`, so the two
conventions diverge on padding only.

## OAK backend mapping

| entity   | OAK socket    | `name`  | `kind`      | role               |
|----------|---------------|---------|-------------|--------------------|
| `cam_00` | CAM_B (mono)  | `left`  | `grayscale` | reference (origin) |
| `cam_01` | CAM_A (color) | `rgb`   | `rgb`       | —                  |
| `cam_02` | CAM_C (mono)  | `right` | `grayscale` | —                  |

**Reference = `cam_00` = the LEFT camera (CAM_B), not an RGB camera.** SLAM is run
around the left camera, so the whole rig moves rigidly with it; `rgb`/`right` are
expressed relative to it. This deliberately **overrides** the generic
`reference_index_for_names` default (first RGB stream) — the OAK choice is set in
`calibration.py` / `sources/depthai.py`, so changing one without the other would make
the green-tinted reference frustum and the rig origin disagree.

## Why the rig node carries no transform

The rig is stationary today, so the per-sensor `Transform3D` + `Pinhole` are logged
once as **static** and the rig node carries **no transform** (implicit identity); the
per-frame `VideoStream` is the only time-varying data. A future SLAM pass logs
`world_T_rig` **temporally** on `/world/rig_00`, moving every sensor rigidly — logging
a *static* identity there would shadow that temporal pose and trip Rerun's
"static + temporal" conflict, so we deliberately omit it.

## Reserved sensors (not logged today)

- **IMU** would land at `/world/rig_00/imu_00` (peer of the cameras: a static
  `rig_T_imu` plus `gyro`/`accel` scalars). There is **no IMU backend**: on the target
  OAK-D-W unit even a minimal IMU stream crashes the device
  (`INTERNAL_ERROR_CORE` / `IMUHalTask`, see the README). Do not enable it without
  deliberately testing that failure.
- **depth** would hang off a camera's pinhole (`…/cam_<NN>/pinhole/depth`).

## References

- Generic rig types + convention: [`simplecv/rig.py`](../../simplecv/simplecv/rig.py)
- simplecv exo/ego schema (moving, many-cameras-as-many-rigs): [`exoego_schema.md`](../../simplecv/docs/exoego_schema.md)
- slam-evals schema (the SLAM-side counterpart): [`../../slam-evals/docs/schema.md`](../../slam-evals/docs/schema.md)
- COLMAP rig/sensor model: <https://colmap.github.io/concepts.html#rigs>
