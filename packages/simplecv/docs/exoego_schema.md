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

Any change to the layout should increment the schema version and update this doc.
