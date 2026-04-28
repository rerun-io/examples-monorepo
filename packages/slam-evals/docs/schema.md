# slam-evals catalog schema

The slam-evals package ingests VSLAM-LAB sequences into a Rerun catalog, splitting each sequence into one `.rrd` file per source data stream. The catalog server composes the files into a single segment per sequence at query/view time. This doc describes the on-disk layout, the entity tree the viewer sees, and the conventions everything follows.

## Transform notation

Use right-to-left composition throughout the codebase:

```
cam_points    = cam_T_world @ world_points        # world  → camera
world_points  = world_T_cam @ cam_points          # camera → world
world_T_sensor = world_T_rig @ rig_T_sensor      # composes along the entity tree
```

In Rerun, a `Transform3D` logged at `/a/b` represents `a_T_b` (the parent-to-child transform). Composition along an entity path is automatic: when the renderer needs `world_T_cam_0`, it walks `/world → /world/rig_0 → /world/rig_0/cam_0` and multiplies each step's `parent_T_child`. The notation makes that walk explicit and dimensionally checkable at a glance.

VSLAM-LAB's `T_BS` is body-from-sensor, i.e. `T_BS = body_T_sensor`. We rename `body` → `rig_0` (see "Naming conventions") so it becomes `rig_0_T_sensor` in the codebase.

## Source data layout (VSLAM-LAB)

Each sequence on disk is a directory under `<benchmark_root>/<dataset>/<sequence>/`:

```
<benchmark_root>/EUROC/MH_01_easy/
├── calibration.yaml      # camera intrinsics + distortion + T_BS extrinsics, IMU noise params
├── groundtruth.csv       # ts (ns), tx, ty, tz, qx, qy, qz, qw  (world_T_body)
├── rgb.csv               # timestamps (and on-disk paths) for rgb_0, optional rgb_1, optional depth_0/1
├── rgb_0/*.png           # primary camera frames
├── rgb_1/*.png           # optional: second camera (stereo modalities)
├── depth_0/*.png         # optional: 16-bit depth registered to rgb_0 (rgbd modalities)
├── depth_1/*.png         # optional: depth registered to rgb_1 (stereo-rgbd, e.g. OPENLORIS)
└── imu_0.csv             # optional: ts (ns), wx/y/z (rad/s), ax/y/z (m/s²) — *-vi modalities
```

The required triad for a valid sequence is `rgb_0/`, `rgb.csv`, `groundtruth.csv`. Modality (mono / stereo / rgbd ± `-vi`) is derived from which optional files are present.

## RRD layer layout

The catalog stores each sequence as **one segment with multiple layers**, where each layer is its own `.rrd` file:

```
data/slam-evals/rrd/EUROC/MH_01_easy/
├── calibration.rrd       # layer="calibration"
├── groundtruth.rrd       # layer="groundtruth"
├── view_coordinates.rrd  # layer="view_coordinates" (per-dataset world axes)
├── video_0.rrd           # layer="video_0"
├── video_1.rrd           # layer="video_1" (stereo only)
├── depth_0.rrd           # layer="depth_0" (rgbd only)
├── depth_1.rrd           # layer="depth_1" (stereo-rgbd only)
└── imu_0.rrd             # layer="imu_0"   (-vi only)
```

All layer files for a given sequence share `recording_id = f"{dataset}__{sequence}"` and `application_id = "slam-evals"`. The catalog uses these to know they belong to the same segment.

Layer count by modality (counts include `view_coordinates.rrd` when the
dataset has a `DatasetSpec` registered in `slam_evals.data.datasets`;
sequences in unregistered datasets ship without that layer and the
viewer falls back to its default world frame):

| Modality       | Layers (with view_coordinates)                                                          | Count |
|----------------|-----------------------------------------------------------------------------------------|-------|
| mono           | calibration, groundtruth, view_coordinates, video_0                                     | 4     |
| mono-vi        | + imu_0                                                                                 | 5     |
| stereo         | + video_1                                                                               | 5     |
| stereo-vi      | + video_1, imu_0                                                                        | 6     |
| rgbd           | + depth_0                                                                               | 5     |
| rgbd-vi        | + depth_0, imu_0                                                                        | 6     |
| stereo-rgbd-vi | calibration, groundtruth, view_coordinates, video_0, video_1, depth_0, depth_1, imu_0   | 8     |

## How the catalog works (mental model)

The catalog is **an index over `.rrd` files**, not a separate store. Three steps:

**1. Ingest** writes recording properties INTO each `.rrd` via `send_property()`:
```python
rec = rr.RecordingStream(application_id="slam-evals", recording_id=rec_id, send_properties=True)
rec.send_property("video_0", rr.AnyValues(codec="hevc", fps=30.0, num_frames=487, width=640, height=480))
# … log video stream chunks at /world/rig_0/cam_0/pinhole/video …
rec.save("video_0.rrd")
```

**2. Catalog mount** registers each layer file under the right layer name:
```python
server = rr.server.Server(datasets={"vslam": []})
dataset = server.client().get_dataset("vslam")
dataset.register([uri], layer_name="video_0").wait()
```
Files sharing a `recording_id` collapse into one segment automatically.

> **Why one `vslam` dataset and not per-source-benchmark datasets** (`euroc`, `kitti`, …): the source benchmark is already a column (`property:info:dataset`) so per-dataset filtering is one expression, and cross-benchmark analytics — the killer feature once baselines + ATE land — stays a single `segment_table()` call instead of an N-way join across catalog datasets. Splitting is a mechanical refactor (loop over dataset names at mount, one extra path level on disk) we can do the day per-dataset blueprints become a real need.

**3. Query** aggregates properties across all layers in the segment:
```python
df = dataset.segment_table().to_pandas()
# columns include property:info:modality, property:video_0:codec,
# property:depth_0:depth_factor, property:imu_0:rate_hz, …
```

Two consequences:

- **Properties are immutable** once written into an `.rrd`. Regenerate the layer to change one.
- **The catalog is by-reference**. Delete a layer file, and on next mount its data + properties disappear from the catalog.

## Entity tree (COLMAP-aligned)

The entity tree follows COLMAP's rig/sensor model (https://colmap.github.io/concepts.html):

- **`/world/rig_<i>` is the rig** — the rigid platform whose pose moves over time. The rig pose IS the ground-truth trajectory: `world_T_rig_<i>` over `video_time`, logged from `groundtruth.rrd`.
- **Sensors are peer children of the rig**, each carrying a static `Transform3D` for `rig_<i>_T_sensor` (a.k.a. T_BS in VSLAM-LAB terms). Cameras and IMU sit at the same level — IMU is **not** nested under a camera.
- **Reference sensor** (per COLMAP, "the sensor that defines the rig origin") has `rig_T_sensor = identity`. For EUROC-style data this is the IMU; for KITTI-style it's typically `cam_0`. We log everyone's `T_BS` verbatim from `calibration.yaml` regardless.
- **Sensor-specific data** (image stream, depth stream, gyro/accel scalars) hangs off the sensor entity.
- A **frame** in COLMAP is a synchronized measurement set at a timestamp. In Rerun this is implicit via the shared `video_time` timeline; partial frames (some sensor missing a sample at time *t*) are handled by per-entity chunking.

Composition along the entity path: `world_T_cam_0 = world_T_rig_0 @ rig_0_T_cam_0` — Rerun does this automatically when rendering.

### Composed entity tree (what the viewer sees per segment)

After the catalog stitches the layers together, a stereo-rgbd-vi segment looks like:

```
/world/
  rig_0/                                 # world_T_rig_0 over time = GT (groundtruth.rrd)
    cam_0/                               # rig_0_T_cam_0 (static, calibration.rrd)
      pinhole/                           # Pinhole + distortion (static, calibration.rrd)
        video                            # HEVC VideoStream (video_0.rrd)
        depth                            # EncodedDepthImage stream (depth_0.rrd)
    cam_1/                               # rig_0_T_cam_1 (static, calibration.rrd)
      pinhole/
        video                            # HEVC VideoStream (video_1.rrd)
        depth                            # EncodedDepthImage stream (depth_1.rrd)
    imu_0/                               # rig_0_T_imu_0 (static, calibration.rrd)
      gyro                               # 3-component Scalars (imu_0.rrd)
      accel                              # 3-component Scalars (imu_0.rrd)
```

## Layer-by-layer entity paths and properties

| Layer file              | Layer name          | Entity paths logged                                                                              | Layer-level recording properties                                                                                       |
|-------------------------|---------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `calibration.rrd`       | `calibration`       | static `Transform3D` at `/world/rig_0/cam_<i>` and `/world/rig_0/imu_<i>` (each `rig_0_T_sensor`); static `Pinhole/PinholeWithDistortion` at `/world/rig_0/cam_<i>/pinhole` | `info.{modality, dataset, sequence, slug, has_calibration}`; `calibration.{num_cameras, cam0_*, depth_factor, has_imu_params}` |
| `groundtruth.rrd`       | `groundtruth`       | time-varying `Transform3D` (`world_T_rig_0`) at `/world/rig_0` over `video_time`; static `LineStrips3D` GT path at `/world/rig_0_path`; static start/end `Points3D` markers at `/world/rig_0_path/endpoints` | `groundtruth.{num_poses, trajectory_len_m, duration_s, has_rotation}` |
| `view_coordinates.rrd`  | `view_coordinates`  | static `ViewCoordinates` at `/world` (per-dataset axis convention from `slam_evals.data.datasets`) | none                                                                                                                   |
| `video_0.rrd`           | `video_0`           | static `VideoStream(codec=H265)` at `/world/rig_0/cam_0/pinhole/video`; per-packet logs over `video_time` | `video_0.{codec, fps, num_frames, width, height}` |
| `video_1.rrd`           | `video_1`           | same shape at `/world/rig_0/cam_1/pinhole/video`                                                  | `video_1.{codec, fps, num_frames, width, height}` |
| `depth_0.rrd`     | `depth_0`     | per-frame `EncodedDepthImage` (PNG passthrough) at `/world/rig_0/cam_0/pinhole/depth` over `video_time` | `depth_0.{depth_factor, num_frames, width, height}`                                                                    |
| `depth_1.rrd`     | `depth_1`     | same at `/world/rig_0/cam_1/pinhole/depth`                                                        | `depth_1.{depth_factor, num_frames, width, height}`                                                                    |
| `imu_0.rrd`       | `imu_0`       | 3-component `Scalars` at `/world/rig_0/imu_0/gyro` and `/world/rig_0/imu_0/accel` over `video_time` | `imu_0.{num_samples, rate_hz}` (+ noise terms when present in calibration.yaml)                                        |

### Worked `segment_table()` example

```
rerun_segment_id          rerun_layer_names                                      property:info:modality   property:info:slug          property:video_0:codec   property:video_0:fps   property:depth_0:depth_factor   property:imu_0:rate_hz
EUROC__MH_01_easy         [calibration, groundtruth, video_0, video_1, imu_0]    stereo-vi                EUROC/MH_01_easy            hevc                     20.0                   NULL                            200.0
ETH__cables_1             [calibration, groundtruth, video_0, depth_0]           rgbd                     ETH/cables_1                hevc                     30.0                   5000.0                          NULL
TUM_RGBD__freiburg1_xyz   [calibration, groundtruth, video_0, depth_0]           rgbd                     TUM_RGBD/freiburg1_xyz      hevc                     30.0                   5000.0                          NULL
```

Filter at the pandas layer: `df.query("`property:info:modality`.str.startswith('stereo')")`.

## Naming conventions

- `rig_<i>` zero-indexed. Anticipates multi-rig setups (exoego, multi-agent). VSLAM-LAB has a single rig per sequence today (`rig_0`).
- `cam_<i>`, `depth_<i>`, `imu_<i>` per source stream (entity-tree level). Index matches VSLAM-LAB's source file naming (`rgb_0`, `imu_0`, …). Layer files on the catalog side are payload-typed instead — `video_<i>.rrd` carries the image stream that hangs off `cam_<i>` (so the layer name doesn't pretend grayscale data is RGB; see `slam_evals/ingest/layer_video.py`).
- Semantic role (ego / exo / fixed / etc.) lives as a property on `calibration.rrd`, **not** in the entity path.
- Contrast with simplecv's `exoego_schema.md`: that schema is optimised for many-cameras-as-many-rigs (each camera under `/world/exo` or `/world/ego` is essentially its own one-sensor rig). slam-evals is the inverse — one rig with many tightly-calibrated sensors moving together.

## GT vs predictions (forward-looking; not implemented in this slice)

GT and predictions are **not symmetric** and live at different entity paths:

- **GT is constitutive.** The dataset *defines* where the rig was. It's the only sensible parent for the sensor tree (cameras must visualize at *some* canonical pose). GT lives at `/world/rig_<i>` as `world_T_rig_<i>` over time, written by `groundtruth.rrd`.
- **Predictions are observations.** Algorithms guess where the rig was; multiple algorithms can disagree; predictions never replace the rig's pose. Each prediction lives at `/world/runs/<source>/trajectory`.

Layout for a future eval pass:

```
/world/
  rig_0/                                 # GT (constitutive)
    cam_0/...
    imu_0/...
  runs/
    pycuvslam/trajectory                 # Transform3D / LineStrips3D over video_time
    orbslam3/trajectory
    groundtruth_colmap/trajectory        # alternative GT sources also go here
```

This falls out cleanly from the layer model: a `pycuvslam.rrd` layer with the source's `recording_id` joins the same segment via the catalog. `segment_table()` aggregates `property:pycuvslam:ate_rmse`, etc. — leaderboard for free.

If you ever want to *visualize* a predicted rig with camera frustums (not just a trajectory line), re-export the static calibration tree under the prediction's transform: `/world/runs/pycuvslam/rig_0/cam_<i>/pinhole/...`. Heavier but useful for failure analysis.

## Future modalities

The per-stream layer schema makes adding a new sensor mechanical — new sibling under `/world/rig_<i>/`, new layer file:

- `/world/rig_0/lidar_0` + `lidar_0.rrd` (point clouds, scan time)
- `/world/rig_0/gps` + `gps.rrd` (lat/lon over time)
- `/world/rig_0/event_0` + `event_0.rrd` (event camera tensors)
- `/world/rig_0/mocap_0` + `mocap_0.rrd` (alternative high-rate pose source)

New derived layers (not source data) sit alongside algorithm output under `/world/runs/`:

- `/world/runs/depth_anything/cam_0/depth` + `depth_anything_cam_0.rrd` (predicted depth from monoprior)
- `/world/runs/superpoint/cam_0/keypoints` + `superpoint_cam_0.rrd`

Adding any of these requires no changes to existing layers, blueprint, or catalog code — they're new files with new layer names sharing the segment's recording_id.

## References

- COLMAP concepts: https://colmap.github.io/concepts.html
- Rerun recordings: https://rerun.io/docs/concepts/logging-and-ingestion/recordings
- Rerun layers: https://rerun.io/docs/howto/logging-and-ingestion/layers
- simplecv exoego schema: https://github.com/pablovela5620/simplecv/blob/main/docs/exoego_schema.md (related, distinct use case)
