# Exo/Ego Rerun Logging Schema

This document captures the canonical entity layout we expect in Rerun recordings for combined exocentric/egocentric (exo/ego) datasets. Hocap already logs the full set of assets described here, so we treat it as the reference implementation. Any new dataset or exporter should adhere to this contract; violations should raise at ingest time rather than falling back to placeholder tensors.

All arrays follow jaxtyping notation and use metres for positions, radians for axis-angle vectors, and seconds/nanoseconds for timestamps. Transform conventions follow the project-wide rule: `cam_T_world` multiplies world-frame points on the right to produce camera-frame points (i.e. it is the world-to-camera transform; the inverse `world_T_cam` maps camera coordinates back to world space).

## 1. Recording-Level Requirements

- **Timeline:** Every entity publishes data on a shared `video_time` timeline. If a recording uses another clock name the loader fails unless `timeline_alias` is provided explicitly.
- **RGB or monochrome video:** Each camera stream publishes either `VideoStream:sample` or `AssetVideo:blob` frames. Ego rigs may contain RGB, monochrome, or mixed sensors (Hocap: RGB-only; Assembly101: monochrome-only; RRD: RGB + mono slam). As long as at least one image channel is present for the camera the schema is satisfied. Absence of imagery causes ingestion to stop unless `load_videos=False`.
- **Camera metadata:** Every calibrated capture provides a static transform and pinhole model per camera.

```
/world
  /exo/{cam_id}
    Transform3D: {cam_T_world (3x3), cam_t_world (3)}
    /pinhole
      Pinhole: {image_from_camera (3x3), camera_xyz (3), resolution (2)}
      /video
        VideoStream:sample  (preferred)  │  AssetVideo:blob  (fallback)
      /coco133_uv
        Points2DWithConfidence:positions
      /depth   (optional future channels)
    ...
  /ego/{cam_id}
    ... same structure as exo cams ...
```

- **Axis metadata:** `camera_xyz` is logged as `(Right, Down, Forward)` (RDF). Importers should assert this value and fail if a different convention is found.
- **Camera motion:** Exo cameras are expected to be static throughout the capture; their `Transform3D` components must therefore be logged as static values (either via `select_static` or a single keyframe). Ego cameras may move over time and should log dynamic transforms when motion is present. Importers should treat dynamic transforms on exo cameras as an error unless explicitly allowed.

### 1.1 Schema Profiles

To make expectations explicit we distinguish between three schema variants. Each recording should log `schema_version` (see §5) and `annotation_level` metadata so loaders can enforce the correct constraints.

- **Raw (video-only):**
  - Required: `video_time` timeline and image streams under `/pinhole/video`.
  - Optional: placeholder transforms or camera metadata. If extrinsics are unknown, omit the `Transform3D` component or set it to identity.
  - Intended for quick capture dumps. `config.load_labels` must be false; loaders should never attempt to construct `ExoEgoLabels`.

- **Calibrated (a.k.a. semi_annotated):**
  - Required: everything from Raw plus static `Transform3D` and `Pinhole` components for every exo camera (ego cameras may log dynamic transforms when needed).
  - Optional: keypoints, MANO reconstructions, depth, etc.
  - Suitable when calibration is ready but annotations are not. Importers may generate predictions in this state, but still must refuse `ExoEgoLabels`.

- **Annotated (calibrated + labeled):**
  - Required: all Calibrated assets.
  - Required when `config.load_labels` is true: `/world/gt/coco133_xyz` tensor with confidences. Missing COCO-133 data should raise immediately instead of returning NaNs.
  - Optional but recommended: additional labels such as MANO reconstructions, per-camera 2D projections, depth, segmentation, surface normals, etc.

## 2. Ground-Truth Annotations

### 2.1 MANO Surfaces and Parameters *(optional but recommended)*

```
/world/gt/mano/{hand}/mesh
  Rerun Mesh3D  (verts in metres, faces shared across frames)
/world/gt/mano/{hand}/global_orient
  Float[ndarray, "n_frames 3"]               # axis-angle root orientation
/world/gt/mano/{hand}/hand_pose
  Float[ndarray, "n_frames 15 3"]            # axis-angle joint rotations
/world/gt/mano/{hand}/betas
  Float[ndarray, "n_frames 10"] | Float[ndarray, "10"]  # shape coefficients
/world/gt/mano/{hand}/mp_21
  Float[ndarray, "n_frames 21 3"]            # MANO joint centroids
```

Notes:

- `hand ∈ {left, right}`. Data is expressed in the world frame.
- `betas` may be broadcast (static per sequence). Exporters should log frame-aligned stacks; loaders accept either the stacked or broadcast form but convert to `n_frames x 10`.
- Mesh topology is shared between hands; left-hand meshes follow the same winding order after mirroring.

### 2.2 COCO-133 Keypoints *(required for annotated recordings)*

```
/world/gt/coco133_xyz
  Points3D:positions                # Float[ndarray, "n_frames 133 3"], metres
  simplecv.KeypointConfidence3D:confidences  # Float[ndarray, "n_frames 133"]
```

- The 133-keypoint tensor is defined in the COCO Wholebody taxonomy using world coordinates.
- Datasets may add an optional component such as `/world/gt/coco133_xyz/source` (string) if provenance is relevant, but importers should not rely on it.

### 2.3 Projected 2D Keypoints *(derived)*

```
/world/{exo|ego}/{cam_id}/pinhole/coco133_uv
  Points2DWithConfidence:positions  # Float[ndarray, "n_frames 133 2"]
```

- Each camera (exo or ego) stores its own 2D projections beneath its pinhole entity so the viewer can selectively display imagery, keypoints, depth, etc. independently. Coordinates use pixels; confidence is aligned with the 3D stack. Missing points are `NaN` with confidence `0.0`.
- These arrays are deterministic projections of `/world/gt/coco133_xyz`; log them if you want to avoid recomputation at view time, or derive them on the fly.
- Any auxiliary depth map should use metres for range values; point clouds derived from depth must therefore align with the COCO-133 metric scale.

### 2.4 Wrist 6DOF Transforms *(required)*

```
/world/gt/left_wrist
/world/gt/right_wrist
  Transform3D: translation (3) + quaternion (xyzw, 4)
```

- Source: `body_poses.csv`, joints `left_hand_wrist_twist` and `right_hand_wrist_twist`
- Quaternions use xyzw convention (Rerun default)
- Logged per video frame (resampled from high-rate Quest body tracker)
- Provides 6 degrees of freedom for wrist pose tracking

## 3. Predictions *(optional)*

Predicted outputs mirror the ground-truth layout under `/world/pred/...`. This keeps GT and inference artefacts aligned and allows side-by-side visualization.

Prediction paths mirror the ground-truth layout:

- `/world/pred/mano/{hand}/...`
- `/world/pred/coco133_xyz`
- `/world/{exo|ego}/{cam_id}/pinhole/pred/coco133_uv`

- Predictions should always include confidence channels; if the model does not emit confidences, log an explicit all-ones array to make that choice obvious.

## 4. Additional Optional Channels

The schema leaves room for, but does not require:

- MANO reconstructions (`/world/gt/mano/...`), if the dataset exposes full hand meshes or parameters.
- Depth maps (`/world/{exo|ego}/{cam_id}/pinhole/depth`) and associated surface normals.
- Segmentation masks (`/world/{exo|ego}/{cam_id}/pinhole/segmentation`).

Any optional channel must declare its provenance using either an explicit `_source` string component or a dedicated `AnnotationContext`.

## 5. Validation Rules

When ingesting a recording:

1. Enumerate cameras under `/world/exo/*` and `/world/ego/*`. Fail if no RGB stream exists for any required camera.
2. Check that `Transform3D` and `Pinhole` components are present and static for every camera.
3. Resolve ground-truth tensors; raise on absence of `/world/gt/coco_133` or `/world/gt/mano/*` when `config.load_labels` is true.
4. Ensure timeline alignment by comparing per-component timestamp arrays. Mismatched lengths should error rather than silently truncating.
5. Report schema version via `/world/metadata/schema_version` (a static string such as `"exoego:v1"`). Importers can downcast or refuse older revisions.
6. Record annotation level via `/world/metadata/annotation_level ∈ {"raw","calibrated","annotated"}` to communicate which profile the recording follows.

## 6. Mapping to `ExoEgoLabels`

The loader populates `ExoEgoLabels` as follows:

```python
ExoEgoLabels(
    xyzc_stack=/world/gt/coco133_xyz,
    mano_stack=(
        ManoStack(
            betas=/world/gt/mano/{hand}/betas,
            so3=/world/gt/mano/{hand}/hand_pose,
            trans=/world/gt/mano/{hand}/mp_21[:, 0, :],
        )
        if MANO data present
        else None
    ),
)
```

Downstream consumers should rely on the dataclass rather than re-parsing Rerun logs.

## 7. Dataset Author Checklist

Before exporting:

- [ ] Log videos for each camera under the paths above.
- [ ] Write a single global timeline name (`video_time`).
- [ ] Emit MANO parameters, joints, and meshes in metres.
- [ ] Emit COCO-133 keypoints (3D + confidences) in the world frame.
- [ ] (Optional) Emit 2D projection stacks per camera.
- [ ] Record `/world/metadata/schema_version = "exoego:v1"`.
- [ ] Validate the recording using `pixi run -e dev view-exoego-data --rr-config.connect hocap` and ensure no ingestion warnings remain.

Following this schema keeps Hocap, RRD, and future datasets interoperable while leaving room for richer annotations. Any deviation should increment the schema version and update this document.
