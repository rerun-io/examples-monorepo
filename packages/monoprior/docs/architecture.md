# Monoprior Architecture

Monoprior is a library for monocular geometric priors: depth estimation (metric and relative), surface normal estimation, and multi-view geometry recovery. It wraps several state-of-the-art models behind a unified predictor API, logs results to Rerun for 3D visualization, and exposes both CLI tools and a Gradio web UI.

## Model families

### Metric depth (absolute scale in meters)

Predict depth maps with real-world scale. Defined in `monopriors/models/metric_depth/`.

| Predictor class | Model | Source |
|---|---|---|
| `Metric3DPredictor` | Metric3D V2 | [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D) |
| `UniDepthMetricPredictor` | UniDepth | [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth) |

Base class: `BaseMetricPredictor`
Output: `MetricDepthPrediction(depth_meters, confidence, K_33)`

### Relative depth (scale-invariant)

Predict depth/disparity without absolute scale. Defined in `monopriors/models/relative_depth/`.

| Predictor class | Model | Source |
|---|---|---|
| `DepthAnythingV1Predictor` | Depth Anything V1 | [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything) |
| `DepthAnythingV2Predictor` | Depth Anything V2 | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| `Metric3DRelativePredictor` | Metric3D (relative mode) | [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D) |
| `MogeV1Predictor` | MoGe V1 | [microsoft/MoGe](https://github.com/microsoft/MoGe) |
| `UniDepthRelativePredictor` | UniDepth (relative mode) | [lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth) |

There is also a `BaseVideoRelativePredictor` for temporal video depth (`VideoDepthAnythingPredictor`), which takes `UInt8[ndarray, "T H W 3"]` frame sequences and returns per-frame predictions with temporal consistency.

Base class: `BaseRelativePredictor`
Output: `RelativeDepthPrediction(disparity, depth, confidence, K_33)`

### Surface normals

Predict per-pixel surface normal direction. Defined in `monopriors/models/surface_normal/`.

| Predictor class | Model | Source |
|---|---|---|
| `DSineNormalPredictor` | DSINE | [baegmon/DSINE](https://github.com/baegmon/DSINE) |
| `StableNormalPredictor` | Stable Normal Maps | [Stable-X/StableNormal](https://github.com/Stable-X/StableNormal) |
| `OmniNormalPredictor` | OmniData | [EPFL-VILAB/omnidata](https://github.com/EPFL-VILAB/omnidata) |

Base class: `BaseNormalPredictor`
Output: `SurfaceNormalPrediction(normal_hw3, confidence_hw1)`

### Multi-view geometry (VGGT)

Recovers camera poses and dense depth from unposed image collections. Defined in `monopriors/models/multiview/vggt_model.py`.

| Predictor class | Model | Source |
|---|---|---|
| `VGGTPredictor` | VGGT | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |

Output: `VGGTPredictions` (pyserde dataclass) containing per-camera depth, confidence, intrinsics, extrinsics (`cam_T_world`), and world-space points.

### Depth completion

Refines sparse or noisy depth using an RGB guide. Defined in `monopriors/models/depth_completion/`.

| Predictor class | Model | Source |
|---|---|---|
| `PromptDAPredictor` | PromptDA | [PromptDA](https://github.com/AnyQuantAI/PromptDA) |

Base class: `BaseCompletionPredictor`
Output: `CompletionDepthPrediction(depth_mm, confidence)`

### Composite models

`monopriors/monoprior_models.py` combines depth + normals into a single call:

```
MonoPriorModel (abstract)
 └── DsineAndUnidepth
      ├── metric depth  → Metric3DPredictor
      └── surface normals → DSineNormalPredictor
```

Output: `MonoPriorPrediction(metric_pred, normal_pred)`

## Predictor API pattern

Every model family follows the same factory + callable pattern:

```python
from monopriors.models.relative_depth import get_relative_predictor

predictor = get_relative_predictor("DepthAnythingV2Predictor")(device="cuda")
result: RelativeDepthPrediction = predictor(rgb=rgb_hw3, K_33=None)
```

Factory functions (`get_relative_predictor`, `get_metric_predictor`, `get_normal_predictor`) use `Literal` types for predictor names and `match` statements for dispatch. All predictors have a `set_model_device()` method for moving weights between CPU/GPU.

## Core data types

All arrays use jaxtyping annotations for dtype and shape.

```
MetricDepthPrediction
  depth_meters : Float[ndarray, "h w"]     # absolute depth in meters
  confidence   : Float[ndarray, "h w"]
  K_33         : Float[ndarray, "3 3"]     # camera intrinsics

RelativeDepthPrediction
  disparity    : Float32[ndarray, "h w"]   # inverse depth (scale-free)
  depth        : Float32[ndarray, "h w"]   # 1/disparity (scale-free)
  confidence   : Float32[ndarray, "h w"]
  K_33         : Float32[ndarray, "3 3"]

SurfaceNormalPrediction
  normal_hw3      : Float[ndarray, "h w 3"]
  confidence_hw1  : Float[ndarray, "h w 1"]

VGGTPredictions  (@serde dataclass)
  pose_enc         : UInt8[ndarray, "*batch num_cams 9"]
  depth            : Float32[ndarray, "*batch num_cams H W 1"]
  depth_conf       : Float32[ndarray, "*batch num_cams H W"]
  world_points     : Float32[ndarray, "*batch num_cams H W 3"]
  intrinsic        : Float32[ndarray, "*batch num_cams 3 3"]
  cam_T_world_b34  : Float32[ndarray, "*batch num_cams 3 4"]
```

Camera parameters use `simplecv.camera_parameters.PinholeParameters` which bundles `Intrinsics` (K matrix + resolution) and `Extrinsics` (both `cam_T_world` and `world_T_cam` directions).

## Module map

```
monopriors/
  __init__.py                     # beartype_this_package() activation
  monoprior_models.py             # Composite model (depth + normals)
  depth_utils.py                  # depth<->disparity, depth->points, edge masks, intrinsics estimation
  scale_utils.py                  # scale-shift alignment (relative->metric)
  camera_utils.py                 # torch-based pose utilities
  camera_numpy_utils.py           # numpy PCA-based auto_orient_and_center_poses
  rr_logging_utils.py             # Rerun logging helpers for depth/normal predictions
  dc_utils.py                     # Frame reading and video I/O

  metric_depth_models/            # Metric depth predictors + factory
  relative_depth_models/          # Relative depth predictors + factory
  surface_normal_models/          # Normal predictors + factory
  multiview_models/               # VGGT multi-view geometry
  depth_completion_models/        # PromptDA depth completion

  apis/                           # High-level inference pipelines (tyro configs)
    relative_depth_inference.py   # Single-image relative depth → rerun
    multiview_inference.py        # Multi-view depth + pose recovery → rerun + COLMAP export
    multiview_calibration.py      # Multi-view calibration with SAM2 + pose optimization
    polycam_inference.py          # Polycam dataset processing
    vda_inference.py              # Video Depth Anything temporal inference
    promptda_polycam.py           # PromptDA on Polycam data

  gradio_ui/                      # Gradio web UI components
    depth_compare_ui.py           # Side-by-side model comparison
    depth_inference_ui.py         # Single-model depth inference
    multiview_calibration_ui.py   # Multi-view calibration UI

  data/                           # Dataset loaders
    nerfstudio_data.py            # Nerfstudio transforms.json format
    sdfstudio_data.py             # SDFStudio meta_data.json format

  third_party/                    # Vendored model implementations
    depth_anything_v2/            # DepthAnything V2 model code
    dsine/                        # DSINE surface normal code
    promptda/                     # PromptDA + DINOv2 backbone
    video_depth_anything/         # Video DepthAnything temporal module
```

## CLI tools

Entry points are organized under `tools/` in three subdirectories. All are exposed as pixi tasks (run with `pixi run -e monoprior <task>`).

### tools/demos/ — CLI demo scripts

| Script | Pixi task | What it does |
|---|---|---|
| `relative_depth.py` | `relative-depth` | Single-image relative depth + rerun viewer |
| `multiview_depth.py` | `multiview-depth` | VGGT multi-view inference, exports COLMAP format |
| `multiview_calibration.py` | `multiview-calibration` | Multi-view refinement with SAM2 |
| `video_depth.py` | `video-depth` | Video Depth Anything temporal inference |
| `polycam_inference.py` | `polycam-inference` | Process Polycam dataset with DsineAndUnidepth |
| `promptda_polycam.py` | `promptda-polycam` | PromptDA depth completion on Polycam data |
| `compare_normals.py` | `compare-normals` | Compare surface normal models |

### tools/apps/ — Gradio web UIs

| Script | Pixi task | What it does |
|---|---|---|
| `depth_compare_app.py` | `depth-compare-app` | Depth comparison UI with Rerun viewer |
| `video_depth_app.py` | `video-depth-app` | Video depth estimation UI |
| `calibration_app.py` | `calibration-app` | Multi-view calibration UI |

### tools/utils/ — Packaging, conversion, deployment

| Script | Pixi task | What it does |
|---|---|---|
| `hf_spaces_launcher.py` | (startup helper) | Checks environment, kills port 7860, launches Gradio |
| `upload_to_hf.py` | `upload-hf` | Upload wheel to HuggingFace |
| `nerfstudio_to_sdfstudio.py` | (manual) | Convert Nerfstudio format to SDFStudio |
| `view_sdfstudio.py` | (manual) | View SDFStudio data |

## Inference pipelines

### Single-image depth

```
RGB image (h, w, 3)
  → [optional] estimate intrinsics from FOV
  → predictor(rgb, K_33)
  → MetricDepthPrediction | RelativeDepthPrediction
  → depth_to_points() → 3D point cloud
  → depth_edges_mask() → filter flying pixels
  → log to Rerun (pinhole + image + depth + points)
```

### Multi-view (VGGT)

```
Image set [N images]
  → preprocess (crop/pad to 518px width, divisible by 14)
  → VGGT model → VGGTPredictions
      ├── pose_encoding_to_extri_intri → camera poses
      ├── depth maps per view
      └── confidence masks
  → auto_orient_and_center_poses (PCA)
  → multidepth_to_points → combined point cloud
  → [optional] MoGe refinement via scale-shift alignment
  → voxel downsampling (binary search for target count)
  → log to Rerun (3D spatial view + per-camera 2D tabs)
  → [optional] export COLMAP-format poses
```

### Depth comparison (Gradio)

```
Upload image → select 2 models (metric or relative)
  → run both predictors
  → log side-by-side to Rerun with compare blueprint
  → stream binary RRD to Gradio Rerun viewer
```

## Rerun integration

`rr_logging_utils.py` provides:
- `log_relative_pred()` / `log_metric_pred()` — log camera transform, pinhole, RGB, depth, confidence, and back-projected point cloud
- `create_compare_depth_blueprint()` — side-by-side 3D + 2D views for model comparison

Gradio callbacks use `@rr.thread_local_stream()` + `rr.binary_stream()` to stream incremental RRD bytes to the `gradio-rerun` viewer component.

Multi-view inference creates tabbed blueprints grouping cameras 4 per tab, with depth/filtered-depth/MoGe-depth sub-tabs per camera.

## Utility modules

| Module | Key functions |
|---|---|
| `depth_utils` | `estimate_intrinsics`, `depth_to_points`, `multidepth_to_points`, `depth_edges_mask`, `depth_to_disparity`, `disparity_to_depth`, `clip_disparity` |
| `scale_utils` | `compute_scale_and_shift` (least-squares alignment of relative depth to metric), `compute_scale`, `get_interpolate_frames` |
| `camera_utils` | `rotation_matrix_between`, `focus_of_attention` (torch) |
| `camera_numpy_utils` | `auto_orient_and_center_poses` (PCA-based orientation, numpy) |

## Runtime type checking

`__init__.py` unconditionally calls `beartype_this_package()`, enabling runtime shape/dtype enforcement on every jaxtyping-annotated function and dataclass in the package. This activates automatically when the package is imported.
