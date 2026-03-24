# Decompose MultiViewCalibrator into Daggr-Composable Nodes

## Context

The `MultiViewCalibrator` in `multiview_calibration.py` is monolithic — it bundles VGGT (geometry), SAM3 (segmentation), and MoGe (depth refinement) into a single class. The goal is to decompose it into self-contained nodes that each have their own CLI + Gradio app, following the pattern established by wilor-nano. These nodes can then be composed via daggr into a unified workflow, while the existing monolithic Gradio UI continues to work for the live Rerun streaming experience.

### Two types of apps
- **Self-contained nodes**: One model, clear input/output contract, daggr-composable (VGGT, SAM3, MoGe, Scale Alignment)
- **Wider-range apps**: Composition/analysis tools that combine multiple models (multiview calibration UI, depth compare). These are NOT daggr nodes — they are end-user experiences built on top of the same API functions.

### Daggr reality check
- `graph.launch()` is a custom Svelte canvas, NOT a standard `gr.Blocks` app
- No `.invoke()` yet (issue #35), no streaming within nodes, no `gradio_rerun` in the canvas
- Intermediate data passes as file paths (JSON/files), not Python objects
- BUT: each node IS a standalone Gradio app, so you get two composition modes:
  - **Daggr canvas** — visual DAG for debugging, step-by-step execution
  - **Composed Gradio app** — `.then()` chain with Rerun streaming for production UX

### Decisions
- SAM3 multi-view node lives in **sam3-rerun** package (alongside existing single-image apps)
- Monolithic `MultiViewCalibrator` **kept as thin orchestrator** (delegates to decomposed APIs)
- Scale alignment is its **own node** — reusable across any depth estimation method pair
- All node I/O uses **typed dataclasses** (with jaxtyping annotations), not `.npz` files. Serialization for daggr is handled at the Gradio `pred_fn` boundary via `pyserde`.

## Pipeline DAG (New)

```
                ┌──→ VGGT ──────────┐
                │                    │
shared_images ──┼──→ SAM3 ──────────┼──→ Scale Alignment ──→ Fusion
                │                    │
                └──→ MoGe ──────────┘

All 3 model nodes run in parallel. Scale alignment waits for all 3. Fusion waits for alignment.
```

## Node Contracts (Dataclasses)

Each node's API matches the **network's actual I/O contract**. Single-image networks (SAM3, MoGe) take a single image. Multi-view networks (VGGT) take a list. Batching over views is the caller's responsibility (the Gradio `pred_fn` or orchestrator loops).

### Node 1: VGGT Geometry (GPU)

The network **genuinely takes a list** — VGGT processes all views jointly to produce consistent multi-view geometry.

An existing `VGGTInferenceConfig` already lives in `apis/multiview_inference.py` (line 436) with `keep_top_percent`, `preprocessing_mode`, `image_dir`, `videos_dir`, `rr_config`. There's also `MultiViewCalibratorConfig` in `apis/multiview_calibration.py` with overlapping fields. Rather than creating yet another config, refactor: extract the VGGT-specific fields into a standalone config that both inference APIs import.

```python
# Refactored from VGGTInferenceConfig — extract the network-specific fields
@dataclass
class VGGTGeometryConfig:
    """Configuration for VGGT multi-view geometry prediction."""
    preprocessing_mode: Literal["crop", "pad"] = "pad"
    """Image preprocessing strategy."""
    keep_top_percent: KeepTopPercent = 30.0
    """Fraction of high-confidence pixels retained after filtering."""
    device: Literal["cuda", "cpu"] = "cuda"
    verbose: bool = False

@dataclass
class VGGTGeometryResult:
    """Output of VGGT multi-view geometry prediction."""
    mv_pred_list: list[MultiviewPred]
    """Oriented multi-view predictions (poses, depths, confidences, intrinsics)."""
    depth_confidences: list[UInt8[ndarray, "H W"]]
    """Binary confidence masks after robust filtering."""
```

Then `VGGTInferenceConfig` and `MultiViewCalibratorConfig` can compose this config rather than duplicating the fields.

- **Input**: `list[UInt8[ndarray, "H W 3"]]`
- **Contains**: `VGGTPredictor` + `orient_mv_pred_list()` + `robust_filter_confidences()`
- **Port**: 7870

### Node 2: SAM3 Segmentation (GPU, in `sam3-rerun` package)

The network takes a **single image** — the existing `SAM3Predictor.predict_single_image()` API is already pure. The `segment_people()` helper in `multiview_calibration.py` wraps it with union-mask + dilation logic. The Gradio `pred_fn` and orchestrator loop over views as needed.

```python
# Already exists — no new config needed, just expose the existing interface:
# SAM3Config(device, sam3_checkpoint)
# SAM3Predictor.predict_single_image(rgb_hw3, text) -> SAM3Results

# The segment_people() helper becomes the node's pipeline function:
def segment_people(
    rgb: UInt8[ndarray, "H W 3"],
    *,
    seg_predictor: SAM3Predictor,
    text: str = "person",
    mask_threshold: float = 0.5,
    dilation: int = 0,
) -> Bool[ndarray, "H W"] | None
```

- **Input**: single `UInt8[ndarray, "H W 3"]` + text prompt
- **Output**: single `Bool[ndarray, "H W"] | None` (union mask or None)
- **Port**: 7871
- **Existing app**: `sam3_rerun_ui.py` already has the right layout and `pred_fn`. Update it with Config accordion + `_sync_config()` to match the node spec. The `segment_people()` wrapper in `multiview_calibration.py` stays as orchestrator-level glue (loops over views + dilation).

### Node 3: Metric Depth (GPU, any `BaseMetricPredictor`)

The network takes a **single image** + optional K_33. Uses the existing factory pattern: `get_metric_predictor(name)(device)` returns any `BaseMetricPredictor`.

```python
@dataclass
class MetricDepthNodeConfig:
    """Configuration for metric depth estimation — works with any metric predictor."""
    predictor_name: METRIC_PREDICTORS = "MoGeV2MetricPredictor"
    """Which metric depth predictor to use (MoGeV2MetricPredictor, UniDepthMetricPredictor, etc.)."""
    device: Literal["cuda", "cpu"] = "cuda"

# Already exists — no new result dataclass needed:
# BaseMetricPredictor.__call__(rgb, K_33=None) -> MetricDepthPrediction
# MetricDepthPrediction { depth_meters, confidence, K_33 }
```

- **Input**: single `UInt8[ndarray, "H W 3"]` + optional `Float[ndarray, "3 3"]`
- **Output**: single `MetricDepthPrediction` (depth_meters, confidence, K_33)
- **Runs independently**: estimates its own K_33 from the model
- **Uses**: `get_metric_predictor()` factory (from `models/metric_depth/__init__.py`)
- **Available predictors**: `MoGeV2MetricPredictor`, `UniDepthMetricPredictor`
- **Port**: 7872

### Node 4: Depth Alignment (CPU, general-purpose)

Pure utility: aligns one depth map to another's coordinate frame. **No knowledge of VGGT, MoGe, or any specific network.** The caller decides which depth is "reference" and which is "target".

```python
@dataclass
class DepthAlignmentConfig:
    """Configuration for aligning a target depth map to a reference depth's coordinate frame."""
    edge_threshold: float = 0.01
    """Threshold for depth edge masking on the aligned output."""
    scale_only: bool = False
    """Use scale-only alignment (no shift) when True."""

@dataclass
class DepthAlignmentResult:
    """Output of single-view depth alignment."""
    aligned_depth: Float32[ndarray, "H W"]
    """Target depth aligned to the reference depth's coordinate frame."""

def run_depth_alignment(
    *,
    reference_depth: Float32[ndarray, "H W"],
    target_depth: Float32[ndarray, "H W"],
    confidence_mask: Bool[ndarray, "H W"] | None = None,
    exclusion_mask: Bool[ndarray, "H W"] | None = None,
    config: DepthAlignmentConfig = DepthAlignmentConfig(),
) -> DepthAlignmentResult:
    """Align target_depth to reference_depth's scale/shift using valid pixels.

    Args:
        reference_depth: Depth map defining the target coordinate frame.
        target_depth: Depth map to be aligned.
        confidence_mask: Binary mask of trusted pixels in reference_depth.
        exclusion_mask: Binary mask of pixels to exclude (e.g., people).
        config: Alignment configuration.
    """
```

- **Input**: two depth maps (any source) + optional confidence and exclusion masks
- **Output**: single aligned depth map
- **Contains**: `compute_scale_and_shift()`, `depth_edges_mask()`, mask logic
- **Key reusability**: works with ANY depth pair from ANY network
- **Port**: 7873 (or `FnNode` — no GPU needed)
- **Note**: The orchestrator/Gradio app loops this over views. Each call aligns one view.

### Node 5: Fusion (CPU, `FnNode`)

```python
@dataclass
class FusionConfig:
    """Configuration for point cloud and mesh fusion."""
    grid_resolution: int = 512
    """TSDF grid resolution."""
    target_points: int = 150_000
    """Target point count after voxel downsampling."""

@dataclass
class FusionResult:
    """Output of depth fusion: point cloud and mesh."""
    pcd: o3d.geometry.PointCloud
    """Fused and downsampled point cloud."""
    mesh: o3d.geometry.TriangleMesh | None
    """TSDF-fused mesh, or None if fusion fails."""

def run_fusion(
    *,
    depth_list: list[Float32[ndarray, "H W"]],
    pinhole_param_list: list[PinholeParameters],
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    config: FusionConfig = FusionConfig(),
) -> FusionResult:
```

- **Input**: depth maps + pinhole params + RGBs (generic — no coupling to upstream node types)
- **Contains**: `mv_pred_to_pointcloud()`, voxel downsampling, `Open3DScaleInvariantFuser`
- **No GPU model** — pure geometry

## Gradio App UI Designs

Each node follows the established layout pattern: **left column** (scale=1) with Tabs(Inputs/Outputs) + examples, **right column** (scale=5) with Rerun viewer. Click chain: `click → _switch_to_outputs → recording_id → _sync_config → [parse inputs] → pred_fn`.

**Key principles**:
- Every Config dataclass field gets a corresponding widget in the Config accordion
- `_sync_config()` reads widget values into the config singleton (same pattern as calibration UI)
- Each node logs to Rerun under `PARENT_LOG_PATH = Path("world")` with consistent sub-paths
- Nodes that process multiple views use per-camera paths: `world/camera_{i}/pinhole/...`

### VGGT Geometry App

**Config accordion** (maps 1:1 to `VGGTGeometryConfig` fields):
- `preprocessing_mode` → `gr.Radio(choices=["crop", "pad"], value="pad")`
- `keep_top_percent` → `gr.Slider(minimum=1, maximum=100, step=1, value=30)`
- `verbose` → `gr.Checkbox(value=True)` (controls per-camera detail logging)

**Rerun log paths**:
```
world/                              ViewCoordinates.RFU
world/camera_{i}/                   camera transform (from log_pinhole)
world/camera_{i}/pinhole/           PinholeProjection
world/camera_{i}/pinhole/image      rr.Image (RGB)
world/camera_{i}/pinhole/depth      rr.DepthImage (VGGT raw depth)
world/camera_{i}/pinhole/filtered_depth   rr.DepthImage (confidence-filtered)
world/camera_{i}/pinhole/confidence rr.Image (confidence mask, grayscale)
world/point_cloud                   rr.Points3D (from unfiltered depths)
```

**Blueprint**: reuse `create_final_view(parent_log_path, num_images)` — 3D view left, per-camera depth/confidence tabs right. No TSDF mesh view (that's the fusion node).

**Layout**:
```
┌─────────────────────┬─────────────────────────────────────────────┐
│ [Inputs] [Outputs]  │                                             │
│ ┌─────────────────┐ │                                             │
│ │ Input Images    │ │   3D: oriented cameras + point cloud        │
│ │ (gr.File multi) │ │   2D tabs: depth | filtered | confidence   │
│ └─────────────────┘ │         per camera (grouped by 4)           │
│ [Run VGGT Geometry] │                                             │
│ ▸ Config            │                                             │
│   preprocessing_mode│            Rerun Viewer                     │
│   keep_top_percent  │                                             │
│   verbose           │                                             │
│ Examples:           │                                             │
│  [car_landscape_12] │                                             │
│  [rp_capture_6]     │                                             │
└─────────────────────┴─────────────────────────────────────────────┘
```

### SAM3 Segmentation App (update existing `sam3_rerun_ui.py`)

Already exists with the right layout. Update to add Config accordion and `_sync_config()`:

**Config accordion** (add to existing UI):
- `text` → `gr.Textbox(value="person")` (already exists as `text_prompt`, wire into config)
- `mask_threshold` → `gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5)` (new)
- `dilation` → `gr.Slider(minimum=0, maximum=100, step=5, value=50)` (new)

**Rerun log paths** (existing, single-image):
```
image                       rr.Image (RGB)
image/segmentation_ids      rr.SegmentationImage (class-colored masks)
```
Plus `rr.AnnotationContext` at `/` for class colors. Already working.

### Metric Depth App (any `BaseMetricPredictor`)

**Config accordion** (maps 1:1 to `MetricDepthNodeConfig` fields):
- `predictor_name` → `gr.Dropdown(choices=list(get_args(METRIC_PREDICTORS)), value="MoGeV2MetricPredictor")`

Similar to how `depth_inference_ui.py` has a model dropdown for relative predictors. The existing `depth_inference_ui.py` even has a `"Metric (TODO)"` radio — this fulfills that TODO.

**Rerun log paths** (single image):
```
world/                          ViewCoordinates.RDF
world/camera/pinhole/           PinholeProjection (from MoGe's estimated K_33)
world/camera/pinhole/image      rr.Image (RGB)
world/camera/pinhole/depth      rr.DepthImage (metric depth, meter=1)
world/camera/pinhole/confidence rr.Image (confidence map)
world/point_cloud               rr.Points3D (unprojected from metric depth + K_33)
```

**Blueprint**:
```python
rrb.Horizontal(
    rrb.Spatial3DView(origin="world", contents=["+ $origin/**", "- world/camera/pinhole/depth"]),
    rrb.Vertical(
        rrb.Spatial2DView(origin="world/camera/pinhole/image"),
        rrb.Spatial2DView(origin="world/camera/pinhole/depth"),
        rrb.Spatial2DView(origin="world/camera/pinhole/confidence"),
    ),
    column_shares=[3, 1],
)
```

**Layout** (same pattern as VGGT):
```
┌─────────────────────┬─────────────────────────────────────────────┐
│ [Inputs] [Outputs]  │                                             │
│ ┌─────────────────┐ │                                             │
│ │ Input Image     │ │   3D: point cloud from metric depth        │
│ │ (gr.Image)      │ │   2D: image | depth | confidence           │
│ └─────────────────┘ │                                             │
│ [Run Metric Depth]  │            Rerun Viewer (scale=5)           │
│ ▸ Config            │                                             │
│   predictor_name    │                                             │
│ Examples:           │                                             │
│  [example1.jpg]     │                                             │
│  [example2.jpg]     │                                             │
└─────────────────────┴─────────────────────────────────────────────┘
```

### Depth Alignment App

**Config accordion** (maps 1:1 to `DepthAlignmentConfig` fields):
- `edge_threshold` → `gr.Slider(minimum=0.001, maximum=0.1, step=0.005, value=0.01)`
- `scale_only` → `gr.Checkbox(value=False)`

**Rerun log paths** (single view alignment):
```
world/                              ViewCoordinates.RDF
world/reference_depth               rr.DepthImage (reference)
world/target_depth                  rr.DepthImage (target, before alignment)
world/aligned_depth                 rr.DepthImage (target, after alignment)
world/confidence_mask               rr.Image (which pixels were used for alignment)
world/exclusion_mask                rr.Image (excluded regions, e.g. people)
```

**Blueprint**:
```python
rrb.Grid(
    rrb.Spatial2DView(origin="world/reference_depth", name="Reference"),
    rrb.Spatial2DView(origin="world/target_depth", name="Target"),
    rrb.Spatial2DView(origin="world/aligned_depth", name="Aligned"),
    rrb.Spatial2DView(origin="world/confidence_mask", name="Confidence"),
)
```

**Layout** (same pattern as VGGT):
```
┌─────────────────────┬─────────────────────────────────────────────┐
│ [Inputs] [Outputs]  │                                             │
│ ┌─────────────────┐ │                                             │
│ │ Reference Depth │ │   2D grid: reference | target |             │
│ │ (gr.Image)      │ │            aligned  | confidence            │
│ └─────────────────┘ │                                             │
│ ┌─────────────────┐ │            Rerun Viewer (scale=5)           │
│ │ Target Depth    │ │                                             │
│ │ (gr.Image)      │ │                                             │
│ └─────────────────┘ │                                             │
│ [Align Depths]      │                                             │
│ ▸ Config            │                                             │
│   edge_threshold    │                                             │
│   scale_only        │                                             │
│ Examples:           │                                             │
│  (paired depths)    │                                             │
└─────────────────────┴─────────────────────────────────────────────┘
```

### Fusion App

Same layout pattern. Although it's a `FnNode` in daggr, it gets its own standalone Gradio app for debugging/visualization:

**Config accordion** (maps 1:1 to `FusionConfig` fields):
- `grid_resolution` → `gr.Slider(minimum=128, maximum=1024, step=64, value=512)`
- `target_points` → `gr.Slider(minimum=50000, maximum=500000, step=10000, value=150000)`

**Rerun log paths**:
```
world/                      ViewCoordinates.RFU
world/point_cloud           rr.Points3D (downsampled)
world/gt_mesh               rr.Mesh3D (TSDF-fused)
world/camera_{i}/           camera transforms (from pinhole params)
```

**Layout** (same pattern):
```
┌─────────────────────┬─────────────────────────────────────────────┐
│ [Inputs] [Outputs]  │                                             │
│ ┌─────────────────┐ │                                             │
│ │ Depth Maps      │ │   3D: point cloud + mesh + cameras         │
│ │ (gr.File multi) │ │                                             │
│ └─────────────────┘ │                                             │
│ ┌─────────────────┐ │            Rerun Viewer (scale=5)           │
│ │ Pinhole Params  │ │                                             │
│ │ (gr.File JSON)  │ │                                             │
│ └─────────────────┘ │                                             │
│ ┌─────────────────┐ │                                             │
│ │ RGB Images      │ │                                             │
│ │ (gr.File multi) │ │                                             │
│ └─────────────────┘ │                                             │
│ [Run Fusion]        │                                             │
│ ▸ Config            │                                             │
│   grid_resolution   │                                             │
│   target_points     │                                             │
│ Examples:           │                                             │
└─────────────────────┴─────────────────────────────────────────────┘
```

## Serialization Strategy

**In-process** (monolithic orchestrator, composed Gradio app): dataclasses passed directly as Python objects. No serialization needed.

**Daggr boundary** (between GradioNodes): `pyserde` `to_json`/`from_json` — same as wilor-nano's `DetectionResult`. All result dataclasses get `@serde` decorator. Arrays serialize as JSON (base64 or nested lists). If JSON size becomes an issue later, we can optimize then — keep it simple now.

## Files to Create

```
monopriors/
  apis/
    vggt_geometry.py                            # VGGTGeometryConfig, VGGTGeometryResult, run_vggt_geometry()
    metric_depth.py                               # MetricDepthNodeConfig, run_metric_depth() — works with any BaseMetricPredictor
    depth_alignment.py                           # DepthAlignmentConfig, DepthAlignmentResult, run_depth_alignment()
    fusion.py                                   # FusionConfig, FusionResult, run_fusion()

  gradio_ui/
    vggt_geometry_ui.py                         # pred_fn, _sync_config, main() -> gr.Blocks
    metric_depth_ui.py                            # pred_fn, _sync_config, main() -> gr.Blocks (any metric predictor)
    depth_alignment_ui.py                       # pred_fn, main() -> gr.Blocks

tools/
  apps/
    vggt_geometry_app.py                        # 3-line launcher
    metric_depth_app.py                           # 3-line launcher
    depth_alignment_app.py                      # 3-line launcher
  demos/
    vggt_geometry.py                            # tyro CLI
    metric_depth.py                               # tyro CLI
    depth_alignment.py                          # tyro CLI
  daggr_multiview_calibration.py                # Daggr graph wiring all 5 nodes

docs/
  daggr_proposal.md                             # Proposal for daggr team (graph.as_gradio + graph.invoke)
```

In `packages/sam3-rerun/` (modify existing, no new files):
```
sam3_rerun/gradio_ui/sam3_rerun_ui.py           # Update: add Config accordion, _sync_config, match node spec
```

## Files to Modify

- `multiview_calibration.py` — `MultiViewCalibrator.__call__()` becomes thin orchestrator importing from new api modules. Helper functions (`orient_mv_pred_list`, `segment_people`, `mv_pred_to_pointcloud`) move to their respective api modules.
- `multiview_calibration_ui.py` — unchanged (calls refactored calibrator in-process)
- `multiview_inference.py` — import `orient_mv_pred_list` from `vggt_geometry` (currently duplicated)
- `pixi.toml` — add tasks for new apps/demos

## Code Reuse (no changes)

- `VGGTPredictor`, `MultiviewPred`, `robust_filter_confidences` — `models/multiview/vggt_model.py`
- `BaseMetricPredictor`, `MetricDepthPrediction`, `get_metric_predictor()`, `METRIC_PREDICTORS` — `models/metric_depth/`
- `SAM3Predictor`, `SAM3Results` — `sam3_rerun/api/predictor.py`
- `compute_scale_and_shift` — `scale_utils.py`
- `depth_edges_mask`, `multidepth_to_points` — `depth_utils.py`
- Blueprint helpers (`create_final_view`, etc.) — stay in `multiview_calibration.py`

## Phased Implementation

### Phase 0: Daggr proposal document
1. Write `docs/daggr_proposal.md` — the case for `graph.as_gradio()` + `graph.invoke()`
2. Not code, just a document for later use with the daggr team

### Phase 1: VGGT node
1. `apis/vggt_geometry.py` — move `orient_mv_pred_list()` here, create `VGGTGeometryConfig` + `VGGTGeometryResult` + `run_vggt_geometry()`
2. `gradio_ui/vggt_geometry_ui.py` — standalone app with `pred_fn`
3. Entry points + pixi tasks
4. **Verify**: run standalone, compare output against monolithic calibrator

### Phase 2: SAM3 node (update existing in sam3-rerun)
1. Update `sam3_rerun_ui.py` — add Config accordion (`text`, `mask_threshold`, `dilation`), add `_sync_config()`, update `pred_fn` to match node spec
2. No new files — existing app + entry point stay
3. **Verify**: run existing app, verify config controls work, compare masks

### Phase 3: Metric depth node (generalized)
1. `apis/metric_depth.py` — `MetricDepthNodeConfig` + `run_metric_depth()`
   - Uses `get_metric_predictor(name)` factory — works with MoGeV2, UniDepth, or any future `BaseMetricPredictor`
   - Returns `MetricDepthPrediction` (depth_meters, confidence, K_33)
2. `gradio_ui/metric_depth_ui.py` — standalone app with predictor dropdown, single image input
3. Entry points + pixi tasks
4. **Verify**: run standalone with MoGeV2 and UniDepth, compare outputs

### Phase 4: Depth alignment node
1. `apis/depth_alignment.py` — `DepthAlignmentConfig` + `DepthAlignmentResult` + `run_depth_alignment()`
   - Takes: reference depth + target depth + optional confidence/exclusion masks (all generic, per-view)
   - Returns: `DepthAlignmentResult` (single aligned depth map)
2. `gradio_ui/depth_alignment_ui.py` — accepts serialized upstream results
3. Entry points + pixi tasks
4. **Verify**: chain VGGT → MoGe V2 → SAM3 → Depth Alignment manually

### Phase 5: Fusion node + daggr graph
1. `apis/fusion.py` — `FusionConfig` + `FusionResult` + `run_fusion()`
2. `daggr_multiview_calibration.py` — wire all 5 nodes
3. Pixi tasks
4. **Verify**: full daggr graph end-to-end

### Phase 6: Refactor monolithic calibrator
1. `MultiViewCalibrator.__call__()` delegates to decomposed APIs in-process:
   - `run_vggt_geometry(rgb_list)` → geometry
   - loop: `metric_predictor(rgb)` per view → metric depths (via `get_metric_predictor()` factory)
   - loop: `segment_people(rgb)` per view → masks
   - loop: `run_depth_alignment(reference_depth, target_depth, ...)` per view → aligned depths
   - `run_fusion(depth_list, pinholes, rgbs)` → point cloud + mesh
2. `multiview_calibration_ui.py` unchanged (still works via refactored orchestrator)
3. **Verify**: monolithic Gradio UI produces identical output, Rerun streaming works

## Verification

- Each node's standalone Gradio app works independently with example data
- Each node's CLI demo produces correct output
- Daggr graph runs end-to-end: same point cloud + mesh as monolithic calibrator
- Monolithic Gradio UI still works with Rerun streaming after Phase 6
- Numerical: aligned depths within 1e-5 of monolithic for same inputs

## Phase 0: Daggr Team Proposal (docs/daggr_proposal.md)

### The problem

We build CV pipelines as composable daggr graphs (VGGT → SAM3 → MoGe → Scale Alignment → Fusion). Today daggr gives us ONE execution mode: `graph.launch()` which opens the Svelte canvas. But we need THREE modes for the same graph definition:

1. **Canvas mode** (`graph.launch()`) — visual DAG for debugging. Already works.
2. **Gradio app mode** (`graph.as_gradio()`) — auto-generate a standard `gr.Blocks` from the graph. Gives us: Gradio's API, auth, sharing, custom components (`gradio_rerun` for 3D streaming), `.then()` chaining, HF Spaces deployment.
3. **Programmatic mode** (`graph.invoke(inputs)`) — execute headlessly. Gives us: REST/FastAPI integration, batch processing, testing, composability.

### Why `graph.as_gradio()` matters

Every daggr node already has Gradio-typed I/O. The type information for a full Gradio app is already in the graph definition — daggr just needs to assemble it.

```python
graph = Graph(name="MV Calibration", nodes=[vggt, sam3, moge, alignment, fusion])

# Mode 1: Canvas (existing)
graph.launch()

# Mode 2: Gradio app (proposed)
demo: gr.Blocks = graph.as_gradio()
demo.queue().launch()

# Mode 3: Programmatic (proposed, issue #35)
result = graph.invoke({"img_files": ["img1.jpg", "img2.jpg"]})
```

**Benefits over hand-writing a Gradio app:**
- **DRY**: graph definition IS the app definition
- **Custom components**: `gradio_rerun`, etc. — canvas only supports built-in Svelte components
- **Streaming**: generators work in Gradio, not in the canvas
- **Ecosystem**: HF Spaces, OAuth, Gradio Client SDK, MCP

### Our concrete use case

We maintain a monorepo of CV pipelines. Each pipeline follows: standalone Gradio apps per model → daggr graph for composition. Today we must maintain BOTH a daggr graph AND a hand-written Gradio app because:
- Canvas doesn't support `gradio_rerun`
- Canvas doesn't support streaming
- Can't call the pipeline programmatically

With `graph.as_gradio()` + `graph.invoke()`, we define the pipeline ONCE, use it in all three modes.

### Implementation sketch

**`graph.as_gradio()`**: Walk DAG, collect InputNode ports + terminal outputs, create `gr.Blocks` with components, "Run" button triggers topological execution.

**`graph.invoke(inputs)`**: Validate inputs, execute topologically, return terminal output dict.

## Key Reference Files

- `packages/monoprior/monopriors/apis/multiview_calibration.py` — source of logic to decompose
- `packages/wilor-nano/src/wilor_nano/gradio_ui/hand_detection_ui.py` — reference for daggr-compatible `pred_fn`
- `packages/wilor-nano/tools/daggr_wilor.py` — reference for daggr graph wiring
- `packages/sam3-rerun/src/sam3_rerun/api/predictor.py` — SAM3 API to wrap
- `packages/sam3-rerun/src/sam3_rerun/gradio_ui/sam3_rerun_ui.py` — SAM3 Gradio app pattern
- `packages/monoprior/monopriors/models/multiview/vggt_model.py` — VGGT types
- `packages/monoprior/monopriors/models/metric_depth/moge_v2.py` — MoGe V2 metric predictor
- `packages/monoprior/monopriors/models/metric_depth/base_metric_depth.py` — `MetricDepthPrediction` dataclass
- `packages/monoprior/monopriors/scale_utils.py` — `compute_scale_and_shift()`
