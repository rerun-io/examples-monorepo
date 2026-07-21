# Application Architecture: CLI + Gradio + Rerun + Daggr

This document describes the layered architecture used by the multiview calibration pipeline as a reference pattern for building composable, node-based CV applications. The same structure can be applied to any model pipeline that needs CLI access, a web UI, and graph composition via daggr.

## Why this architecture

The core problem: a CV pipeline (like multiview calibration) needs to be usable in three different contexts:

1. **CLI** -- batch processing, scripting, CI pipelines
2. **Gradio UI** -- interactive demos with streaming 3D visualization
3. **Daggr graph** -- composing multiple pipelines into a DAG where outputs of one node feed into another

Rather than writing three separate implementations, the architecture uses a shared API layer that all three entry points call into. The key insight is that the pipeline's true input is `list[RGB]` and its true output is `MVCalibResults` -- everything else (file I/O, widget parsing, streaming) is context-specific glue.

## The three layers

```
tools/                        Entry points (thin wrappers)
  demos/                        CLI scripts (tyro)
  apps/                         Gradio app launchers
  daggr_*.py                    Daggr graph definitions

gradio_ui/                    UI layer (Gradio-specific glue)
  multiview_calibration_ui.py   Widget management, streaming, state

apis/                         API layer (pure computation + Rerun logging)
  multiview_calibration.py      Pipeline orchestration, config, models
```

### Layer 1: API (`apis/multiview_calibration.py`)

The computational core. Has no knowledge of Gradio, file dialogs, or widget values.

**What lives here:**
- `MultiViewCalibratorConfig` -- dataclass with all pipeline parameters, compatible with tyro for CLI parsing
- `run_multiview_calibration()` -- shared geometry and post-processing function using a caller-owned predictor
- `log_calibration_results()` -- Rerun blueprint, final output logging, and TSDF fusion
- `MVCalibResults` -- return dataclass (depth maps, pinhole parameters, point cloud)
- `load_rgb_images()` -- shared I/O utility for loading image files as RGB arrays
- Rerun blueprint builders (`create_final_view`, `create_tabbed_camera_view`, etc.)

**Design choices:**
- The predictor is an explicit `run_multiview_calibration()` argument. The CLI owns a private predictor; Gradio owns a predictor-cache lease. The function never retains either one.
- `log_calibration_results()` calls `rr.log()` using the thread-local recording, so the caller controls where data goes (global recording for CLI, `with recording:` context for Gradio streaming)
- The pipeline takes `list[UInt8[ndarray, "H W 3"]]` -- already-loaded RGB arrays, not file paths. This keeps I/O concerns out of the pipeline
- Geometry settings are composed through `MultiviewGeometryConfig`, which is shared by the calibration, geometry, and standalone inference paths.

### Layer 2: Gradio UI (`gradio_ui/multiview_calibration_ui.py`)

Translates between Gradio widgets and the API layer. Manages model caches and streaming.

**What lives here:**
- `MultiviewCalibrationRequest` -- immutable snapshot of uploaded images and every widget value for one run
- `PREDICTOR_CACHE` -- atomic, single-resident VGGT/G3T predictor cache keyed by effective runtime configuration
- `AUXILIARY_MODEL_CACHE` -- lazy per-device SAM3 and MoGe cache
- `_prepare_request()` -- validates widget values and converts `gr.File` paths to the pipeline's `list[RGB]` contract
- `multiview_calibration_fn()` -- streaming callback that creates a `BinaryStream`, runs the pipeline inside a recording context, and yields bytes to the Rerun viewer
- `main()` -- builds and returns the `gr.Blocks` layout

**The click chain:**
```python
run_btn.click(
    _switch_to_outputs,      # UI: switch tabs
).then(
    lambda: uuid.uuid4(),    # Session: fresh recording ID
).then(
    _prepare_request,        # Atomically capture files + widget configuration
).then(
    multiview_calibration_fn # Pipeline: run calibration, stream to Rerun viewer
)
```

Each run receives one immutable request, so concurrent sessions cannot observe a backend or operation configuration selected by another session. Loading remains lazy: repeated runs reuse the matching predictor, while backend replacement closes the old multi-GB model before constructing the new one.

### Layer 3: Entry points (`tools/`)

Minimal wrappers that wire a UI or config to the API.

**CLI** (`tools/demos/multiview_calibration.py`):
```python
import tyro
from monopriors.apis.multiview_calibration import MVInferenceConfig, main

if __name__ == "__main__":
    main(tyro.cli(MVInferenceConfig))
```

**Gradio app** (`tools/apps/calibration_app.py`):
```python
from monopriors.gradio_ui.multiview_calibration_ui import main

demo = main()
if __name__ == "__main__":
    demo.queue().launch(ssr_mode=False)
```

**Daggr graph** (`tools/daggr_multiview_calibration.py`) -- the calibration pipeline is composed from four independent node apps rather than a single calibration endpoint. Each node runs on its own port (multiview geometry on 7870, SAM3 segmentation on 7871, metric depth on 7872, depth alignment on 7873):
```python
from daggr import GradioNode, Graph

# Node 1: Multiview Geometry -- oriented poses, depths, confidences
multiview_node = GradioNode(
    "http://localhost:7870",
    api_name="/multiview_geometry_fn",
    name="Multiview Geometry",
    inputs={"img_files": shared_images},
    outputs={"rrd": Rerun(streaming=True, visible=False), "status": gr.Textbox()},
)

# Plus sam3_node (7871, /sam3d_prediction_fn), metric_depth_node (7872,
# /metric_depth_fn), and alignment_node (7873, /depth_alignment_fn)

graph = Graph(
    name="Multi-View Calibration Pipeline",
    nodes=[multiview_node, sam3_node, metric_depth_node, alignment_node],
)
```

## How Rerun streaming works

The Gradio Rerun viewer (`gradio-rerun`) accepts `bytes` -- raw RRD data chunks. The streaming pattern:

```python
def multiview_calibration_fn(recording_id, request):
    # 1. Create a recording stream bound to this session
    recording = rr.RecordingStream(application_id="app", recording_id=recording_id)
    stream = recording.binary_stream()

    # 2. Run inside the recording context and predictor-cache lease
    with recording:
        with PREDICTOR_CACHE.acquire(request.config.predictor_config) as predictor:
            output = run_multiview_calibration(
                rgb_list=request.rgb_list,
                multiview_predictor=predictor,
                config=request.config,
                ...,
            )
        log_calibration_results(rgb_list=request.rgb_list, output=output, ...)

    # 3. Yield the accumulated bytes to the Gradio Rerun viewer
    yield stream.read(), "Calibration complete"
```

Key: the API doesn't know about streaming. Its `rr.log()` calls target the active recording. The `with recording:` context in the UI redirects them to the binary stream, while the CLI uses its globally configured viewer/file sink.

## Daggr composition: wilor-nano as reference

The wilor-nano package demonstrates how this architecture enables graph composition. Two independent Gradio apps (hand detection on port 7860, keypoint estimation on port 7861) are composed into a DAG:

```python
# Each app is a self-contained Gradio server with its own models
detection_node = GradioNode(
    "http://localhost:7860",
    api_name="/pred_fn",
    inputs={"rgb_hw3": shared_image},
    outputs={"rrd": Rerun(...), "detection_json": gr.JSON()},
)

keypoint_node = GradioNode(
    "http://localhost:7861",
    api_name="/pred_fn",
    inputs={
        "rgb_hw3": shared_image,                       # shared input
        "detection_json": detection_node.detection_json # upstream output
    },
    outputs={"rrd": Rerun(...), "keypoint_json": gr.JSON()},
)

graph = Graph(name="WiLor Pipeline", nodes=[detection_node, keypoint_node])
```

**What makes this work:**
- Each Gradio app exposes a named API endpoint (the callback function name)
- Inputs/outputs are typed Gradio components, so daggr knows how to wire them
- Upstream outputs (`detection_node.detection_json`) become downstream inputs automatically
- Each node runs in its own process with its own GPU memory -- no model conflicts

**The tradeoff vs monolithic:**

| | Monolithic (monoprior calibration) | Composed (wilor-nano via daggr) |
|---|---|---|
| **When to use** | Models are tightly coupled (VGGT feeds SAM3 feeds MoGe) | Stages are independently useful |
| **Model loading** | Single process, shared GPU memory | Separate processes, isolated memory |
| **Config management** | Immutable per-run request + keyed model cache | Each node has its own config |
| **Latency** | Single process, no serialization overhead | JSON serialization between nodes |
| **Reusability** | Pipeline is one unit | Each node is independently deployable |

## Applying this pattern to a new pipeline

To add a new CV pipeline (e.g., "scene reconstruction") following this architecture:

### 1. API layer (`apis/scene_reconstruction.py`)
```python
@dataclass
class SceneReconstructionConfig:
    """Tyro-compatible config for CLI + UI."""
    resolution: int = 512
    use_normals: bool = True
    device: Literal["cuda", "cpu"] = "cuda"

def run_scene_reconstruction(
    *, rgb_list, predictor, config
) -> SceneResults:
    """Shared computation with explicit, caller-owned model dependencies."""
    ...
```

### 2. Gradio UI (`gradio_ui/scene_reconstruction_ui.py`)
```python
@dataclass(frozen=True, slots=True)
class SceneReconstructionRequest:
    rgb_list: list[np.ndarray]
    config: SceneReconstructionConfig

def _prepare_request(img_files, resolution, use_normals): ...
def reconstruction_fn(recording_id, request): ...
def main() -> gr.Blocks: ...
```

### 3. Entry points (`tools/`)
```python
# tools/demos/scene_reconstruction.py (CLI)
main(tyro.cli(SceneReconstructionConfig))

# tools/apps/reconstruction_app.py (Gradio)
demo = main()
demo.queue().launch()

# tools/daggr_reconstruction.py (graph node)
node = GradioNode("http://localhost:7860", api_name="/reconstruction_fn", ...)
```

## Key principles

1. **The pipeline's input is domain data, not UI artifacts.** `list[RGB]`, not `gr.File` paths. Keep I/O translation in the UI layer.

2. **Rerun logging uses thread-local recordings.** The pipeline calls `rr.log()` without knowing the destination. The caller (`with recording:` or global) controls routing.

3. **Model ownership is explicit.** The CLI owns a private model; the UI owns a keyed cache. Per-run requests are immutable, and computation functions never retain leased models.

4. **Each `.then()` step has one job.** Tab switching, session management, atomic request capture, and pipeline execution are separate steps. Failures don't cascade past their boundary.

5. **Gradio apps are also daggr nodes.** If your callback has a named API endpoint and typed inputs/outputs, daggr can compose it into a graph. Design the Gradio app to work standalone first, then wire it into a graph.
