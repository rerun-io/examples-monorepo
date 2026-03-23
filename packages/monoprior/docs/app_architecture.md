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
- `MultiViewCalibrator` -- orchestrator class that loads and runs models (VGGT, SAM3, MoGe)
- `run_calibration_pipeline()` -- the shared pipeline function: sets up Rerun blueprint, runs calibration, logs results, fuses TSDF mesh
- `MVCalibResults` -- return dataclass (depth maps, pinhole parameters, point cloud)
- `load_rgb_images()` -- shared I/O utility for loading image files as RGB arrays
- Rerun blueprint builders (`create_final_view`, `create_tabbed_camera_view`, etc.)

**Design choices:**
- `run_calibration_pipeline()` calls `rr.log()` using the thread-local recording, so the caller controls where data goes (global recording for CLI, `with recording:` context for Gradio streaming)
- The pipeline takes `list[UInt8[ndarray, "H W 3"]]` -- already-loaded RGB arrays, not file paths. This keeps I/O concerns out of the pipeline
- `MultiViewCalibrator.__call__()` is a pure function: RGB in, results out. Rerun logging only happens when `config.verbose=True` (per-camera detail) and always in `run_calibration_pipeline` (final outputs)

### Layer 2: Gradio UI (`gradio_ui/multiview_calibration_ui.py`)

Translates between Gradio widgets and the API layer. Manages model singletons and streaming.

**What lives here:**
- `_MV_CALIBRATOR` -- module-level singleton, loaded once at import, reused across runs
- `_sync_config()` -- reads widget values, updates config, conditionally re-inits models only when toggling features ON that require new model weights
- `_parse_and_load_images()` -- converts `gr.File` paths to `list[RGB]` via the shared `load_rgb_images()`
- `multiview_calibration_fn()` -- streaming callback that creates a `BinaryStream`, runs the pipeline inside a recording context, and yields bytes to the Rerun viewer
- `main()` -- builds and returns the `gr.Blocks` layout

**The click chain:**
```python
run_btn.click(
    _switch_to_outputs,      # UI: switch tabs
).then(
    lambda: uuid.uuid4(),    # Session: fresh recording ID
).then(
    _sync_config,            # Config: sync widgets -> calibrator singleton
).then(
    _parse_and_load_images,  # I/O: Gradio file paths -> RGB arrays (via gr.State)
).then(
    multiview_calibration_fn # Pipeline: run calibration, stream to Rerun viewer
)
```

Each step has a single responsibility. If the config sync fails, images are never loaded. If image loading fails, the pipeline never runs. The chain is readable top-to-bottom.

**Why `_sync_config` exists:**
Loading VGGT + SAM3 + MoGe takes ~7 seconds. The UI can't reload them on every run. Instead, models are loaded once at import, and `_sync_config` patches the config in-place for runtime-only fields (`keep_top_percent`, `preprocessing_mode`). Only when a user enables a previously-disabled feature (toggling `segment_people` or `refine_depth_maps` from OFF to ON) does it re-create the calibrator, because that feature's model was never loaded.

**Why `_parse_and_load_images` is a separate step:**
The pipeline's input contract is `list[RGB]`. Gradio's `gr.File` returns file paths. Keeping the translation in its own `.then()` step means `multiview_calibration_fn` never touches file paths -- it receives pre-loaded arrays from `gr.State`, matching exactly what the API layer expects.

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

**Daggr node** (not yet implemented for calibration, but the pattern from wilor-nano):
```python
from daggr import GradioNode, Graph

calibration_node = GradioNode(
    "http://localhost:7860",
    api_name="/multiview_calibration_fn",
    name="Multiview Calibration",
    inputs={"img_files": shared_images},
    outputs={"rrd": Rerun(streaming=True), "status": gr.Textbox()},
)
```

## How Rerun streaming works

The Gradio Rerun viewer (`gradio-rerun`) accepts `bytes` -- raw RRD data chunks. The streaming pattern:

```python
def multiview_calibration_fn(recording_id, rgb_list):
    # 1. Create a recording stream bound to this session
    recording = rr.RecordingStream(application_id="app", recording_id=recording_id)
    stream = recording.binary_stream()

    # 2. Run the pipeline inside the recording context
    #    All rr.log() calls in run_calibration_pipeline go to this stream
    with recording:
        run_calibration_pipeline(rgb_list=rgb_list, mv_calibrator=_MV_CALIBRATOR, ...)

    # 3. Yield the accumulated bytes to the Gradio Rerun viewer
    yield stream.read(), "Calibration complete"
```

Key: `run_calibration_pipeline` doesn't know about streaming. It just calls `rr.log()`. The `with recording:` context in the UI layer redirects those calls to the binary stream. This is why the same pipeline function works for both CLI (global recording → viewer/file) and Gradio (scoped recording → binary stream → viewer component).

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
| **Config management** | `_sync_config` manages one calibrator | Each node has its own config |
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

class SceneReconstructor:
    """Model orchestrator. Loads models once, reused across calls."""
    def __init__(self, config): ...
    def __call__(self, *, rgb_list) -> SceneResults: ...

def run_reconstruction_pipeline(
    *, rgb_list, reconstructor, parent_log_path, timeline
) -> SceneResults:
    """Shared pipeline: blueprint + reconstruct + log. Uses thread-local rr recording."""
    ...
```

### 2. Gradio UI (`gradio_ui/scene_reconstruction_ui.py`)
```python
_CONFIG = SceneReconstructionConfig(device="cuda", verbose=True)
_RECONSTRUCTOR = SceneReconstructor(config=_CONFIG)

def _sync_config(resolution, use_normals): ...
def _parse_and_load_images(img_files): ...
def reconstruction_fn(recording_id, rgb_list): ...
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

3. **Model singletons live in the UI module.** Loaded once at import, patched by `_sync_config`. The API layer creates fresh instances (CLI use case where startup cost is paid once).

4. **Each `.then()` step has one job.** Tab switching, session management, config sync, image parsing, and pipeline execution are separate steps. Failures don't cascade past their boundary.

5. **Gradio apps are also daggr nodes.** If your callback has a named API endpoint and typed inputs/outputs, daggr can compose it into a graph. Design the Gradio app to work standalone first, then wire it into a graph.
