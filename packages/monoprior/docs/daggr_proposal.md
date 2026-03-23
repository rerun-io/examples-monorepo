# Proposal: `graph.as_gradio()` and `graph.invoke()` for Daggr

## Summary

We propose two new methods on `daggr.Graph`:

1. **`graph.as_gradio()`** — auto-generate a standard `gr.Blocks` app from a daggr graph definition
2. **`graph.invoke(inputs)`** — execute the graph headlessly as a Python function call

These complement the existing `graph.launch()` (Svelte canvas) to give users three execution modes from a single graph definition.

## The problem

We build computer vision pipelines as composable daggr graphs. A typical pipeline (multi-view 3D calibration) looks like:

```
                ┌──→ VGGT (geometry) ──────┐
                │                           │
shared_images ──┼──→ SAM3 (segmentation) ──┼──→ Depth Alignment ──→ Fusion
                │                           │
                └──→ MoGe (metric depth) ──┘
```

Each node is an independent Gradio app with its own model, CLI, and UI. Daggr composes them into a DAG. This works well for **visual debugging** via `graph.launch()`, but we hit walls when we need the same pipeline in other contexts:

| Need | `graph.launch()` | Missing capability |
|------|------------------|--------------------|
| Live 3D visualization (Rerun viewer) | Canvas has no `gradio_rerun` component | `graph.as_gradio()` |
| Streaming intermediate results (`yield`) | No generator support within nodes | `graph.as_gradio()` |
| Batch processing over 1000 scenes | No headless execution | `graph.invoke()` |
| REST API endpoint | No programmatic access | `graph.invoke()` |
| pytest integration tests | Can't call graph as a function | `graph.invoke()` |
| HF Spaces deployment | Canvas is not a standard Gradio app | `graph.as_gradio()` |

Today we maintain **two parallel implementations** for every pipeline: a daggr graph (for composition/debugging) and a hand-written Gradio app (for everything else). They must stay in sync, which is error-prone and doubles the maintenance burden.

## Proposed API

```python
from daggr import GradioNode, FnNode, InputNode, Graph
import gradio as gr

# Define nodes (same as today)
shared_images = InputNode(ports={"images": gr.File(file_count="multiple")})

vggt = GradioNode("http://localhost:7870", api_name="/pred_fn", ...)
sam3 = GradioNode("http://localhost:7871", api_name="/pred_fn", ...)
moge = GradioNode("http://localhost:7872", api_name="/pred_fn", ...)
alignment = GradioNode("http://localhost:7873", api_name="/pred_fn", ...)
fusion = FnNode(fn=run_fusion, ...)

graph = Graph(
    name="Multi-View Calibration",
    nodes=[vggt, sam3, moge, alignment, fusion],
)

# Mode 1: Visual canvas (existing)
graph.launch()

# Mode 2: Standard Gradio app (proposed)
demo: gr.Blocks = graph.as_gradio()
demo.queue().launch()  # Full Gradio: API, auth, sharing, custom components, streaming

# Mode 3: Programmatic execution (proposed, extends issue #35)
result = graph.invoke({"images": ["img1.jpg", "img2.jpg"]})
print(result["fusion_output"])  # Access terminal node outputs by port name
```

## Why `graph.as_gradio()` matters

### The type information already exists

Every daggr node already declares Gradio-typed inputs and outputs:

```python
vggt = GradioNode(
    inputs={"images": gr.File(file_count="multiple")},   # ← Gradio component
    outputs={"geometry_json": gr.JSON(), "rrd": Rerun()}, # ← Gradio component
)
```

An `InputNode` has `ports={"prompt": gr.Textbox()}`. The full schema for a Gradio app — input components, output components, and the execution function — is already encoded in the graph. Daggr just needs to assemble it into `gr.Blocks`.

### What it unlocks

- **Custom components**: `gradio_rerun` for 3D visualization, `gradio_molecule3d`, or any third-party component. The canvas only supports its built-in Svelte component set.
- **Streaming**: Gradio generators (`yield`) work natively. We stream Rerun binary data incrementally to the viewer — impossible in the canvas today.
- **Full Gradio ecosystem**: HF Spaces deployment, OAuth, `gr.Client` for remote calls, MCP server integration, Gradio API endpoints.
- **DRY**: The graph definition IS the app definition. No maintaining two parallel implementations.

### Implementation sketch

1. Walk the DAG, collect all `InputNode` ports → these become `gr.Blocks` input components
2. Collect terminal node outputs → these become `gr.Blocks` output components
3. Create a "Run" button that triggers topological execution
4. For each node in topological order: call its underlying function (`FnNode`) or Gradio API (`GradioNode`)
5. If a node's function is a generator, `yield` intermediate results to the corresponding output component
6. Return the assembled `gr.Blocks`

## Why `graph.invoke()` matters

### For testing

```python
def test_calibration_pipeline():
    graph = build_calibration_graph()
    result = graph.invoke({"images": test_images})
    assert "fusion_output" in result
    # Verify output structure, numerical accuracy, etc.
```

### For batch processing

```python
graph = build_calibration_graph()
for scene_dir in all_scenes:
    images = sorted(scene_dir.glob("*.jpg"))
    result = graph.invoke({"images": [str(p) for p in images]})
    save_output(result, scene_dir / "calibration_output")
```

### For server integration

```python
app = FastAPI()
graph = build_calibration_graph()

@app.post("/calibrate")
async def calibrate(images: list[UploadFile]):
    paths = [save_temp(img) for img in images]
    result = graph.invoke({"images": paths})
    return {"status": "complete", "output": result["fusion_output"]}
```

### Implementation sketch

1. Validate `inputs` dict against the graph's `InputNode` ports
2. Execute nodes in topological order, passing outputs to downstream inputs via edge mapping
3. For `GradioNode`: use `gradio_client.Client.predict()` (already used internally by daggr)
4. For `FnNode`: call the function directly
5. Return a dict mapping terminal output port names to their values

## Our use case

We maintain a [monorepo of CV pipelines](https://github.com/rerun-io/examples-monorepo) built on [Rerun](https://rerun.io) for 3D visualization. Each pipeline follows the same architecture:

1. **Self-contained nodes**: Each model (VGGT geometry, SAM3 segmentation, MoGe depth) has its own standalone Gradio app with CLI, config dataclass, and Rerun integration
2. **Daggr graph**: Composes the standalone apps into a pipeline via `GradioNode` wiring
3. **Composed Gradio app**: A hand-written `gr.Blocks` app that chains the same API functions via `.then()` for the production UX (live Rerun streaming)

Today, (2) and (3) are separate codebases that must stay in sync. With `graph.as_gradio()`, we define the pipeline once as a daggr graph and get both the canvas and the Gradio app from the same definition.

### Existing daggr usage

We already use daggr for our [WiLor hand pose pipeline](https://github.com/rerun-io/examples-monorepo/blob/main/packages/wilor-nano/tools/daggr_wilor.py), which composes a hand detection app and a keypoint estimation app into a two-node DAG. We want to scale this pattern to larger pipelines (5+ nodes) where maintaining a parallel hand-written Gradio app becomes impractical.

## Relation to existing issues

This proposal extends [issue #35](https://github.com/gradio-app/daggr/issues/35) ("Add `.invoke()` Method to Graph API") with the additional `graph.as_gradio()` capability. The `.invoke()` method addresses programmatic execution; `graph.as_gradio()` addresses the UI/ecosystem gap between the daggr canvas and standard Gradio apps.
