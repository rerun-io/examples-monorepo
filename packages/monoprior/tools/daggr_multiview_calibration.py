"""Daggr workflow composing multi-view calibration from independent nodes.

This requires running each node app on a separate port:
- VGGT Geometry:    pixi run -e monoprior vggt-geometry-app     (port 7870)
- SAM3 Segmentation: pixi run -e sam3-rerun sam3-rerun-app      (port 7871)
- Metric Depth:     pixi run -e monoprior metric-depth-app      (port 7872)
- Depth Alignment:  pixi run -e monoprior depth-alignment-app   (port 7873)

Then run this script to launch the daggr graph:
    pixi run -e monoprior-dev python tools/daggr_multiview_calibration.py
"""

import gradio as gr
from daggr import GradioNode, Graph
from gradio_rerun import Rerun

# Shared input — all model nodes receive the same images
shared_images = gr.File(label="Input Images", file_count="multiple", file_types=[".png", ".jpg", ".jpeg"])

# Node 1: Multiview Geometry — produces oriented poses, depths, confidences
multiview_node = GradioNode(
    "http://localhost:7870",
    api_name="/multiview_geometry_fn",
    name="Multiview Geometry",
    inputs={"img_files": shared_images},
    outputs={
        "rrd": Rerun(streaming=True, visible=False),
        "status": gr.Textbox(visible=True),
    },
)

# Node 2: SAM3 Segmentation — produces per-view person masks
sam3_node = GradioNode(
    "http://localhost:7871",
    api_name="/sam3d_prediction_fn",
    name="SAM3 Segmentation",
    inputs={"img": gr.Image(label="Input Image")},
    outputs={
        "rrd": Rerun(streaming=True, visible=False),
        "status": gr.Textbox(visible=True),
    },
)

# Node 3: Metric Depth — produces metric-scale depth per view
metric_depth_node = GradioNode(
    "http://localhost:7872",
    api_name="/metric_depth_fn",
    name="Metric Depth",
    inputs={"img": gr.Image(label="Input Image")},
    outputs={
        "rrd": Rerun(streaming=True, visible=False),
        "status": gr.Textbox(visible=True),
    },
)

# Node 4: Depth Alignment — aligns metric depth to VGGT coordinate frame
alignment_node = GradioNode(
    "http://localhost:7873",
    api_name="/depth_alignment_fn",
    name="Depth Alignment",
    inputs={
        "reference_img": gr.Image(label="Reference Depth"),
        "target_img": gr.Image(label="Target Depth"),
    },
    outputs={
        "rrd": Rerun(streaming=True, visible=False),
        "status": gr.Textbox(visible=True),
    },
)

graph = Graph(
    name="Multi-View Calibration Pipeline",
    nodes=[multiview_node, sam3_node, metric_depth_node, alignment_node],
)

if __name__ == "__main__":
    graph.launch()
