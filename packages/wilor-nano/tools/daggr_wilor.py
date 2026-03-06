"""Daggr workflow connecting hand detection and keypoint estimation.

This requires running both apps on separate ports:
- Detection app: pixi run -e dev python tools/app_detection.py (defaults to 7860)
- Keypoint app: pixi run -e dev python tools/app_keypoint.py (port 7861)

Then run this script to launch the daggr graph:
- pixi run -e dev python tools/daggr_wilor.py
"""

from daggr import GradioNode, Graph
import gradio as gr
from gradio_rerun import Rerun


# Shared image input - both detection and keypoint nodes use this
shared_image = gr.Image(label="Input Image")

# Detection node - connects to hand detection app at port 7860
detection_node = GradioNode(
    "http://localhost:7860",
    api_name="/pred_fn",
    name="Hand Detection Node",
    inputs={"rgb_hw3": shared_image},
    outputs={
        "rrd": Rerun(streaming=True, visible=False),
        "detection_json": gr.JSON(visible=True),
    },
)

# Keypoint node - connects to keypoint estimation app at port 7861
# Uses same shared_image input + detection_json from upstream
keypoint_node = GradioNode(
    "http://localhost:7861",
    api_name="/pred_fn",
    name="Keypoint Estimation Node",
    inputs={
        "rgb_hw3": shared_image,  # Same shared input as detection
        "detection_json": detection_node.detection_json,  # Detection results
    },
    outputs={
        "rrd": Rerun(streaming=True, visible=False),
        "keypoint_json": gr.JSON(visible=True),
    },
)


graph = Graph(
    name="WiLor Hand Pose Pipeline",
    nodes=[detection_node, keypoint_node],
)

if __name__ == "__main__":
    graph.launch()