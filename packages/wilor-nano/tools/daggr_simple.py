"""Daggr workflow for the minimal detection-to-keypoint Rerun demo.

Run the two Gradio nodes first:
- GRADIO_SERVER_PORT=7863 pixi run -e wilor-dev --frozen python packages/wilor-nano/tools/app_simple_detection.py
- GRADIO_SERVER_PORT=7864 pixi run -e wilor-dev --frozen python packages/wilor-nano/tools/app_simple_keypoint.py

Then launch this graph:
- GRADIO_SERVER_PORT=7865 pixi run -e wilor-dev --frozen python packages/wilor-nano/tools/daggr_simple.py
"""

import os

import gradio as gr
from daggr import GradioNode, Graph  # pyrefly: ignore[missing-import]
from gradio_rerun import Rerun

from wilor_nano.gradio_ui.simple_pipeline_shared import DEFAULT_IMAGE_PATH

DETECTION_APP_URL: str = os.environ.get("WILOR_SIMPLE_DETECTION_URL", "http://localhost:7860")
KEYPOINT_APP_URL: str = os.environ.get("WILOR_SIMPLE_KEYPOINT_URL", "http://localhost:7861")

shared_image = gr.Image(label="Input Image", value=str(DEFAULT_IMAGE_PATH))

detection_node = GradioNode(
    DETECTION_APP_URL,
    api_name="/simple_detection_fn",
    name="Simple Detection",
    inputs={"rgb_hw3": shared_image},
    outputs={
        "detection_screenshot": gr.Image(label="Detection Screenshot"),
        "detection_rrd": Rerun(streaming=True),
        "detection_rrd_file": gr.File(label="Detection RRD"),
        "detection_json": gr.JSON(label="Detection JSON"),
    },
)

keypoint_node = GradioNode(
    KEYPOINT_APP_URL,
    api_name="/simple_keypoint_fn",
    name="Simple Keypoint",
    inputs={
        "rgb_hw3": shared_image,
        "detection_json": detection_node.detection_json,
    },
    outputs={
        "keypoint_screenshot": gr.Image(label="Keypoint Screenshot"),
        "keypoint_rrd": Rerun(streaming=True),
        "keypoint_rrd_file": gr.File(label="Keypoint RRD"),
        "keypoint_json": gr.JSON(label="Keypoint JSON"),
    },
)

graph = Graph(
    name="Simple Detection + Keypoint",
    nodes=[detection_node, keypoint_node],
)

if __name__ == "__main__":
    graph.launch()
