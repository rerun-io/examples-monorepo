from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import gradio as gr
import numpy as np
import rerun as rr
from gradio_rerun import Rerun
from jaxtyping import Float32, UInt8
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json

from wilor_nano.gradio_ui.simple_detection_ui import SimpleDetectionJson, centered_box_xyxy
from wilor_nano.gradio_ui.simple_pipeline_shared import (
    DEFAULT_APPLICATION_ID,
    DEFAULT_IMAGE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIEWER_PORT,
    build_simple_blueprint,
    ensure_native_viewer,
    layer_rrd_path,
    load_rgb_image,
    resolve_rgb_image,
    save_native_screenshot,
    switch_to_outputs,
)

SimpleNodeArtifacts = tuple[str | None, bytes, str, str]
SimpleGradioArtifacts = tuple[str | None, bytes, str, dict[str, object]]


@serde
@dataclass(frozen=True, slots=True)
class SimpleKeypointJson:
    """Minimal serde payload produced by the simple keypoint node."""

    application_id: str
    """Rerun application id used for the recording."""
    recording_id: str
    """Rerun recording id reused from the detection JSON."""
    xy: list[float]
    """Single keypoint in pixel-space XY coordinates."""


def default_detection_json() -> str:
    """Create default detection JSON for the standalone keypoint demo."""
    rgb_hw3: UInt8[ndarray, "h w 3"] = load_rgb_image(DEFAULT_IMAGE_PATH)
    image_shape: tuple[int, int, int] = (int(rgb_hw3.shape[0]), int(rgb_hw3.shape[1]), int(rgb_hw3.shape[2]))
    box_xyxy: Float32[ndarray, "1 4"] = centered_box_xyxy(image_shape)
    detection: SimpleDetectionJson = SimpleDetectionJson(
        application_id=DEFAULT_APPLICATION_ID,
        recording_id=str(uuid.uuid4()),
        image_shape=list(image_shape),
        box_xyxy=[float(value) for value in box_xyxy[0].tolist()],
    )
    detection_json: str = to_json(detection)
    return detection_json


def detection_from_json(detection_json: str | dict[str, object] | SimpleDetectionJson) -> SimpleDetectionJson:
    """Parse the simple detection serde payload."""
    if isinstance(detection_json, SimpleDetectionJson):
        return detection_json
    if isinstance(detection_json, str):
        detection: SimpleDetectionJson = from_json(SimpleDetectionJson, detection_json)
        return detection
    detection_dict_json: str = json.dumps(detection_json)
    detection_from_dict: SimpleDetectionJson = from_json(SimpleDetectionJson, detection_dict_json)
    return detection_from_dict


def center_keypoint_from_detection(detection: SimpleDetectionJson) -> Float32[ndarray, "1 2"]:
    """Create one keypoint at the center of the detection box."""
    x1: float = detection.box_xyxy[0]
    y1: float = detection.box_xyxy[1]
    x2: float = detection.box_xyxy[2]
    y2: float = detection.box_xyxy[3]
    center_x: float = (x1 + x2) / 2.0
    center_y: float = (y1 + y2) / 2.0
    keypoint_xy: Float32[ndarray, "1 2"] = np.asarray([[center_x, center_y]], dtype=np.float32)
    return keypoint_xy


def log_simple_keypoint(
    *,
    rgb_hw3: UInt8[ndarray, "h w 3"] | None = None,
    detection_json: str | dict[str, object] | SimpleDetectionJson,
    image_path: Path = DEFAULT_IMAGE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    use_native_viewer: bool = True,
    capture_screenshot: bool = True,
    viewer_port: int = DEFAULT_VIEWER_PORT,
) -> SimpleNodeArtifacts:
    """Log the input image plus one keypoint using the detection serde payload."""
    resolved_image: tuple[UInt8[ndarray, "h w 3"], str] = resolve_rgb_image(rgb_hw3=rgb_hw3, image_path=image_path)
    resolved_rgb_hw3: UInt8[ndarray, "h w 3"] = resolved_image[0]
    detection: SimpleDetectionJson = detection_from_json(detection_json)
    keypoint_xy: Float32[ndarray, "1 2"] = center_keypoint_from_detection(detection)
    rrd_path: Path = layer_rrd_path(output_dir, recording_id=detection.recording_id, layer="keypoint")

    rec: rr.RecordingStream = rr.RecordingStream(
        application_id=detection.application_id, recording_id=detection.recording_id
    )
    binary_stream: rr.BinaryStream = rec.binary_stream()
    if use_native_viewer:
        native_viewer_url: str = ensure_native_viewer(port=viewer_port)
        rec.set_sinks(binary_stream, rr.FileSink(str(rrd_path)), rr.GrpcSink(native_viewer_url))
    else:
        rec.set_sinks(binary_stream, rr.FileSink(str(rrd_path)))

    with rec:
        rec.send_blueprint(build_simple_blueprint(), make_active=True, make_default=True)
        rec.set_time("iteration", sequence=0)
        rec.log("image", rr.Image(image=resolved_rgb_hw3))
        rec.log(
            "image/keypoints/center",
            rr.Points2D(positions=keypoint_xy, labels="center_keypoint", colors=(0, 255, 0), radii=8.0),
        )

    rrd_bytes_or_none: bytes | None = binary_stream.read(flush=True, flush_timeout_sec=5.0)
    rrd_bytes: bytes = rrd_bytes_or_none if rrd_bytes_or_none is not None else b""
    screenshot_path: Path | None = None
    if use_native_viewer and capture_screenshot:
        screenshot_path = save_native_screenshot(output_dir, prefix="simple_keypoint", port=viewer_port)

    keypoint: SimpleKeypointJson = SimpleKeypointJson(
        application_id=detection.application_id,
        recording_id=detection.recording_id,
        xy=[float(value) for value in keypoint_xy[0].tolist()],
    )
    keypoint_json: str = to_json(keypoint)
    screenshot_path_str: str | None = str(screenshot_path) if screenshot_path is not None else None
    return screenshot_path_str, rrd_bytes, str(rrd_path), keypoint_json


def run_simple_keypoint(
    rgb_hw3: UInt8[ndarray, "h w 3"] | None,
    detection_json: str | dict[str, object],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    use_native_viewer: bool = True,
    capture_screenshot: bool = True,
) -> Iterator[SimpleGradioArtifacts]:
    """Gradio event handler for the simple keypoint node."""
    screenshot_path: str | None
    rrd_bytes: bytes
    rrd_path: str
    keypoint_json: str
    screenshot_path, rrd_bytes, rrd_path, keypoint_json = log_simple_keypoint(
        rgb_hw3=rgb_hw3,
        detection_json=detection_json,
        output_dir=output_dir,
        use_native_viewer=use_native_viewer,
        capture_screenshot=capture_screenshot,
    )
    keypoint_value: dict[str, object] = cast(dict[str, object], json.loads(keypoint_json))
    outputs: SimpleGradioArtifacts = (screenshot_path, rrd_bytes, rrd_path, keypoint_value)
    yield outputs


def main() -> gr.Blocks:
    """Build the simple keypoint Gradio app."""
    viewer = Rerun(
        streaming=True,
        panel_states={
            "time": "collapsed",
            "blueprint": "hidden",
            "selection": "hidden",
        },
        height=800,
    )
    default_json: str = default_detection_json()
    default_value: dict[str, object] = cast(dict[str, object], json.loads(default_json))

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        rgb_hw3 = gr.Image(
                            value=str(DEFAULT_IMAGE_PATH),
                            interactive=True,
                            label="Image",
                            type="numpy",
                            image_mode="RGB",
                        )
                        detection_json = gr.JSON(value=default_value, label="Detection JSON")
                        create_rrd = gr.Button("Log Keypoint")

                    with gr.TabItem("Outputs", id="outputs"):
                        native_screenshot = gr.Image(label="Native Viewer Screenshot", type="filepath")
                        rrd_file = gr.File(label="Saved Keypoint RRD")
                        keypoint_json = gr.JSON(label="Keypoint JSON")

                gr.Examples(
                    examples=[[str(DEFAULT_IMAGE_PATH), default_value]],
                    inputs=[rgb_hw3, detection_json],
                    cache_examples=False,
                    examples_per_page=2,
                )

            with gr.Column(scale=5):
                viewer.render()

        create_rrd.click(
            fn=switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(
            run_simple_keypoint,
            inputs=[rgb_hw3, detection_json],
            outputs=[native_screenshot, viewer, rrd_file, keypoint_json],
            api_name="simple_keypoint_fn",
        )

    return demo
