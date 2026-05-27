from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import gradio as gr
import numpy as np
import rerun as rr
from gradio_rerun import Rerun
from jaxtyping import Float32, UInt8
from numpy import ndarray
from serde import serde
from serde.json import to_json

from wilor_nano.gradio_ui.simple_pipeline_shared import (
    DEFAULT_APPLICATION_ID,
    DEFAULT_IMAGE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIEWER_PORT,
    build_simple_blueprint,
    ensure_native_viewer,
    ensure_recording_id,
    layer_rrd_path,
    resolve_rgb_image,
    save_native_screenshot,
    switch_to_outputs,
)

DEFAULT_BOX_SCALE: Final[float] = 0.25
SimpleNodeArtifacts = tuple[str | None, bytes, str, str]
SimpleGradioArtifacts = tuple[str | None, bytes, str, dict[str, object]]


@serde
@dataclass(frozen=True, slots=True)
class SimpleDetectionJson:
    """Minimal serde payload produced by the simple detection node."""

    application_id: str
    """Rerun application id used for the recording."""
    recording_id: str
    """Rerun recording id shared with downstream nodes."""
    image_shape: list[int]
    """Input RGB image shape as [height, width, channels]."""
    box_xyxy: list[float]
    """Single detection box in pixel-space XYXY coordinates."""


def centered_box_xyxy(
    image_shape: tuple[int, int, int],
    scale: float = DEFAULT_BOX_SCALE,
) -> Float32[ndarray, "1 4"]:
    """Create one centered XYXY box for an image shape.

    Args:
        image_shape: RGB image shape as height, width, channels.
        scale: Fraction of image width and height covered by the box.

    Returns:
        Float32[ndarray, "1 4"]: One XYXY box.
    """
    height: int = image_shape[0]
    width: int = image_shape[1]
    box_width: float = float(width) * scale
    box_height: float = float(height) * scale
    x1: float = (float(width) - box_width) / 2.0
    y1: float = (float(height) - box_height) / 2.0
    x2: float = x1 + box_width
    y2: float = y1 + box_height
    box_xyxy: Float32[ndarray, "1 4"] = np.asarray([[x1, y1, x2, y2]], dtype=np.float32)
    return box_xyxy


def log_simple_detection(
    *,
    rgb_hw3: UInt8[ndarray, "h w 3"] | None = None,
    image_path: Path = DEFAULT_IMAGE_PATH,
    recording_id: uuid.UUID | str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    use_native_viewer: bool = True,
    capture_screenshot: bool = True,
    viewer_port: int = DEFAULT_VIEWER_PORT,
) -> SimpleNodeArtifacts:
    """Log the input image plus one centered box and return screenshot, RRDs, and serde JSON."""
    resolved_image: tuple[UInt8[ndarray, "h w 3"], str] = resolve_rgb_image(rgb_hw3=rgb_hw3, image_path=image_path)
    resolved_rgb_hw3: UInt8[ndarray, "h w 3"] = resolved_image[0]
    image_shape: tuple[int, int, int] = (
        int(resolved_rgb_hw3.shape[0]),
        int(resolved_rgb_hw3.shape[1]),
        int(resolved_rgb_hw3.shape[2]),
    )
    box_xyxy: Float32[ndarray, "1 4"] = centered_box_xyxy(image_shape)
    recording_id_str: str = ensure_recording_id(recording_id)
    rrd_path: Path = layer_rrd_path(output_dir, recording_id=recording_id_str, layer="detection")

    rec: rr.RecordingStream = rr.RecordingStream(application_id=DEFAULT_APPLICATION_ID, recording_id=recording_id_str)
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
            "image/center_box",
            rr.Boxes2D(array=box_xyxy, array_format=rr.Box2DFormat.XYXY, labels="center_box"),
        )

    rrd_bytes_or_none: bytes | None = binary_stream.read(flush=True, flush_timeout_sec=5.0)
    rrd_bytes: bytes = rrd_bytes_or_none if rrd_bytes_or_none is not None else b""
    screenshot_path: Path | None = None
    if use_native_viewer and capture_screenshot:
        screenshot_path = save_native_screenshot(output_dir, prefix="simple_detection", port=viewer_port)

    detection: SimpleDetectionJson = SimpleDetectionJson(
        application_id=DEFAULT_APPLICATION_ID,
        recording_id=recording_id_str,
        image_shape=list(image_shape),
        box_xyxy=[float(value) for value in box_xyxy[0].tolist()],
    )
    detection_json: str = to_json(detection)
    screenshot_path_str: str | None = str(screenshot_path) if screenshot_path is not None else None
    return screenshot_path_str, rrd_bytes, str(rrd_path), detection_json


def run_simple_detection(
    rgb_hw3: UInt8[ndarray, "h w 3"] | None,
    recording_id: uuid.UUID | str | None = None,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    use_native_viewer: bool = True,
    capture_screenshot: bool = True,
) -> Iterator[SimpleGradioArtifacts]:
    """Gradio event handler for the simple detection node."""
    screenshot_path: str | None
    rrd_bytes: bytes
    rrd_path: str
    detection_json: str
    screenshot_path, rrd_bytes, rrd_path, detection_json = log_simple_detection(
        rgb_hw3=rgb_hw3,
        recording_id=recording_id,
        output_dir=output_dir,
        use_native_viewer=use_native_viewer,
        capture_screenshot=capture_screenshot,
    )
    detection_value: dict[str, object] = cast(dict[str, object], json.loads(detection_json))
    outputs: SimpleGradioArtifacts = (screenshot_path, rrd_bytes, rrd_path, detection_value)
    yield outputs


def main() -> gr.Blocks:
    """Build the simple detection Gradio app."""
    viewer = Rerun(
        streaming=True,
        panel_states={
            "time": "collapsed",
            "blueprint": "hidden",
            "selection": "hidden",
        },
        height=800,
    )

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
                        create_rrd = gr.Button("Log Detection")

                    with gr.TabItem("Outputs", id="outputs"):
                        native_screenshot = gr.Image(label="Native Viewer Screenshot", type="filepath")
                        rrd_file = gr.File(label="Saved Detection RRD")
                        detection_json = gr.JSON(label="Detection JSON")

                gr.Examples(
                    examples=[[str(DEFAULT_IMAGE_PATH)]],
                    inputs=[rgb_hw3],
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
            run_simple_detection,
            inputs=[rgb_hw3],
            outputs=[native_screenshot, viewer, rrd_file, detection_json],
            api_name="simple_detection_fn",
        )

    return demo
