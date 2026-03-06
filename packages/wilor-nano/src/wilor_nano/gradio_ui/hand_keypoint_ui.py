"""Gradio UI for hand keypoint estimation.

This UI takes an image and detection results (as JSON from the detection UI)
and runs the WiLor keypoint detector to estimate 2D/3D hand keypoints.
"""

import json
import uuid
from pathlib import Path
from typing import Final, cast

import gradio as gr
import numpy as np
import rerun as rr
from gradio_rerun import Rerun
from jaxtyping import Float, UInt8
from numpy import ndarray
from serde.json import from_json
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_IDS, MEDIAPIPE_LINKS

from wilor_nano.hand_detection import DetectionResult
from wilor_nano.hand_keypoints import (
    HandKeypointDetectorConfig,
    KeypointResults,
    WilorHandKeypointDetector,
)

TEST_INPUT_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "assets"
assert TEST_INPUT_DIR.exists(), f"{TEST_INPUT_DIR} does not exist"

# Initialize keypoint detector
keypoint_detector = WilorHandKeypointDetector(cfg=HandKeypointDetectorConfig(verbose=True))


def _switch_to_outputs():
    return gr.update(selected="outputs")


def _ensure_uuid(recording_id: uuid.UUID | str | None) -> uuid.UUID:
    """Normalize provided recording id to a uuid.UUID instance."""
    if recording_id is None:
        return uuid.uuid4()
    if isinstance(recording_id, uuid.UUID):
        return recording_id
    return uuid.UUID(str(recording_id))


def get_recording(recording_id: uuid.UUID | str | None) -> rr.RecordingStream:
    normalized_id: uuid.UUID = _ensure_uuid(recording_id)
    return rr.RecordingStream(application_id="hand_keypoint_ui", recording_id=normalized_id)


def set_annotation_context() -> None:
    """Set the annotation context for hand keypoint visualization.

    Uses MediaPipe hand skeleton topology for automatic skeleton drawing
    via keypoint_connections in the AnnotationContext.
    """
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Hand", color=(0, 255, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
            ]
        ),
        static=True,
    )


def log_keypoint_results(
    keypoint_results: KeypointResults,
    handedness: str,
) -> None:
    """Log keypoint results to Rerun.

    Uses annotation context with keypoint_ids for automatic skeleton drawing.

    Args:
        keypoint_results: The keypoint estimation results.
        handedness: "left" or "right" to indicate which hand.
    """
    keypoints_2d: Float[ndarray, "batch=1 n_joints=21 2"] = keypoint_results.keypoints_2d

    # Log as 2D points with keypoint IDs for automatic skeleton drawing
    rr.log(
        f"image/{handedness}_hand_keypoints",
        rr.Points2D(
            positions=keypoints_2d,
            class_ids=0,
            keypoint_ids=MEDIAPIPE_IDS,
            show_labels=False,
            colors=(0, 255, 0),
        ),
    )


def pred_fn(
    rgb_hw3: UInt8[ndarray, "h w 3"] | None,
    detection_json: str | None,
    recording_id: uuid.UUID | str | None,
):
    """Run keypoint estimation on detected hands.

    Args:
        rgb_hw3: Input RGB image as numpy array.
        detection_json: JSON string or dict from the detection UI containing DetectionResult.
        recording_id: Unique ID for the Rerun recording.

    Yields:
        Tuple of (rerun binary stream bytes, keypoint results JSON).
    """
    if rgb_hw3 is None:
        raise gr.Error("No image provided")

    if detection_json is None:
        raise gr.Error("No detection results provided. Run hand detection first.")

    # Parse detection results - handle dict, JSON string, or Python repr string
    # (daggr passes gr.JSON output as a dict, Gradio Textbox may convert via str() to single-quoted repr)
    if isinstance(detection_json, dict):
        detection_json_str: str = json.dumps(detection_json)
    elif isinstance(detection_json, str):
        try:
            json.loads(detection_json)
            detection_json_str = detection_json
        except (json.JSONDecodeError, TypeError):
            import ast

            parsed: dict = ast.literal_eval(detection_json)
            detection_json_str = json.dumps(parsed)
    else:
        raise gr.Error(f"Unexpected detection_json type: {type(detection_json)}")
    detection_result: DetectionResult = from_json(DetectionResult, detection_json_str)

    rec: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    rec.set_time("iteration", sequence=0)

    # Set annotation context for skeleton visualization
    with rec:
        set_annotation_context()
        rec.log("image", rr.Image(image=rgb_hw3))

    keypoint_results_list: list[dict] = []

    with rec:
        # Process left hand if detected
        if detection_result.left_xyxy is not None:
            left_xyxy: Float[ndarray, "1 4"] = np.array(detection_result.left_xyxy)
            left_keypoints: KeypointResults = cast(
                KeypointResults,
                keypoint_detector(
                    rgb_hw3=rgb_hw3,
                    xyxy=left_xyxy,
                    handedness="left",
                ),
            )
            log_keypoint_results(left_keypoints, "left")
            keypoint_results_list.append(
                {
                    "handedness": "left",
                    "keypoints_2d": left_keypoints.keypoints_2d.tolist(),
                    "scores": left_keypoints.scores.tolist(),
                }
            )

            # Log detection box for reference
            rr.log(
                "image/left_hand_box",
                rr.Boxes2D(array=left_xyxy, array_format=rr.Box2DFormat.XYXY, labels="left_hand"),
            )

        # Process right hand if detected
        if detection_result.right_xyxy is not None:
            right_xyxy: Float[ndarray, "1 4"] = np.array(detection_result.right_xyxy)
            right_keypoints: KeypointResults = cast(
                KeypointResults,
                keypoint_detector(
                    rgb_hw3=rgb_hw3,
                    xyxy=right_xyxy,
                    handedness="right",
                ),
            )
            log_keypoint_results(right_keypoints, "right")
            keypoint_results_list.append(
                {
                    "handedness": "right",
                    "keypoints_2d": right_keypoints.keypoints_2d.tolist(),
                    "scores": right_keypoints.scores.tolist(),
                }
            )

            # Log detection box for reference
            rr.log(
                "image/right_hand_box",
                rr.Boxes2D(array=right_xyxy, array_format=rr.Box2DFormat.XYXY, labels="right_hand"),
            )

    # If no hands detected in detection results
    if not keypoint_results_list:
        keypoint_results_list.append({"message": "No hands detected in the provided detection results"})

    yield stream.read(), keypoint_results_list


def main() -> gr.Blocks:
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
        recording_id = gr.State(str(uuid.uuid4()))

        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        rgb_hw3 = gr.Image(
                            interactive=True,
                            label="Image",
                            type="numpy",
                            image_mode="RGB",
                        )
                        detection_json = gr.Textbox(
                            label="Detection JSON",
                            placeholder="Paste detection JSON from hand_detection_ui",
                            lines=5,
                        )
                        estimate_keypoints = gr.Button("Estimate Keypoints")

                        with gr.Accordion("Config", open=False):
                            gr.Markdown("Keypoint detector configuration (not yet configurable)")

                    with gr.TabItem("Outputs", id="outputs"):
                        keypoint_json_results = gr.JSON(label="Keypoint Results")

                gr.Examples(
                    examples=[
                        [
                            str(TEST_INPUT_DIR / "img.png"),
                            '{"left_xyxy": [[546.883544921875, 926.4755859375, 712.3211669921875, 1000.5693969726562]], "right_xyxy": [[157.1407470703125, 241.38446044921875, 330.0020751953125, 453.7044982910156]], "wholebody_xyxy": null}',
                        ],
                    ],
                    inputs=[rgb_hw3, detection_json],
                    outputs=[viewer, keypoint_json_results],
                    fn=pred_fn,
                    run_on_click=False,
                    cache_examples=False,
                    examples_per_page=2,
                )

            with gr.Column(scale=5):
                viewer.render()

        estimate_keypoints.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(
            pred_fn,
            inputs=[rgb_hw3, detection_json, recording_id],
            outputs=[viewer, keypoint_json_results],
        ).then(
            # update recording id
            fn=lambda: str(uuid.uuid4()),
            inputs=None,
            outputs=[recording_id],
            api_visibility="private",
        )

    return demo
