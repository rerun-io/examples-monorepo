import uuid
from pathlib import Path
from typing import Final

import gradio as gr
import rerun as rr
from gradio_rerun import Rerun
from jaxtyping import UInt8
from numpy import ndarray
from serde.json import to_json

from wilor_nano.hand_detection import DetectionResult, HandDetector, HandDetectorConfig

TEST_INPUT_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "assets"
assert TEST_INPUT_DIR.exists(), f"{TEST_INPUT_DIR} does not exist"

# Initialize hand detector
hand_detector = HandDetector(cfg=HandDetectorConfig())


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
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=normalized_id)


def log_detection_results(result: DetectionResult) -> None:
    if result.left_xyxy is not None:
        rr.log(
            "image/left_hand", rr.Boxes2D(array=result.left_xyxy, array_format=rr.Box2DFormat.XYXY, labels="left_hand")
        )
    if result.right_xyxy is not None:
        rr.log(
            "image/right_hand",
            rr.Boxes2D(array=result.right_xyxy, array_format=rr.Box2DFormat.XYXY, labels="right_hand"),
        )
    if result.wholebody_xyxy is not None:
        rr.log(
            "image/wholebody",
            rr.Boxes2D(array=result.wholebody_xyxy, array_format=rr.Box2DFormat.XYXY, labels="wholebody"),
        )


def pred_fn(rgb_hw3: UInt8[ndarray, "h w 3"] | None, recording_id: uuid.UUID | str | None):
    # Here we get a recording using the provided recording id.
    if rgb_hw3 is None:
        raise gr.Error("No image provided")
    rec: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = rec.binary_stream()
    rec.set_time("iteration", sequence=0)
    rec.log("image", rr.Image(image=rgb_hw3))
    result: DetectionResult = hand_detector(rgb_hw3=rgb_hw3, hand_conf=0.2)

    with rec:
        log_detection_results(result)

    results_json: str = to_json(result)
    # Return image as third output for daggr passthrough
    yield stream.read(), results_json, rgb_hw3


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
                        rgb_hw3 = gr.Image(interactive=True, label="Image", type="numpy", image_mode="RGB")
                        create_rrd = gr.Button("Predict Pose")
                        with gr.Accordion("Config", open=False):
                            ...

                    with gr.TabItem("Outputs", id="outputs"):
                        det_json_results = gr.JSON()
                        # Hidden image output for daggr passthrough
                        image_passthrough = gr.Image(visible=False)

                gr.Examples(
                    examples=[
                        [str(TEST_INPUT_DIR / "img.png")],
                    ],
                    inputs=[rgb_hw3],
                    outputs=[viewer, det_json_results, image_passthrough],
                    fn=pred_fn,
                    run_on_click=False,
                    cache_examples=False,
                    examples_per_page=2,
                )
            with gr.Column(scale=5):
                viewer.render()

        create_rrd.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(
            pred_fn,
            inputs=[rgb_hw3, recording_id],
            outputs=[viewer, det_json_results, image_passthrough],
        ).then(
            # update recording id
            fn=lambda: str(uuid.uuid4()),
            inputs=None,
            outputs=[recording_id],
            api_visibility="private",
        )

    return demo
