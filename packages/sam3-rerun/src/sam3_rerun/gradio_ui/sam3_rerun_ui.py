"""Gradio UI for SAM3 text-conditioned segmentation.

Provides an interactive web interface for running SAM3 on a single image
with configurable text prompt, mask threshold, and dilation. The left panel
holds image input, a run button, and a config accordion; the right panel
streams results into an embedded Rerun viewer.

The SAM3 model is loaded once at module import and reused across runs.
"""

import uuid
from pathlib import Path
from typing import Final

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import spaces
from gradio_rerun import Rerun
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray

from sam3_rerun.api.predictor import SAM3Config, SAM3Predictor, SAM3Results
from sam3_rerun.viz_constants import BOX_PALETTE, SEG_CLASS_OFFSET, SEG_OVERLAY_ALPHA

CFG: SAM3Config = SAM3Config()
MODEL_E2E: SAM3Predictor = SAM3Predictor(config=CFG)
DONE_STATUS: Final[str] = "Ready"
RUNNING_STATUS: Final[str] = "Running prediction..."

# Default config values for the UI
_TEXT_PROMPT: str = "person"
_MASK_THRESHOLD: float = 0.5
_DILATION: int = 0


def _sync_config(text_prompt: str, mask_threshold: float, dilation: int) -> None:
    """Sync UI widget values into module-level config state.

    Args:
        text_prompt: Text prompt for SAM3 segmentation.
        mask_threshold: Probability threshold to binarize masks.
        dilation: Kernel size for mask dilation (0 = no dilation).
    """
    global _TEXT_PROMPT, _MASK_THRESHOLD, _DILATION
    _TEXT_PROMPT = text_prompt
    _MASK_THRESHOLD = mask_threshold
    _DILATION = int(dilation)
# Absolute path to bundled example data used by Gradio examples.
TEST_INPUT_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "example-data"

# Allow Gradio to serve and cache files from the bundled test data directory.
gr.set_static_paths([str(TEST_INPUT_DIR)])


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


@spaces.GPU()
def sam3d_prediction_fn(
    img: UInt8[ndarray, "h w 3"] | None, text_prompt: str | None, recording_id: uuid.UUID | str | None
):
    if text_prompt is None:
        raise gr.Error("Must provide a text prompt.")
    if img is None:
        raise gr.Error("Must provide an image.")

    recording: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = recording.binary_stream()

    with recording:
        blueprint = rrb.Blueprint(
            rrb.Spatial2DView(
                name="Image + Segmentation",
                contents=[
                    "image",
                    "image/segmentation_ids",
                ],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        rr.set_time("iteration", sequence=0)

        sam3_results: SAM3Results = MODEL_E2E.predict_single_image(rgb_hw3=img, text=text_prompt)

        rr.log("image", rr.Image(img))

        h: int = int(img.shape[0])
        w: int = int(img.shape[1])
        seg_map: UInt8[np.ndarray, "h w"] = np.full((h, w), SEG_CLASS_OFFSET, dtype=np.uint8)

        # Build a single segmentation image where each instance gets a unique id.
        for idx, segmask in enumerate(sam3_results.masks):
            mask: Float32[np.ndarray, "h w"] = np.asarray(segmask, dtype=np.float32).squeeze()
            mask_bool: Bool[np.ndarray, "h w"] = mask >= _MASK_THRESHOLD
            class_id: int = SEG_CLASS_OFFSET + idx + 1  # reserve SEG_CLASS_OFFSET for background
            seg_map = np.where(mask_bool, np.uint8(class_id), seg_map)

        class_descriptions: list[rr.ClassDescription] = [
            rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET, label="Background", color=(64, 64, 64, 0)))
        ]
        for idx, color_rgb in enumerate(BOX_PALETTE[:, :3].tolist(), start=1):
            color_rgba: tuple[int, int, int, int] = (
                int(color_rgb[0]),  # type: ignore[arg-type]  # numpy .tolist() lacks type stubs
                int(color_rgb[1]),  # type: ignore[arg-type]
                int(color_rgb[2]),  # type: ignore[arg-type]
                SEG_OVERLAY_ALPHA,
            )
            class_descriptions.append(
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET + idx, label=f"Mask-{idx}", color=color_rgba)
                )
            )

        rr.log("/", rr.AnnotationContext(class_descriptions), static=True)
        rr.log("image/segmentation_ids", rr.SegmentationImage(seg_map))

    yield stream.read(), DONE_STATUS


def _switch_to_outputs() -> gr.Tabs:
    return gr.update(selected="outputs")


def _switch_to_inputs() -> gr.Tabs:
    return gr.update(selected="inputs")


def main():
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
                        img = gr.Image(interactive=True, label="Image", type="numpy", image_mode="RGB")
                        run_btn = gr.Button("Run Segmentation")

                        with gr.Accordion("Config", open=False):
                            text_prompt = gr.Textbox(
                                label="Text Prompt",
                                value=_TEXT_PROMPT,
                            )
                            mask_threshold_slider = gr.Slider(
                                label="Mask Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=_MASK_THRESHOLD,
                            )
                            dilation_slider = gr.Slider(
                                label="Dilation",
                                minimum=0,
                                maximum=100,
                                step=5,
                                value=_DILATION,
                            )

                    with gr.TabItem("Outputs", id="outputs"):
                        status = gr.Text(DONE_STATUS, label="Status")

                gr.Examples(
                    examples=[
                        [str(TEST_INPUT_DIR / "Planche.jpg")],
                        [str(TEST_INPUT_DIR / "Amir-Khan-Lamont-Peterson_2689582.jpg")],
                        [str(TEST_INPUT_DIR / "BNAAHPYGMYSE26U6C6T7VA6544.jpg")],
                        [str(TEST_INPUT_DIR / "yoga-example.jpg")],
                    ],
                    inputs=[img],
                    cache_examples=False,
                    examples_per_page=2,
                )
            with gr.Column(scale=5):
                viewer.render()

        # Switch to Inputs tab when examples populate the input
        img.change(fn=_switch_to_inputs, inputs=None, outputs=[tabs], api_visibility="private")

        # Click chain: UI transition → fresh session → sync config → run prediction
        run_btn.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(  # Sync config widgets into module-level state
            _sync_config,
            inputs=[text_prompt, mask_threshold_slider, dilation_slider],
        ).then(  # Run SAM3 prediction and stream results to the Rerun viewer
            sam3d_prediction_fn,
            inputs=[img, text_prompt, recording_id],
            outputs=[viewer, status],
        )
    return demo
