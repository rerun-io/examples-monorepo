"""Gradio UI for multi-view calibration.

Provides an interactive web interface for running the multi-view calibration
pipeline with configurable parameters. The left panel holds image inputs,
a run button, and a config accordion; the right panel streams results into
an embedded Rerun viewer.

Each run captures one immutable request. Models load lazily: the selected multi-view
backend uses an atomic single-resident cache, while SAM3 and MoGe are cached independently.
"""

import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import gradio as gr
import rerun as rr
from gradio_rerun import Rerun
from jaxtyping import UInt8
from numpy import ndarray

from monopriors.apis.multiview_calibration import (
    PARENT_LOG_PATH,
    TIMELINE,
    MultiViewCalibrator,
    MultiViewCalibratorConfig,
    load_rgb_images,
    run_calibration_pipeline,
)
from monopriors.gradio_ui._calibration_runtime import AUXILIARY_MODEL_CACHE, CalibrationAuxiliaryModels
from monopriors.gradio_ui._multiview_common import parse_multiview_model, parse_preprocessing_mode
from monopriors.gradio_ui._multiview_runtime import PREDICTOR_CACHE
from monopriors.models.multiview.multiview_predictor import MultiviewPredictorConfig

EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "examples" / "multiview"
"""Path to bundled example image sets used by ``gr.Examples``."""

gr.set_static_paths([str(EXAMPLE_DATA_DIR)])

DEFAULT_CALIBRATOR_CONFIG: Final[MultiViewCalibratorConfig] = MultiViewCalibratorConfig(
    predictor_config=MultiviewPredictorConfig(device="cuda", preprocessing_mode="pad"),
    verbose=True,
)


@dataclass(frozen=True, slots=True)
class MultiviewCalibrationRequest:
    """All inputs captured atomically when a user starts one calibration run."""

    rgb_list: list[UInt8[ndarray, "H W 3"]]
    config: MultiViewCalibratorConfig


def _prepare_request(
    img_files: str | list[str],
    model_name: str,
    keep_top_percent: int | float,
    refine_depth_maps: bool,
    segment_people: bool,
    preprocessing_mode: str,
) -> MultiviewCalibrationRequest:
    """Capture uploaded images and widget values into one immutable run request.

    Args:
        img_files: Uploaded image paths.
        model_name: Multi-view model backend (``vggt`` or ``g3t``).
        keep_top_percent: Confidence filtering threshold (1-100).
            Higher values discard more low-confidence pixels.
        refine_depth_maps: Whether to run MoGe depth refinement.
        segment_people: Whether to run SAM3 person segmentation.
        preprocessing_mode: Image preprocessing strategy ("crop" or "pad").
    """
    preprocessing_mode_literal: Literal["crop", "pad"] = parse_preprocessing_mode(preprocessing_mode)
    return MultiviewCalibrationRequest(
        rgb_list=_parse_and_load_images(img_files),
        config=MultiViewCalibratorConfig(
            predictor_config=MultiviewPredictorConfig(
                model_name=parse_multiview_model(model_name),
                device="cuda",
                preprocessing_mode=preprocessing_mode_literal,
            ),
            keep_top_percent=keep_top_percent,
            refine_depth_maps=refine_depth_maps,
            segment_people=segment_people,
            verbose=True,
        ),
    )


def get_recording(recording_id: uuid.UUID) -> rr.RecordingStream:
    """Create a Rerun recording stream for a given session.

    As long as the application and recording IDs remain the same, data
    will be merged by the Rerun viewer.

    Args:
        recording_id: Unique session identifier from Gradio state.

    Returns:
        A new ``RecordingStream`` bound to the session.
    """
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=recording_id)


def _parse_and_load_images(
    img_files: str | list[str],
) -> list[UInt8[ndarray, "H W 3"]]:
    """Parse Gradio file uploads and load them as RGB arrays.

    Converts ``gr.File`` output (single path or list of paths) into
    sorted RGB numpy arrays using the shared ``load_rgb_images`` loader.

    Args:
        img_files: Single path or list of paths from ``gr.File``.

    Returns:
        Sorted list of RGB images as uint8 numpy arrays.
    """
    if isinstance(img_files, str):
        img_paths: list[Path] = [Path(img_files)]
    elif isinstance(img_files, list):
        img_paths = [Path(f) for f in img_files]
    else:
        raise gr.Error("Invalid input for images. Please select image files.")

    if not img_paths:
        raise gr.Error("Please select at least one RGB image before running calibration.")

    img_paths.sort()
    rgb_list: list[UInt8[ndarray, "H W 3"]] = load_rgb_images(img_paths)
    return rgb_list


def multiview_calibration_fn(
    recording_id: uuid.UUID,
    request: MultiviewCalibrationRequest,
) -> Generator[tuple[bytes | None, str], None, None]:
    """Gradio streaming callback that runs the calibration pipeline.

    Delegates to ``run_calibration_pipeline`` inside a ``with recording:``
    context so all Rerun logging targets the UI's binary stream.

    Args:
        recording_id: Session-scoped recording identifier.
        request: Images and exact model/operation configuration for this run.

    Yields:
        Tuple of (Rerun binary stream bytes, status message string).
    """
    recording: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = recording.binary_stream()

    with recording, PREDICTOR_CACHE.acquire(request.config.predictor_config) as multiview_predictor:
        auxiliary_models: CalibrationAuxiliaryModels = AUXILIARY_MODEL_CACHE.get(
            device=request.config.predictor_config.device,
            segment_people=request.config.segment_people,
            refine_depth_maps=request.config.refine_depth_maps,
        )
        calibrator: MultiViewCalibrator = MultiViewCalibrator(
            parent_log_path=PARENT_LOG_PATH,
            config=request.config,
            multiview_predictor=multiview_predictor,
            seg_predictor=auxiliary_models.seg_predictor,
            moge_predictor=auxiliary_models.moge_predictor,
        )
        run_calibration_pipeline(
            rgb_list=request.rgb_list,
            mv_calibrator=calibrator,
            parent_log_path=PARENT_LOG_PATH,
            timeline=TIMELINE,
        )
    yield stream.read(), "Calibration complete"


def _switch_to_outputs():
    """Switch the Gradio Tabs component to the Outputs tab."""
    return gr.update(selected="outputs")


def _switch_to_inputs():
    """Switch the Gradio Tabs component to the Inputs tab."""
    return gr.update(selected="inputs")


def main() -> gr.Blocks:
    """Build and return the multiview calibration Gradio app.

    Layout:
        - **Left column** (scale=1): Tabs with Inputs (file upload, run
          button, config accordion) and Outputs (status); example sets below.
        - **Right column** (scale=5): Embedded Rerun viewer.

    Click chain::

        click → _switch_to_outputs → new recording_id → _prepare_request → multiview_calibration_fn

    Returns:
        The assembled ``gr.Blocks`` instance ready for ``.queue().launch()``.
    """
    rr_viewer = Rerun(
        streaming=True,
        panel_states={
            "time": "collapsed",
            "blueprint": "collapsed",
            "selection": "collapsed",
        },
        height=800,
    )

    with gr.Blocks() as demo:
        recording_id = gr.State(uuid.uuid4())
        request_state = gr.State()

        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        input_imgs = gr.File(
                            label="Input Images",
                            file_count="multiple",
                            file_types=[".png", ".jpg", ".jpeg"],
                        )
                        run_calibration_btn = gr.Button("Run Multi-view Calibration")

                        with gr.Accordion("Config", open=False):
                            model_dropdown = gr.Dropdown(
                                label="Multi-view Model",
                                choices=["vggt", "g3t"],
                                value=DEFAULT_CALIBRATOR_CONFIG.predictor_config.model_name,
                            )
                            keep_top_percent_slider = gr.Slider(
                                label="Keep Top Percent (confidence filtering)",
                                minimum=1.0,
                                maximum=100.0,
                                step=1.0,
                                value=DEFAULT_CALIBRATOR_CONFIG.keep_top_percent,
                            )
                            refine_depth_checkbox = gr.Checkbox(
                                label="Refine Depth Maps (MoGe)",
                                value=DEFAULT_CALIBRATOR_CONFIG.refine_depth_maps,
                            )
                            segment_people_checkbox = gr.Checkbox(
                                label="Segment People (SAM3)",
                                value=DEFAULT_CALIBRATOR_CONFIG.segment_people,
                            )
                            preprocessing_radio = gr.Radio(
                                label="Preprocessing Mode",
                                choices=["crop", "pad"],
                                value=DEFAULT_CALIBRATOR_CONFIG.predictor_config.preprocessing_mode,
                            )

                    with gr.TabItem("Outputs", id="outputs"):
                        status_text = gr.Textbox(label="Status", interactive=False)

                car_example_images: list[str] = sorted(
                    str(p) for p in (EXAMPLE_DATA_DIR / "car_landscape_12").glob("*.jpg")
                )
                gr.Examples(
                    examples=[
                        [car_example_images],
                    ],
                    inputs=[input_imgs],
                    cache_examples=False,
                )

            with gr.Column(scale=5):
                rr_viewer.render()

        # Switch to Inputs tab when examples populate the input
        input_imgs.change(fn=_switch_to_inputs, inputs=None, outputs=[tabs], api_visibility="private")

        # Click chain: UI transition → fresh session → sync config → load images → run pipeline
        run_calibration_btn.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(  # Generate a fresh recording ID so each run gets its own Rerun session
            fn=lambda: uuid.uuid4(),
            inputs=None,
            outputs=[recording_id],
            api_visibility="private",
        ).then(  # Atomically capture files and all widget values for this run
            _prepare_request,
            inputs=[
                input_imgs,
                model_dropdown,
                keep_top_percent_slider,
                refine_depth_checkbox,
                segment_people_checkbox,
                preprocessing_radio,
            ],
            outputs=[request_state],
        ).then(  # Run calibration and stream results to the Rerun viewer
            multiview_calibration_fn,
            inputs=[recording_id, request_state],
            outputs=[rr_viewer, status_text],
        )

    return demo
