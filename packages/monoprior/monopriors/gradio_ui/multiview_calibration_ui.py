"""Gradio UI for multi-view calibration.

Provides an interactive web interface for running the multi-view calibration
pipeline with configurable parameters. The left panel holds image inputs,
a run button, and a config accordion; the right panel streams results into
an embedded Rerun viewer.

Models (VGGT, SAM3, MoGe) are loaded once at module import and reused
across calibration runs. Config changes that require new models (toggling
``segment_people`` or ``refine_depth_maps`` from OFF to ON) trigger a
lazy re-initialisation of the calibrator.
"""

import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Final

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

EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "examples" / "multiview"
"""Path to bundled example image sets used by ``gr.Examples``."""

gr.set_static_paths([str(EXAMPLE_DATA_DIR)])

_MV_CONFIG: MultiViewCalibratorConfig = MultiViewCalibratorConfig(
    device="cuda",
    verbose=True,
)
"""Module-level calibrator config, kept in sync with the UI widgets."""

_MV_CALIBRATOR: MultiViewCalibrator = MultiViewCalibrator(
    parent_log_path=PARENT_LOG_PATH,
    config=_MV_CONFIG,
)
"""Module-level calibrator singleton. Re-created only when model-affecting
config fields are toggled ON (see ``_sync_config``)."""


def _sync_config(
    keep_top_percent: int | float,
    refine_depth_maps: bool,
    segment_people: bool,
    preprocessing_mode: str,
) -> None:
    """Sync UI widget values into the module-level config and model singleton.

    Compares incoming widget values against the current ``_MV_CONFIG``.
    Runtime-only fields (``keep_top_percent``, ``preprocessing_mode``) are
    applied in-place. Model-affecting fields (``segment_people``,
    ``refine_depth_maps``) only trigger a full re-init when toggled from
    OFF to ON, since the required model was never loaded.

    Args:
        keep_top_percent: Confidence filtering threshold (1-100).
            Higher values discard more low-confidence pixels.
        refine_depth_maps: Whether to run MoGe depth refinement.
        segment_people: Whether to run SAM3 person segmentation.
        preprocessing_mode: Image preprocessing strategy ("crop" or "pad").
    """
    global _MV_CONFIG, _MV_CALIBRATOR

    needs_reinit: bool = False
    if segment_people and not _MV_CONFIG.segment_people:
        needs_reinit = True
    if refine_depth_maps and not _MV_CONFIG.refine_depth_maps:
        needs_reinit = True

    new_config: MultiViewCalibratorConfig = MultiViewCalibratorConfig(
        keep_top_percent=keep_top_percent,
        refine_depth_maps=refine_depth_maps,
        segment_people=segment_people,
        preprocessing_mode=preprocessing_mode,
        device="cuda",
        verbose=True,
    )

    if needs_reinit:
        _MV_CONFIG = new_config
        _MV_CALIBRATOR = MultiViewCalibrator(parent_log_path=PARENT_LOG_PATH, config=_MV_CONFIG)
    else:
        _MV_CONFIG = new_config
        _MV_CALIBRATOR.config = new_config
        _MV_CALIBRATOR.vggt_predictor.preprocessing_mode = preprocessing_mode


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
    rgb_list: list[UInt8[ndarray, "H W 3"]],
) -> Generator[tuple[bytes | None, str], None, None]:
    """Gradio streaming callback that runs the calibration pipeline.

    Delegates to ``run_calibration_pipeline`` inside a ``with recording:``
    context so all Rerun logging targets the UI's binary stream.

    Args:
        recording_id: Session-scoped recording identifier.
        rgb_list: Pre-loaded RGB images.

    Yields:
        Tuple of (Rerun binary stream bytes, status message string).
    """
    recording: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = recording.binary_stream()

    with recording:
        run_calibration_pipeline(
            rgb_list=rgb_list,
            mv_calibrator=_MV_CALIBRATOR,
            parent_log_path=PARENT_LOG_PATH,
            timeline=TIMELINE,
        )
    yield stream.read(), "Calibration complete"


def _switch_to_outputs():
    """Switch the Gradio Tabs component to the Outputs tab."""
    return gr.update(selected="outputs")


def main() -> gr.Blocks:
    """Build and return the multiview calibration Gradio app.

    Layout:
        - **Left column** (scale=1): Tabs with Inputs (file upload, run
          button, config accordion) and Outputs (status); example sets below.
        - **Right column** (scale=5): Embedded Rerun viewer.

    Click chain::

        click → _switch_to_outputs → new recording_id → _sync_config → _parse_and_load_images → multiview_calibration_fn

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
        rgb_list_state = gr.State([])

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
                            keep_top_percent_slider = gr.Slider(
                                label="Keep Top Percent (confidence filtering)",
                                minimum=1.0,
                                maximum=100.0,
                                step=1.0,
                                value=_MV_CONFIG.keep_top_percent,
                            )
                            refine_depth_checkbox = gr.Checkbox(
                                label="Refine Depth Maps (MoGe)",
                                value=_MV_CONFIG.refine_depth_maps,
                            )
                            segment_people_checkbox = gr.Checkbox(
                                label="Segment People (SAM3)",
                                value=_MV_CONFIG.segment_people,
                            )
                            preprocessing_radio = gr.Radio(
                                label="Preprocessing Mode",
                                choices=["crop", "pad"],
                                value=_MV_CONFIG.preprocessing_mode,
                            )

                    with gr.TabItem("Outputs", id="outputs"):
                        status_text = gr.Textbox(label="Status", interactive=False)

                car_example_images: list[str] = sorted(
                    str(p) for p in (EXAMPLE_DATA_DIR / "car_landscape_12").glob("*.jpg")
                )
                rp_capture_images: list[str] = sorted(
                    str(p) for p in (EXAMPLE_DATA_DIR / "rp_capture_6").glob("*.jpg")
                )
                gr.Examples(
                    examples=[
                        [car_example_images],
                        [rp_capture_images],
                    ],
                    inputs=[input_imgs],
                    cache_examples=False,
                )

            with gr.Column(scale=5):
                rr_viewer.render()

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
        ).then(  # Sync the calibrator singleton with the current UI config widgets
            _sync_config,
            inputs=[
                keep_top_percent_slider,
                refine_depth_checkbox,
                segment_people_checkbox,
                preprocessing_radio,
            ],
        ).then(  # Parse Gradio file uploads into RGB arrays
            _parse_and_load_images,
            inputs=[input_imgs],
            outputs=[rgb_list_state],
        ).then(  # Run calibration and stream results to the Rerun viewer
            multiview_calibration_fn,
            inputs=[recording_id, rgb_list_state],
            outputs=[rr_viewer, status_text],
        )

    return demo
