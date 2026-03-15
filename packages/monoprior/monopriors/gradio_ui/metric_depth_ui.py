"""Gradio UI for metric depth estimation.

Provides an interactive web interface for running any ``BaseMetricPredictor``
on a single image. The left panel holds image input, a run button, and a
config accordion with predictor selection; the right panel streams results
into an embedded Rerun viewer.

The metric depth model is loaded once at module import and reused across runs.
Changing the predictor selection triggers a lazy re-initialisation.

Click chain::

    click → _switch_to_outputs → recording_id → _sync_config
          → _run_prediction → _log_results
"""

import gc
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Final, get_args

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import UInt8
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import log_pinhole

from monopriors.apis.metric_depth import MetricDepthNodeConfig, create_metric_predictor, run_metric_depth
from monopriors.models.metric_depth import METRIC_PREDICTORS, BaseMetricPredictor, MetricDepthPrediction

PARENT_LOG_PATH: Final[Path] = Path("world")
EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "examples" / "multiview"

gr.set_static_paths([str(EXAMPLE_DATA_DIR)])

_CONFIG: MetricDepthNodeConfig = MetricDepthNodeConfig(device="cuda")
"""Module-level config, kept in sync with UI widgets."""

_PREDICTOR: BaseMetricPredictor = create_metric_predictor(_CONFIG)
"""Module-level predictor singleton. Re-created when predictor_name changes."""


def _sync_config(predictor_name: str) -> None:
    """Sync UI widget values into the module-level config and predictor singleton.

    Args:
        predictor_name: Which metric predictor to use.
    """
    global _CONFIG, _PREDICTOR
    import torch

    needs_reinit: bool = predictor_name != _CONFIG.predictor_name

    _CONFIG = MetricDepthNodeConfig(
        predictor_name=predictor_name,  # type: ignore[arg-type]  # Gradio dropdown returns str
        device="cuda",
    )

    if needs_reinit:
        del _PREDICTOR
        gc.collect()
        torch.cuda.empty_cache()
        _PREDICTOR = create_metric_predictor(_CONFIG)


def _get_recording(recording_id: uuid.UUID) -> rr.RecordingStream:
    """Create a Rerun recording stream for a given session."""
    return rr.RecordingStream(application_id="metric_depth", recording_id=recording_id)


def _run_prediction(
    img: UInt8[ndarray, "H W 3"] | None,
) -> MetricDepthPrediction:
    """Run metric depth prediction on a single image.

    Pure prediction step — no Rerun logging. Result is stored in ``gr.State``
    and passed to ``_log_results`` for visualization.

    Args:
        img: Input RGB image from ``gr.Image``.

    Returns:
        MetricDepthPrediction with depth_meters, confidence, and K_33.
    """
    if img is None:
        raise gr.Error("Please provide an image.")

    result: MetricDepthPrediction = run_metric_depth(
        rgb=img,
        predictor=_PREDICTOR,
    )
    return result


def _log_results(
    recording_id: uuid.UUID,
    img: UInt8[ndarray, "H W 3"],
    result: MetricDepthPrediction,
) -> Generator[tuple[bytes | None, str], None, None]:
    """Post-processing: log prediction results to Rerun viewer.

    Builds pinhole from estimated K_33, logs image/depth/confidence,
    and streams everything to the Rerun viewer.

    Args:
        recording_id: Session-scoped recording identifier.
        img: Original RGB image (for coloring).
        result: MetricDepthPrediction from ``_run_prediction``.

    Yields:
        Tuple of (Rerun binary stream bytes, status message string).
    """
    recording: rr.RecordingStream = _get_recording(recording_id)
    stream: rr.BinaryStream = recording.binary_stream()

    with recording:
        # Blueprint
        blueprint: rrb.Blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=f"{PARENT_LOG_PATH}",
                    contents=["+ $origin/**", f"- {PARENT_LOG_PATH}/camera/pinhole/depth"],
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(origin=f"{PARENT_LOG_PATH}/camera/pinhole/image"),
                    rrb.Spatial2DView(origin=f"{PARENT_LOG_PATH}/camera/pinhole/depth"),
                    rrb.Spatial2DView(origin=f"{PARENT_LOG_PATH}/camera/pinhole/confidence"),
                ),
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        rr.log(f"{PARENT_LOG_PATH}", rr.ViewCoordinates.RDF, static=True)

        # Build pinhole from estimated K_33
        h: int = img.shape[0]
        w: int = img.shape[1]
        intrinsics: Intrinsics = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(result.K_33[0, 0]),
            fl_y=float(result.K_33[1, 1]),
            cx=float(result.K_33[0, 2]),
            cy=float(result.K_33[1, 2]),
            width=w,
            height=h,
        )
        extrinsics: Extrinsics = Extrinsics(
            world_R_cam=np.eye(3, dtype=np.float32),
            world_t_cam=np.zeros(3, dtype=np.float32),
        )
        pinhole: PinholeParameters = PinholeParameters(
            name="camera_0",
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        cam_log_path: Path = PARENT_LOG_PATH / "camera"
        log_pinhole(pinhole, cam_log_path=cam_log_path, image_plane_distance=0.05, static=True)

        # Log image, depth, confidence
        rr.log(f"{cam_log_path}/pinhole/image", rr.Image(img, color_model=rr.ColorModel.RGB).compress(), static=True)
        rr.log(f"{cam_log_path}/pinhole/depth", rr.DepthImage(result.depth_meters, meter=1), static=True)
        rr.log(f"{cam_log_path}/pinhole/confidence", rr.Image(result.confidence, color_model=rr.ColorModel.L), static=True)

    yield stream.read(), "Metric depth complete"


def _switch_to_outputs():
    """Switch the Gradio Tabs component to the Outputs tab."""
    return gr.update(selected="outputs")


def _switch_to_inputs():
    """Switch the Gradio Tabs component to the Inputs tab."""
    return gr.update(selected="inputs")


def main() -> gr.Blocks:
    """Build and return the metric depth Gradio app.

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
        prediction_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        input_img = gr.Image(
                            label="Input Image",
                            type="numpy",
                            image_mode="RGB",
                        )
                        run_btn = gr.Button("Run Metric Depth")

                        with gr.Accordion("Config", open=False):
                            predictor_dropdown = gr.Dropdown(
                                label="Predictor",
                                choices=list(get_args(METRIC_PREDICTORS)),
                                value=_CONFIG.predictor_name,
                            )

                    with gr.TabItem("Outputs", id="outputs"):
                        status_text = gr.Textbox(label="Status", interactive=False)

                example_images: list[list[str]] = sorted(
                    [[str(p)] for p in (EXAMPLE_DATA_DIR / "car_landscape_12").glob("*.jpg")][:4]
                )
                gr.Examples(
                    examples=example_images,
                    inputs=[input_img],
                    cache_examples=False,
                )

            with gr.Column(scale=5):
                rr_viewer.render()

        # Switch to Inputs tab when examples populate the input
        input_img.change(fn=_switch_to_inputs, inputs=None, outputs=[tabs], api_visibility="private")

        # Click chain: UI transition → fresh session → sync config → predict → log results
        run_btn.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(
            fn=lambda: uuid.uuid4(),
            inputs=None,
            outputs=[recording_id],
            api_visibility="private",
        ).then(  # Sync the predictor singleton with the current UI config widgets
            _sync_config,
            inputs=[predictor_dropdown],
        ).then(  # Run prediction (pure — no Rerun logging)
            _run_prediction,
            inputs=[input_img],
            outputs=[prediction_state],
        ).then(  # Post-process: log results to Rerun viewer
            _log_results,
            inputs=[recording_id, input_img, prediction_state],
            outputs=[rr_viewer, status_text],
        )

    return demo
