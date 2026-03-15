"""Gradio UI for multi-view geometry prediction.

Provides an interactive web interface for running a multi-view geometry
predictor (currently VGGT) on multiple images to produce oriented camera
poses, depth maps, and confidence masks. The left panel holds image inputs,
a run button, and a config accordion; the right panel streams results into
an embedded Rerun viewer.

The model is loaded once at module import and reused across runs.
Config changes that affect the model (``preprocessing_mode``) trigger
lazy re-initialisation of the predictor.
"""

import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Final

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import log_pinhole

from monopriors.apis.multiview_calibration import (
    PARENT_LOG_PATH,
    TIMELINE,
    load_rgb_images,
)
from monopriors.apis.multiview_geometry import (
    MultiviewGeometryConfig,
    MultiviewGeometryResult,
    run_multiview_geometry,
)
from monopriors.models.multiview.vggt_model import VGGTPredictor

EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "examples" / "multiview"


def create_multiview_blueprint(parent_log_path: Path, num_images: int) -> rrb.ContainerLike:
    """Create a Rerun blueprint for VGGT geometry (no MoGe depth tab).

    3D view on the left showing oriented cameras + point cloud.
    Per-camera tabs on the right with Depth, Filtered Depth, and Confidence.
    """
    from monopriors.apis.multiview_calibration import chunk_cameras

    camera_chunks: list[range] = chunk_cameras(num_images)
    tabs: list[rrb.Vertical] = []
    for camera_range in camera_chunks:
        camera_rows: list[rrb.Horizontal] = []
        for i in camera_range:
            row: rrb.Horizontal = rrb.Horizontal(
                contents=[
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/image",
                        name="Image",
                    ),
                    rrb.Tabs(
                        contents=[
                            rrb.Spatial2DView(
                                origin=f"{parent_log_path}/camera_{i}/pinhole/depth",
                                name="Depth",
                            ),
                            rrb.Spatial2DView(
                                origin=f"{parent_log_path}/camera_{i}/pinhole/filtered_depth",
                                name="Filtered Depth",
                            ),
                        ],
                        active_tab=1,
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/confidence",
                        name="Confidence",
                    ),
                ]
            )
            camera_rows.append(row)
        if camera_range.start + 1 == camera_range.stop:
            tab_name: str = f"Camera {camera_range.start + 1}"
        else:
            tab_name = f"Cameras {camera_range.start + 1}-{camera_range.stop}"
        tabs.append(rrb.Vertical(contents=camera_rows, name=tab_name))

    view_3d: rrb.Spatial3DView = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=[
            "+ $origin/**",
            # Exclude raw depth and confidence from 3D (noisy). Keep filtered_depth — Rerun
            # auto-unprojects DepthImage under a pinhole into a 3D point cloud.
            *[f"- /{parent_log_path}/camera_{i}/pinhole/depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/camera_{i}/pinhole/confidence" for i in range(num_images)],
        ],
        line_grid=rrb.archetypes.LineGrid3D(visible=False),
    )
    view_2d: rrb.Tabs = rrb.Tabs(contents=tabs, name="Depths Tab")
    return rrb.Horizontal(contents=[view_3d, view_2d], column_shares=[3, 2])
"""Path to bundled example image sets."""

gr.set_static_paths([str(EXAMPLE_DATA_DIR)])

_CONFIG: MultiviewGeometryConfig = MultiviewGeometryConfig(
    device="cuda",
    verbose=True,
)
"""Module-level config, kept in sync with UI widgets."""

_PREDICTOR: VGGTPredictor = VGGTPredictor(
    device=_CONFIG.device,
    preprocessing_mode=_CONFIG.preprocessing_mode,
)
"""Module-level VGGT singleton. Re-created only when preprocessing_mode changes."""


def _sync_config(
    keep_top_percent: int | float,
    preprocessing_mode: str,
    verbose: bool,
) -> None:
    """Sync UI widget values into the module-level config and predictor singleton.

    Args:
        keep_top_percent: Confidence filtering threshold (1-100).
        preprocessing_mode: Image preprocessing strategy ('crop' or 'pad').
        verbose: Whether to log per-camera detail.
    """
    global _CONFIG, _PREDICTOR

    needs_reinit: bool = preprocessing_mode != _CONFIG.preprocessing_mode

    _CONFIG = MultiviewGeometryConfig(
        keep_top_percent=keep_top_percent,
        preprocessing_mode=preprocessing_mode,
        device="cuda",
        verbose=verbose,
    )

    if needs_reinit:
        _PREDICTOR = VGGTPredictor(
            device=_CONFIG.device,
            preprocessing_mode=_CONFIG.preprocessing_mode,
        )
    else:
        _PREDICTOR.preprocessing_mode = preprocessing_mode


def _get_recording(recording_id: uuid.UUID) -> rr.RecordingStream:
    """Create a Rerun recording stream for a given session."""
    return rr.RecordingStream(application_id="multiview_geometry", recording_id=recording_id)


def _parse_and_load_images(
    img_files: str | list[str],
) -> list[UInt8[ndarray, "H W 3"]]:
    """Parse Gradio file uploads and load them as RGB arrays.

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
        raise gr.Error("Please select at least one RGB image before running.")

    img_paths.sort()
    rgb_list: list[UInt8[ndarray, "H W 3"]] = load_rgb_images(img_paths)
    return rgb_list


def multiview_geometry_fn(
    recording_id: uuid.UUID,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
) -> Generator[tuple[bytes | None, str], None, None]:
    """Gradio streaming callback that runs VGGT geometry prediction.

    Args:
        recording_id: Session-scoped recording identifier.
        rgb_list: Pre-loaded RGB images.

    Yields:
        Tuple of (Rerun binary stream bytes, status message string).
    """
    recording: rr.RecordingStream = _get_recording(recording_id)
    stream: rr.BinaryStream = recording.binary_stream()

    with recording:
        # Setup blueprint (VGGT-specific: no MoGe depth tab)
        final_view: rrb.ContainerLike = create_multiview_blueprint(
            parent_log_path=PARENT_LOG_PATH, num_images=len(rgb_list)
        )
        blueprint: rrb.Blueprint = rrb.Blueprint(final_view, collapse_panels=True)
        rr.send_blueprint(blueprint=blueprint)
        rr.log(f"{PARENT_LOG_PATH}", rr.ViewCoordinates.RFU, static=True)
        rr.set_time(TIMELINE, duration=0)

        # Run VGGT geometry
        result: MultiviewGeometryResult = run_multiview_geometry(
            rgb_list=rgb_list,
            vggt_predictor=_PREDICTOR,
            config=_CONFIG,
        )

        # Log per-camera results: pinhole + filtered depth (auto-unprojected to 3D by Rerun)
        for mv_pred, depth_conf in zip(result.mv_pred_list, result.depth_confidences, strict=True):
            cam_log_path: Path = PARENT_LOG_PATH / mv_pred.cam_name
            pinhole_log_path: Path = cam_log_path / "pinhole"
            log_pinhole(
                mv_pred.pinhole_param,
                cam_log_path=cam_log_path,
                image_plane_distance=0.05,
                static=True,
            )
            # Filtered depth under the pinhole — Rerun auto-unprojects this into 3D
            filtered_depth_map: Float32[ndarray, "H W"] = np.where(depth_conf > 0, mv_pred.depth_map, 0)
            rr.log(f"{pinhole_log_path}/filtered_depth", rr.DepthImage(filtered_depth_map, meter=1), static=True)

            if _CONFIG.verbose:
                rr.log(
                    f"{pinhole_log_path}/image",
                    rr.Image(mv_pred.rgb_image, color_model=rr.ColorModel.RGB).compress(),
                    static=True,
                )
                rr.log(
                    f"{pinhole_log_path}/confidence",
                    rr.Image(depth_conf, color_model=rr.ColorModel.L).compress(),
                    static=True,
                )
                rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(mv_pred.depth_map, meter=1), static=True)

    yield stream.read(), "Multiview geometry complete"


def _switch_to_outputs():
    """Switch the Gradio Tabs component to the Outputs tab."""
    return gr.update(selected="outputs")


def _switch_to_inputs():
    """Switch the Gradio Tabs component to the Inputs tab."""
    return gr.update(selected="inputs")


def main() -> gr.Blocks:
    """Build and return the VGGT geometry Gradio app.

    Layout:
        - **Left column** (scale=1): Tabs with Inputs (file upload, run
          button, config accordion) and Outputs (status); example sets below.
        - **Right column** (scale=5): Embedded Rerun viewer.

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
                        run_btn = gr.Button("Run VGGT Geometry")

                        with gr.Accordion("Config", open=False):
                            keep_top_percent_slider = gr.Slider(
                                label="Keep Top Percent (confidence filtering)",
                                minimum=1.0,
                                maximum=100.0,
                                step=1.0,
                                value=_CONFIG.keep_top_percent,
                            )
                            preprocessing_radio = gr.Radio(
                                label="Preprocessing Mode",
                                choices=["crop", "pad"],
                                value=_CONFIG.preprocessing_mode,
                            )
                            verbose_checkbox = gr.Checkbox(
                                label="Verbose (per-camera detail logging)",
                                value=_CONFIG.verbose,
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

        # Switch to Inputs tab when examples populate the input
        input_imgs.change(fn=_switch_to_inputs, inputs=None, outputs=[tabs], api_visibility="private")

        # Click chain: UI transition → fresh session → sync config → load images → run pipeline
        run_btn.click(
            fn=_switch_to_outputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        ).then(  # Generate a fresh recording ID so each run gets its own Rerun session
            fn=lambda: uuid.uuid4(),
            inputs=None,
            outputs=[recording_id],
            api_visibility="private",
        ).then(  # Sync the predictor singleton with the current UI config widgets
            _sync_config,
            inputs=[
                keep_top_percent_slider,
                preprocessing_radio,
                verbose_checkbox,
            ],
        ).then(  # Parse Gradio file uploads into RGB arrays
            _parse_and_load_images,
            inputs=[input_imgs],
            outputs=[rgb_list_state],
        ).then(  # Run VGGT geometry and stream results to the Rerun viewer
            multiview_geometry_fn,
            inputs=[recording_id, rgb_list_state],
            outputs=[rr_viewer, status_text],
        )

    return demo
