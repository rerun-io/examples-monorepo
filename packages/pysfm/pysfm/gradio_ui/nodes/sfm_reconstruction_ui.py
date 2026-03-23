"""Gradio UI for COLMAP SfM reconstruction.

Provides an interactive web interface for running Structure-from-Motion
reconstruction via pycolmap. The left panel holds inputs (directory upload),
a run button, and a config accordion; the right panel streams results into
an embedded Rerun viewer.
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
from jaxtyping import UInt8
from numpy import ndarray

from pysfm.apis.sfm_reconstruction import (
    TIMELINE,
    SfMCameraModel,
    SfMConfig,
    SfMResult,
    create_sfm_blueprint,
    run_sfm,
)

# ---------------------------------------------------------------------------
# Example data path
# ---------------------------------------------------------------------------
EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "examples" / "sfm_reconstruction"
"""Path to bundled example inputs.

Note: .parents[3] navigates from pysfm/gradio_ui/nodes/sfm_reconstruction_ui.py
up to the package root.
"""

gr.set_static_paths([str(EXAMPLE_DATA_DIR)])


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_CONFIG: SfMConfig = SfMConfig(verbose=True)
"""Module-level config, kept in sync with UI widgets."""


# ---------------------------------------------------------------------------
# _sync_config: widgets → config singleton
# ---------------------------------------------------------------------------
def _sync_config(
    camera_model: SfMCameraModel,
    random_seed: int,
) -> None:
    """Sync UI widget values into the module-level config.

    Args:
        camera_model: COLMAP camera model name.
        random_seed: Random seed for reproducibility.
    """
    global _CONFIG
    _CONFIG = SfMConfig(
        camera_model=camera_model,
        random_seed=int(random_seed),
        verbose=True,
    )


# ---------------------------------------------------------------------------
# _prepare_image_dir: Gradio directory upload → Path
# ---------------------------------------------------------------------------
def _prepare_image_dir(
    uploaded_files: str | list[str] | None,
) -> Path:
    """Extract the image directory from Gradio's directory upload.

    ``gr.File(file_count="directory")`` returns a list of file paths
    all residing in the same Gradio temp directory. We extract the
    common parent to get the directory path pycolmap needs.

    Args:
        uploaded_files: File path(s) from gr.File directory upload.

    Returns:
        Path to the directory containing the uploaded images.

    Raises:
        gr.Error: If no files were uploaded.
    """
    if uploaded_files is None:
        raise gr.Error("Please upload a directory of images before running.")

    if isinstance(uploaded_files, str):
        file_paths: list[Path] = [Path(uploaded_files)]
    elif isinstance(uploaded_files, list):
        file_paths = [Path(f) for f in uploaded_files]
    else:
        raise gr.Error("Invalid input. Please upload a directory of images.")

    if not file_paths:
        raise gr.Error("Please upload a directory of images before running.")

    # All files from a directory upload share the same parent
    image_dir: Path = file_paths[0].parent
    return image_dir


# ---------------------------------------------------------------------------
# Streaming callback
# ---------------------------------------------------------------------------
def sfm_reconstruction_fn(
    recording_id: uuid.UUID,
    image_dir: Path,
) -> Generator[tuple[bytes | None, str], None, None]:
    """Gradio streaming callback that runs SfM reconstruction.

    Creates a scoped Rerun recording, runs the pipeline inside it,
    and yields binary stream bytes to the Rerun viewer component.

    Args:
        recording_id: Session-scoped recording identifier.
        image_dir: Directory containing input images.

    Yields:
        Tuple of (Rerun binary stream bytes, status message string).
    """
    import cv2

    recording: rr.RecordingStream = rr.RecordingStream(
        application_id="sfm_reconstruction", recording_id=recording_id
    )
    stream: rr.BinaryStream = recording.binary_stream()

    parent_log_path: Path = Path("world")

    with recording:
        yield stream.read(), "Running SfM reconstruction..."

        # Run pipeline
        result: SfMResult = run_sfm(
            image_dir=image_dir,
            config=_CONFIG,
        )

        # Send blueprint
        blueprint: rrb.Blueprint = rrb.Blueprint(
            create_sfm_blueprint(parent_log_path=parent_log_path),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint=blueprint)
        rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

        # Log 3D point cloud (static — visible at all times)
        if result.points3d_xyz.shape[0] > 0:
            rr.log(
                f"{parent_log_path}/point_cloud",
                rr.Points3D(
                    positions=result.points3d_xyz,
                    colors=result.points3d_rgb,
                    radii=np.full(result.points3d_xyz.shape[0], 0.005, dtype=np.float32),
                ),
                static=True,
            )

        yield stream.read(), f"Logging {result.num_images_registered} cameras..."

        # Log cameras sequentially over the timeline
        camera_path: str = f"{parent_log_path}/camera"
        for i, img_result in enumerate(result.images):
            rr.set_time(TIMELINE, sequence=i)

            rr.log(
                camera_path,
                rr.Transform3D(
                    mat3x3=img_result.world_T_cam[:3, :3],
                    translation=img_result.world_T_cam[:3, 3],
                ),
            )

            width: int = img_result.image_size[0]
            height: int = img_result.image_size[1]
            rr.log(
                f"{camera_path}/pinhole",
                rr.Pinhole(
                    image_from_camera=img_result.intrinsics,
                    width=width,
                    height=height,
                ),
            )

            # Log image from the uploaded directory
            image_path: Path = image_dir / img_result.image_name
            if image_path.exists():
                bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rr.log(
                        f"{camera_path}/pinhole/image",
                        rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(),
                    )

            # Log 2D keypoints
            if img_result.keypoints_xy.shape[0] > 0:
                rr.log(
                    f"{camera_path}/pinhole/image/keypoints",
                    rr.Points2D(
                        positions=img_result.keypoints_xy,
                        radii=np.full(img_result.keypoints_xy.shape[0], 2.0, dtype=np.float32),
                        colors=np.full((img_result.keypoints_xy.shape[0], 3), [34, 138, 167], dtype=np.uint8),
                    ),
                )

            yield stream.read(), f"Logged camera {i + 1}/{result.num_images_registered}"

    yield stream.read(), f"SfM complete — {result.num_images_registered} images registered, {result.points3d_xyz.shape[0]} 3D points"


# ---------------------------------------------------------------------------
# Tab switching helpers
# ---------------------------------------------------------------------------
def _switch_to_outputs() -> dict:
    """Switch the Gradio Tabs component to the Outputs tab."""
    return gr.update(selected="outputs")


def _switch_to_inputs() -> dict:
    """Switch the Gradio Tabs component to the Inputs tab."""
    return gr.update(selected="inputs")


# ---------------------------------------------------------------------------
# main() → gr.Blocks
# ---------------------------------------------------------------------------
def main() -> gr.Blocks:
    """Build and return the SfM reconstruction Gradio app.

    Layout:
        - **Left column** (scale=1): Tabs with Inputs (directory upload, run
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
        # Session state
        recording_id = gr.State(uuid.uuid4())
        image_dir_state: gr.State = gr.State(None)

        with gr.Row():
            # ---- Left column: controls ----
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        input_files = gr.File(
                            label="Image Directory",
                            file_count="directory",
                        )
                        run_btn = gr.Button("Run SfM Reconstruction")

                        with gr.Accordion("Config", open=False):
                            camera_model_dropdown = gr.Dropdown(
                                label="Camera Model",
                                choices=["SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"],
                                value=_CONFIG.camera_model,
                            )
                            random_seed_number = gr.Number(
                                label="Random Seed",
                                value=_CONFIG.random_seed,
                                precision=0,
                            )

                    with gr.TabItem("Outputs", id="outputs"):
                        status_text = gr.Textbox(label="Status", interactive=False)

            # ---- Right column: Rerun viewer ----
            with gr.Column(scale=5):
                rr_viewer.render()

        # Switch to Inputs tab when files change
        input_files.change(
            fn=_switch_to_inputs,
            inputs=None,
            outputs=[tabs],
            api_visibility="private",
        )

        # Click chain: each .then() has ONE job
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
        ).then(
            _sync_config,
            inputs=[camera_model_dropdown, random_seed_number],
        ).then(
            _prepare_image_dir,
            inputs=[input_files],
            outputs=[image_dir_state],
        ).then(
            sfm_reconstruction_fn,
            inputs=[recording_id, image_dir_state],
            outputs=[rr_viewer, status_text],
        )

    return demo
