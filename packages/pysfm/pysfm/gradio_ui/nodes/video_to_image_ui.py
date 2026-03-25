"""Gradio UI for video-to-image frame extraction.

Provides an interactive web interface for extracting evenly-spaced frames
from a single video. The left panel holds a video input, run button, and
config accordion; the right panel streams the video and extracted images
into an embedded Rerun viewer.

Click chain::

    click -> _switch_to_outputs -> recording_id -> _sync_config
          -> _parse_video_path -> video_to_image_fn
"""

import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Final

import cv2
import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import Int, UInt8
from numpy import ndarray
from simplecv.rerun_log_utils import log_video

from pysfm.apis.video_to_image import (
    TIMELINE,
    VideoToImageConfig,
    VideoToImageResult,
    create_video_to_image_blueprint,
    run_video_to_image,
)

# ---------------------------------------------------------------------------
# Example data path
# ---------------------------------------------------------------------------
EXAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "examples" / "unknown-rig"
"""Path to bundled example videos.

Note: .parents[3] navigates from pysfm/gradio_ui/nodes/video_to_image_ui.py
up to the package root.
"""

gr.set_static_paths([str(EXAMPLE_DATA_DIR)])

PARENT_LOG_PATH: Final[Path] = Path("world")


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_CONFIG: VideoToImageConfig = VideoToImageConfig()
"""Module-level config, kept in sync with UI widgets."""


# ---------------------------------------------------------------------------
# _sync_config: widgets -> config singleton
# ---------------------------------------------------------------------------
def _sync_config(num_frames: int) -> None:
    """Sync UI widget values into the module-level config.

    Args:
        num_frames: Number of frames to extract.
    """
    global _CONFIG
    _CONFIG = VideoToImageConfig(num_frames=int(num_frames))


# ---------------------------------------------------------------------------
# _parse_video_path: Gradio video -> Path
# ---------------------------------------------------------------------------
def _parse_video_path(video_file: str | None) -> Path:
    """Convert Gradio video input to a Path.

    Args:
        video_file: File path string from ``gr.Video``.

    Returns:
        Path to the video file.

    Raises:
        gr.Error: If no video was provided.
    """
    if video_file is None:
        raise gr.Error("Please provide a video before running.")
    return Path(video_file)


# ---------------------------------------------------------------------------
# Streaming callback
# ---------------------------------------------------------------------------
def video_to_image_fn(
    recording_id: uuid.UUID,
    video_path: Path,
) -> Generator[tuple[bytes | None, str], None, None]:
    """Gradio streaming callback that extracts frames and visualizes in Rerun.

    Creates a scoped Rerun recording, runs extraction, logs the source
    video and each extracted frame on a synchronized timeline, and yields
    binary stream bytes to the Rerun viewer component.

    Args:
        recording_id: Session-scoped recording identifier.
        video_path: Path to the input video file.

    Yields:
        Tuple of (Rerun binary stream bytes, status message string).
    """
    recording: rr.RecordingStream = rr.RecordingStream(
        application_id="video_to_image", recording_id=recording_id
    )
    stream: rr.BinaryStream = recording.binary_stream()

    with recording:
        yield stream.read(), "Extracting frames..."

        # Run extraction to a temp directory
        tmp_dir: Path = Path(tempfile.mkdtemp(prefix="video_to_image_"))
        result: VideoToImageResult = run_video_to_image(
            video_path=video_path,
            output_dir=tmp_dir,
            config=_CONFIG,
        )

        yield stream.read(), f"Extracted {result.num_frames_extracted} frames. Logging to Rerun..."

        # Send blueprint
        blueprint: rrb.Blueprint = rrb.Blueprint(
            create_video_to_image_blueprint(PARENT_LOG_PATH),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint=blueprint)

        # Log the video asset
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
            video_source=video_path,
            video_log_path=PARENT_LOG_PATH / "video",
            timeline=TIMELINE,
        )

        yield stream.read(), "Logging extracted frames..."

        # Log each extracted image on the same timeline
        for out_idx, frame_idx in enumerate(result.frame_indices):
            rr.set_time(TIMELINE, duration=1e-9 * float(frame_timestamps_ns[frame_idx]))

            bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(result.image_paths[out_idx]), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rr.log(
                    f"{PARENT_LOG_PATH}/extracted/image",
                    rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(),
                )

            yield stream.read(), f"Logged frame {out_idx + 1}/{result.num_frames_extracted}"

    yield stream.read(), f"Done -- extracted {result.num_frames_extracted} frames to {result.output_dir}"


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
# main() -> gr.Blocks
# ---------------------------------------------------------------------------
def main() -> gr.Blocks:
    """Build and return the video-to-image Gradio app.

    Layout:
        - **Left column** (scale=1): Tabs with Inputs (video upload, run
          button, config accordion) and Outputs (status); example videos below.
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
        video_path_state: gr.State = gr.State(None)

        with gr.Row():
            # ---- Left column: controls ----
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        input_video = gr.Video(label="Input Video")
                        run_btn = gr.Button("Extract Frames")

                        with gr.Accordion("Config", open=False):
                            num_frames_slider = gr.Slider(
                                label="Number of Frames",
                                minimum=2,
                                maximum=200,
                                step=1,
                                value=_CONFIG.num_frames,
                            )

                    with gr.TabItem("Outputs", id="outputs"):
                        status_text = gr.Textbox(label="Status", interactive=False)

                # Example videos
                example_videos: list[list[str]] = [
                    [str(EXAMPLE_DATA_DIR / "cam1.mp4")],
                ]
                gr.Examples(
                    examples=example_videos,
                    inputs=[input_video],
                    cache_examples=False,
                )

            # ---- Right column: Rerun viewer ----
            with gr.Column(scale=5):
                rr_viewer.render()

        # Switch to Inputs tab when video changes
        input_video.change(
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
            inputs=[num_frames_slider],
        ).then(
            _parse_video_path,
            inputs=[input_video],
            outputs=[video_path_state],
        ).then(
            video_to_image_fn,
            inputs=[recording_id, video_path_state],
            outputs=[rr_viewer, status_text],
        )

    return demo
