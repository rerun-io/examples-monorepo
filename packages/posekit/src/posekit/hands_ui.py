"""Gradio app: 3D hands in a video (WiLoR or an EgoHand stack), streamed into an embedded Rerun viewer.

The app runs the CLI's code path. A module-level ``_CONFIG`` of the same
:class:`~posekit.apis.video_hands.VideoHandsSettings` the CLI parses is kept in sync with
the widgets by ``_sync_config`` (chained before Run), and the streaming callback drains
:func:`~posekit.apis.video_hands.run_video_hands` into a thread-local recording.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import gradio as gr
import rerun as rr
import torch
from gradio_rerun import Rerun

from posekit.apis.video_hands import EXAMPLES, VideoHandsSettings, build_pipeline, run_video_hands
from posekit.models import HAND_DETECTORS, HAND_POSES, HandPose3dPipeline

STACKS: Final[list[str]] = list(HAND_DETECTORS)
"""Every named hand stack, as on the CLI."""

_CONFIG: VideoHandsSettings = VideoHandsSettings()
"""Module-level settings, kept in sync with the UI widgets by ``_sync_config``."""
_PIPELINE: tuple[str, HandPose3dPipeline] | None = None
"""The last stack's set-up pipeline; switching stacks releases it before the next one loads."""


def default_stack() -> str:
    """The accepted EgoHand stack: NVFP4 on Blackwell, BF16 on older GPUs."""
    return "egohand-fast" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10 else "egohand-fast-bf16"


def _sync_config(video: str | None, camera: str | None, stack: str, max_frames: int) -> None:
    """Copy the widget values into ``_CONFIG``; the stack name selects the same detector and pose configs as the CLI."""
    global _CONFIG
    _CONFIG = replace(
        _CONFIG,
        video_path=_CONFIG.video_path if video is None else Path(video),
        camera_path=None if camera is None else Path(camera),
        detector=HAND_DETECTORS[stack],
        pose=HAND_POSES[stack],
        max_frames=max_frames if max_frames > 0 else None,
    )


def _pipeline(stack: str) -> HandPose3dPipeline:
    """The pipeline of ``stack``, set up once and reused while the stack stays selected."""
    global _PIPELINE
    if _PIPELINE is None or _PIPELINE[0] != stack:
        _PIPELINE = None
        torch.cuda.empty_cache()
        _PIPELINE = (stack, build_pipeline(_CONFIG))
    return _PIPELINE[1]


@rr.thread_local_stream("posekit_hands_ui")
def stream_hands(stack: str) -> Iterator[tuple[bytes | None, str]]:
    """Run the shared video hands core on ``_CONFIG`` and stream the recording to the viewer after every chunk."""
    stream: rr.BinaryStream = rr.binary_stream()
    yield stream.read(), f"Setting up {stack} (the first run builds its TensorRT engines)..."
    pipeline: HandPose3dPipeline = _pipeline(stack)
    for status in run_video_hands(_CONFIG, pipeline):
        yield stream.read(), status


def build() -> gr.Blocks:
    """Inputs and a stack accordion beside a streaming Rerun viewer."""
    viewer = Rerun(streaming=True, panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"}, height=800)
    with gr.Blocks() as demo:
        gr.Markdown(
            "# 3D hands with Rerun\n"
            "posekit's hand stacks on a video: WiLoR on any video, EgoHand with the video's CalibratedCamera .rrd "
            "(metric hands in the real camera frame). Same code as `posekit-video-hands`."
        )
        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        video = gr.File(label="Video", file_types=[".mp4"], type="filepath")
                        camera = gr.File(label="CalibratedCamera .rrd (EgoHand)", file_types=[".rrd"], type="filepath")
                        run = gr.Button("Run", variant="primary")
                        with gr.Accordion("Config", open=False):
                            stack = gr.Dropdown(label="Hand stack", choices=STACKS, value=default_stack())
                            max_frames = gr.Number(label="Max frames (0 = all)", value=0, precision=0)
                    with gr.TabItem("Outputs", id="outputs"):
                        status = gr.Text("Upload a video to begin.", label="Status")
                gr.Examples(
                    examples=[[str(EXAMPLES / "selfcap_oak_left.mp4"), str(EXAMPLES / "selfcap_oak_left.rrd")]],
                    inputs=[video, camera],
                )
            with gr.Column(scale=5):
                viewer.render()
        run.click(fn=lambda: gr.update(selected="outputs"), inputs=None, outputs=[tabs]).then(
            _sync_config, inputs=[video, camera, stack, max_frames], outputs=None
        ).then(stream_hands, inputs=[stack], outputs=[viewer, status])
    return demo


@dataclass(frozen=True, slots=True)
class Config:
    """Serve the hands app."""

    server_name: str = "127.0.0.1"
    """Interface on which Gradio listens."""
    server_port: int = 7872
    """Gradio HTTP port."""


def launch(config: Config) -> None:
    """Serve the app; one request at a time shares the GPU pipeline."""
    demo = build()
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(server_name=config.server_name, server_port=config.server_port, ssr_mode=False, allowed_paths=[str(EXAMPLES)])
