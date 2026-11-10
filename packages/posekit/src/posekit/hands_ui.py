"""Gradio app: 3D hands in a video (WiLoR or an EgoHand stack), streamed into an embedded Rerun viewer.

One input, a video. The app runs the CLI's code path: each Run builds the same
:class:`~posekit.apis.video_hands.VideoHandsSettings` the CLI parses from the widgets and drains
:func:`~posekit.apis.video_hands.run_video_hands` into a thread-local recording. The camera is a
pinhole whose intrinsics the Advanced accordion fills from an assumed field of view when a video
is chosen; edit them if you know the camera.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import gradio as gr
import rerun as rr
import torch
from gradio_rerun import Rerun

from posekit.apis.video_hands import EXAMPLES, VideoHandsSettings, assumed_pinhole, build_pipeline, run_video_hands, video_size
from posekit.models import HAND_DETECTORS, HAND_POSES, HandPose3dPipeline

STACKS: Final[list[str]] = list(HAND_DETECTORS)
"""Every named hand stack, as on the CLI."""
_PIPELINE: tuple[str, HandPose3dPipeline] | None = None
"""The last stack's set-up pipeline (engines stay loaded between runs); switching stacks releases it first."""


def default_stack() -> str:
    """The accepted EgoHand stack: NVFP4 on Blackwell, BF16 on older GPUs."""
    return "egohand-fast" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10 else "egohand-fast-bf16"


def example_rows() -> list[list[str]]:
    """``[video, stack]`` per example: egocentric clips on the accepted EgoHand stack, and WiLoR's original
    demo clip (a person on stage with a microphone, exocentric) on WiLoR (EgoHand is egocentric).

    The GSI clips are head-camera pinholes (1920x1080); the Gen-HumanEgo clips (1600x1300) were remapped from
    the double-sphere head camera into a 65° pinhole, so the app's assumed camera is their true one.
    """
    return [
        [str(EXAMPLES / "gsi_sewing_left_front.mp4"), default_stack()],
        [str(EXAMPLES / "genhumanego_kettle.mp4"), default_stack()],
        [str(EXAMPLES / "genhumanego_garlic.mp4"), default_stack()],
        [str(EXAMPLES / "gsi_workbench_left_front.mp4"), default_stack()],
        [str(EXAMPLES / "hands_video.mp4"), "wilor"],
    ]


def fill_intrinsics(video: str | None) -> tuple[float | None, float | None, float | None, float | None]:
    """``fx, fy, cx, cy`` of the assumed pinhole for the chosen video (the CLI's default for unset intrinsics), to 0.1 px."""
    if video is None:
        return None, None, None, None
    width, height = video_size(Path(video))
    fx, fy, cx, cy = assumed_pinhole(width, height, VideoHandsSettings().hfov_deg)
    return round(fx, 1), round(fy, 1), round(cx, 1), round(cy, 1)


def settings_from_inputs(
    video: str, stack: str, max_frames: int, fx: float | None, fy: float | None, cx: float | None, cy: float | None
) -> VideoHandsSettings:
    """The widget values as the CLI's settings; the stack name selects the same detector and pose configs as the CLI subcommand."""
    return VideoHandsSettings(
        video_path=Path(video), fx=fx, fy=fy, cx=cx, cy=cy, detector=HAND_DETECTORS[stack], pose=HAND_POSES[stack],
        max_frames=max_frames if max_frames > 0 else None,
    )


def _pipeline(stack: str, settings: VideoHandsSettings) -> HandPose3dPipeline:
    """The pipeline of ``stack``, set up once and reused while the stack stays selected."""
    global _PIPELINE
    if _PIPELINE is None or _PIPELINE[0] != stack:
        _PIPELINE = None
        torch.cuda.empty_cache()
        _PIPELINE = (stack, build_pipeline(settings))
    return _PIPELINE[1]


@rr.thread_local_stream("posekit_hands_ui")
def stream_hands(
    video: str | None, stack: str, max_frames: int, fx: float | None, fy: float | None, cx: float | None, cy: float | None
) -> Iterator[tuple[bytes | None, str]]:
    """Run the shared video hands core and stream the recording to the viewer after every chunk."""
    stream: rr.BinaryStream = rr.binary_stream()
    if video is None:
        yield stream.read(), "Choose a video first."
        return
    settings: VideoHandsSettings = settings_from_inputs(video, stack, max_frames, fx, fy, cx, cy)
    yield stream.read(), f"Setting up {stack} (the first run builds its TensorRT engines)..."
    pipeline: HandPose3dPipeline = _pipeline(stack, settings)
    for status in run_video_hands(settings, pipeline):
        yield stream.read(), status


def build() -> gr.Blocks:
    """A video input with an Advanced accordion beside a streaming Rerun viewer."""
    viewer = Rerun(streaming=True, panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"}, height=800)
    with gr.Blocks() as demo:
        gr.Markdown(
            "# 3D hands with Rerun\n"
            "posekit's hand stacks on a video: boxes and the projected 3D hands on the video, the camera and the 3D hands beside it. "
            "The camera is an assumed pinhole unless you edit it under Advanced. Same code as `posekit-video-hands`."
        )
        with gr.Row():
            with gr.Column(scale=1):
                tabs = gr.Tabs(selected="inputs")
                with tabs:
                    with gr.TabItem("Inputs", id="inputs"):
                        video = gr.File(label="Video", file_types=[".mp4"], type="filepath")
                        run = gr.Button("Run", variant="primary")
                        with gr.Accordion("Advanced", open=False):
                            stack = gr.Dropdown(label="Hand stack", choices=STACKS, value=default_stack())
                            max_frames = gr.Number(label="Max frames (0 = all)", value=0, precision=0)
                            gr.Markdown("Pinhole intrinsics in pixels, filled from a 65° horizontal FOV with the principal point at the image centre.")
                            with gr.Row():
                                fx = gr.Number(label="fx")
                                fy = gr.Number(label="fy")
                            with gr.Row():
                                cx = gr.Number(label="cx")
                                cy = gr.Number(label="cy")
                    with gr.TabItem("Outputs", id="outputs"):
                        status = gr.Text("Upload a video to begin.", label="Status")
                gr.Examples(examples=example_rows(), inputs=[video, stack])
            with gr.Column(scale=5):
                viewer.render()
        video.change(fill_intrinsics, inputs=[video], outputs=[fx, fy, cx, cy])
        run.click(fn=lambda: gr.update(selected="outputs"), inputs=None, outputs=[tabs]).then(
            stream_hands, inputs=[video, stack, max_frames, fx, fy, cx, cy], outputs=[viewer, status]
        )
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
