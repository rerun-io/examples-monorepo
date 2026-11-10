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
from jaxtyping import UInt8
from torch import Tensor

from posekit.apis.video_hands import EXAMPLES, VideoHandsSettings, assumed_pinhole, build_pipeline, pinhole_camera, run_video_hands, video_size
from posekit.models import HAND_DETECTORS, HAND_POSES, HandPose3dPipeline

STACKS: Final[list[str]] = list(HAND_DETECTORS)
"""Every named hand stack, as on the CLI."""
VIEWER_ID: Final[str] = "hands-viewer"
"""Element id of the Rerun viewer, so the page script can find its WebViewer."""
# TODO(https://github.com/rerun-io/gradio-rerun-viewer/pull/66): once released, replace the page script and
# REWIND_JS with `.then(lambda: set_time("video_time", duration=0.0), outputs=[viewer])` and delete hands_ui_rewind.js.
REWIND_SCRIPT: Final[Path] = Path(__file__).with_name("hands_ui_rewind.js")
"""Temporary page script that rewinds the viewer after a run (gradio_rerun 0.42 has no time command)."""
REWIND_JS: Final[str] = f"() => window.__rerunRewind('{VIEWER_ID}', 'video_time')"
"""Browser step after a run: once the stream stops growing, pause the viewer at the start of ``video_time``."""
_PIPELINES: dict[str, HandPose3dPipeline] = {}
"""Set-up pipelines by stack name: engines stay loaded for the app's lifetime, so switching stacks costs nothing."""


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


def pipeline_for(stack: str) -> HandPose3dPipeline:
    """The pipeline of ``stack``, set up (TensorRT engines built or loaded) on first use and kept."""
    if stack not in _PIPELINES:
        _PIPELINES[stack] = build_pipeline(VideoHandsSettings(detector=HAND_DETECTORS[stack], pose=HAND_POSES[stack]))
    return _PIPELINES[stack]


def warm_up() -> None:
    """Set up every example's stack and run it once on the example's first chunk.

    Setup builds or loads the TensorRT engines; the first call compiles and captures CUDA graphs
    for that frame size (~4 s for ``egohand-fast``). Done at start, the examples run at full speed
    from the first click; an upload of a new frame size pays well under a second once.
    """
    from torchcodec.decoders import VideoDecoder

    for video, stack in example_rows():
        settings: VideoHandsSettings = settings_from_inputs(video, stack, 0, None, None, None, None)
        decoder = VideoDecoder(settings.video_path, dimension_order="NHWC", device="cuda")
        width, height = video_size(settings.video_path)
        frames: UInt8[Tensor, "chunk h w 3"] = decoder.get_frames_in_range(0, settings.frame_batch_size).data.contiguous()
        print(f"[posekit] warming up {stack} on {settings.video_path.name}", flush=True)
        with torch.inference_mode():
            pipeline_for(stack)(frames, pinhole_camera(settings, width, height))


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
    if stack not in _PIPELINES:
        yield stream.read(), f"Setting up {stack} (its first use builds or loads its TensorRT engines)..."
    pipeline: HandPose3dPipeline = pipeline_for(stack)
    for status in run_video_hands(settings, pipeline):
        yield stream.read(), status


def build() -> gr.Blocks:
    """A video input with an Advanced accordion beside a streaming Rerun viewer."""
    viewer = Rerun(streaming=True, panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"}, height=800, elem_id=VIEWER_ID)
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
        ).then(fn=None, js=REWIND_JS)
    return demo


@dataclass(frozen=True, slots=True)
class Config:
    """Serve the hands app."""

    server_name: str = "127.0.0.1"
    """Interface on which Gradio listens."""
    server_port: int = 7872
    """Gradio HTTP port."""


def page_head() -> str:
    """The page script that lets ``REWIND_JS`` reach the viewer; it must load before the viewer mounts."""
    return f"<script>{REWIND_SCRIPT.read_text()}</script>"


def launch(config: Config) -> None:
    """Warm up the examples' stacks, then serve the app; one request at a time shares the GPU pipelines."""
    warm_up()
    demo = build()
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(server_name=config.server_name, server_port=config.server_port, ssr_mode=False, allowed_paths=[str(EXAMPLES)], head=page_head())
