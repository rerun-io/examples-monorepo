"""The hands app's widgets, how a Run streams into them, and the server; the pipelines live in :mod:`posekit.hands_ui`."""

import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, get_args

import gradio as gr
from gradio_rerun import Rerun

from posekit.apis.video_hands import EXAMPLES, VideoHandsProgress, VideoHandsSettings
from posekit.hands_app_ui.theme import APP_CSS, DARK_HEAD, RERUN_LOGOS
from posekit.hands_ui import (
    PIPELINES,
    STACKS,
    default_stack,
    example_rows,
    fill_intrinsics,
    pipeline_for,
    settings_from_inputs,
    stream_hands,
)
from posekit.predictions import MeshDetail

DESCRIPTION: str = (
    f'<h1 id="app-title"><a href="https://rerun.io" target="_blank" rel="noopener" title="rerun.io">'
    f'<img class="logo-light" src="{RERUN_LOGOS["light"]}" alt="Rerun"><img class="logo-dark" src="{RERUN_LOGOS["dark"]}" alt="Rerun">'
    f'</a> <span class="title-x">×</span> EgoHand</h1>\n\n'
    "3D hands from a head-mounted video with NVIDIA's [EgoHand](https://github.com/NVIDIA/IsaacCapture/tree/main/src/python/isaaccapture_egopost), on a TensorRT fast stack "
    "(~7 ms per frame on an RTX 5090). **Pick an example or upload a clip, then press Run:** the hands stream into the viewer as they "
    f"are computed. The camera is assumed to be a {VideoHandsSettings().hfov_deg:g}° pinhole; set intrinsics under *Advanced settings*."
)
READY: str = "Upload a clip, or pick an example, then press Run."
RUN_LOG_LINES: int = 100
"""Lines the run log keeps: the textbox is sent whole after every chunk, so it holds the recent ones only."""


@dataclass(frozen=True, slots=True)
class Config:
    """Serve the hands app."""

    server_name: str = "127.0.0.1"
    """Interface on which Gradio listens."""
    server_port: int = 7872
    """Gradio HTTP port."""


def summary(progress: VideoHandsProgress, elapsed_s: float, stack: str, mesh_detail: MeshDetail) -> str:
    """The Outputs tab's table: how far the run is, how fast, and with what (the status box below names the camera)."""
    fps: str = f"{progress.frames_done / elapsed_s:.0f} fps" if progress.frames_done and elapsed_s > 0 else "-"
    return (
        "| Run | |\n|---|---|\n"
        f"| Frames | {progress.frames_done} / {progress.num_frames} |\n| Hands | {progress.hands} |\n"
        f"| Elapsed | {elapsed_s:.1f} s ({fps}) |\n| Stack | `{stack}`, mesh `{mesh_detail}` |"
    )


def run_hands(
    video: str | None, stack: str, max_frames: int, fx: float | None, fy: float | None, cx: float | None, cy: float | None,
    mesh_detail: MeshDetail,
) -> Iterator[tuple[Any, str, Any, str, str]]:
    """Stream a run into the viewer, the status box and the Outputs tab (progress bar, summary table, run log)."""
    if video is None:
        yield gr.update(), "Choose a video first.", gr.update(), "", ""
        return
    start: float = time.perf_counter()
    log: deque[str] = deque(maxlen=RUN_LOG_LINES)  # newest first
    if stack not in PIPELINES:
        setup: str = f"Preparing {stack} (the first run may take several minutes)…"
        log.appendleft(f"{0.0:5.1f} s  {setup}")
        yield gr.update(), setup, gr.update(), "", "\n".join(log)
        pipeline_for(stack)
        elapsed_s: float = time.perf_counter() - start
        log.appendleft(f"{elapsed_s:5.1f} s  {stack} ready ({elapsed_s:.0f} s)")
    settings: VideoHandsSettings = settings_from_inputs(video, stack, max_frames, fx, fy, cx, cy, mesh_detail)
    for recording, progress in stream_hands(settings, stack):
        elapsed_s = time.perf_counter() - start
        if progress.stage == "video" and progress.warning:
            log.appendleft(f"{elapsed_s:5.1f} s  {progress.warning}")
        log.appendleft(f"{elapsed_s:5.1f} s  {progress.step}")
        bar = gr.update(value=progress.frames_done, maximum=max(progress.num_frames, 1))
        yield recording, str(progress), bar, summary(progress, elapsed_s, stack, mesh_detail), "\n".join(log)


def build(
    stacks: Sequence[str] = STACKS, examples: list[list[str]] | None = None, stack_default: str | None = None,
    description: str = DESCRIPTION, run: Callable[..., Iterator[tuple[Any, str, Any, str, str]]] = run_hands,
) -> gr.Blocks:
    """Input and Outputs tabs, Run, Stop and the status on the left; the streaming Rerun viewer on the right.

    Args:
        stacks: Hand stacks the dropdown offers.
        examples: ``[video, stack]`` per example; None shows ``example_rows()``.
        stack_default: Stack selected at start; None selects ``default_stack()``.
        description: Markdown above the app.
        run: What the Run button calls, with ``run_hands``'s inputs and outputs (a host wraps ``run_hands`` to run it elsewhere).
    """
    with gr.Blocks(title="Rerun × EgoHand", fill_height=True) as demo:
        gr.Markdown(description, elem_id="app-description")
        with gr.Row(elem_id="main-row", equal_height=False, scale=1):
            with gr.Column(scale=1, min_width=360, elem_id="left-column"):
                with gr.Tabs(selected="input") as tabs:
                    with gr.Tab("Input", id="input"):
                        # format=None with audio kept: Gradio hands the upload over as it is (no transcode).
                        video = gr.Video(label="Source clip", sources=["upload"], format=None, include_audio=True, elem_id="source-video")
                        with gr.Accordion("Advanced settings", open=False):
                            stack = gr.Dropdown(label="Hand stack", choices=list(stacks), value=stack_default or default_stack())
                            max_frames = gr.Number(label="Max frames (0 = all)", value=0, precision=0)
                            mesh_detail = gr.Radio(
                                label="Mesh detail", choices=list(get_args(MeshDetail)), value=VideoHandsSettings().mesh_detail,
                                info="EgoHand's SOMA levels of detail per hand: full = 2,859 vertices, low = 718 (4x fewer bytes to the viewer), "
                                "xlo = 134 (21x fewer). The joints do not change.",
                            )
                            gr.Markdown(
                                f"Pinhole intrinsics in pixels, filled from a {VideoHandsSettings().hfov_deg:g}° horizontal FOV "
                                "with the principal point at the image centre."
                            )
                            with gr.Row():
                                fx = gr.Number(label="fx")
                                fy = gr.Number(label="fy")
                            with gr.Row():
                                cx = gr.Number(label="cx")
                                cy = gr.Number(label="cy")
                        # Picking an example also picks its stack; only the video column shows, as thumbnails.
                        gr.Examples(
                            examples=example_rows() if examples is None else examples, inputs=[video, stack], visible_columns=[0], examples_per_page=4,
                            cache_examples=False, elem_id="examples",
                        )
                    with gr.Tab("Outputs", id="outputs"):
                        progress_bar = gr.Slider(label="Frames processed", minimum=0, maximum=1, value=0, step=1, interactive=False)
                        run_summary = gr.Markdown("No run yet.", elem_id="run-summary")
                        run_log = gr.Textbox(label="Run log (recent, newest first)", lines=6, max_lines=6, autoscroll=False, interactive=False, elem_id="run-log")
                with gr.Row(elem_id="run-row"):
                    run_button = gr.Button("Run", variant="primary")
                    stop = gr.Button("Stop", variant="stop")
                status = gr.Markdown(READY, elem_id="run-status")
            with gr.Column(scale=3):
                viewer = Rerun(
                    streaming=True, panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"}, height="100%",
                    elem_id="viewer",
                )
        video.change(fill_intrinsics, inputs=[video], outputs=[fx, fy, cx, cy])
        run_event = run_button.click(lambda: gr.Tabs(selected="outputs"), outputs=[tabs]).then(
            run, inputs=[video, stack, max_frames, fx, fy, cx, cy, mesh_detail], outputs=[viewer, status, progress_bar, run_summary, run_log]
        )
        stop.click(lambda: "Stopped.", outputs=[status], cancels=[run_event], queue=False)
    return demo


def launch(demo: gr.Blocks, config: Config) -> None:
    """Serve ``demo``: one request at a time shares the GPU pipelines; forced dark like the 4DAnyone Space."""
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(
        server_name=config.server_name, server_port=config.server_port, ssr_mode=False, allowed_paths=[str(EXAMPLES)],
        css=APP_CSS, head=DARK_HEAD, show_error=True,
    )
