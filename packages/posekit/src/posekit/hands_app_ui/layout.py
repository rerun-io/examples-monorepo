"""The app's widgets and how they connect to :mod:`posekit.hands_ui`."""

import time
from collections.abc import Iterator
from typing import Any, get_args

import gradio as gr
from gradio_rerun import Rerun

from posekit.apis.video_hands import EXAMPLES, VideoHandsProgress, VideoHandsSettings
from posekit.hands_app_ui.theme import APP_CSS, VIEWER_HEIGHT
from posekit.hands_ui import STACKS, Config, default_stack, example_rows, example_stack, fill_intrinsics, stream_hands
from posekit.predictions import MeshDetail
from posekit.track_ui_theme import FORCE_DARK_HEAD

DESCRIPTION: str = (
    "# EgoHand × [Rerun](https://rerun.io)\n"
    "A video of hands becomes 3D hands. A hand detector finds every hand, a 3D hand model reconstructs its joints and mesh "
    "in the camera, and each chunk streams into the Rerun viewer as it happens. Same code as `posekit-video-hands`."
)
READY: str = "Upload a clip, or pick an example, then press Run."


def summary(progress: VideoHandsProgress, elapsed_s: float, stack: str, mesh_detail: MeshDetail) -> str:
    """The Outputs tab's table: how far the run is, how fast, and with what."""
    fps: str = f"{progress.frames_done / elapsed_s:.0f} fps" if progress.frames_done and elapsed_s > 0 else "-"
    return (
        "| Run | |\n|---|---|\n"
        f"| Frames | {progress.frames_done} / {progress.num_frames} |\n| Hands | {progress.hands} |\n"
        f"| Elapsed | {elapsed_s:.1f} s ({fps}) |\n| Stack | `{stack}`, mesh `{mesh_detail}` |\n| Camera | {progress.camera_note} |"
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
    log: list[str] = []
    for recording, progress in stream_hands(video, stack, max_frames, fx, fy, cx, cy, mesh_detail):
        elapsed_s: float = time.perf_counter() - start
        if isinstance(progress, str):  # a stack setting up
            log.append(f"{elapsed_s:5.1f} s  {progress}")
            yield recording, progress, gr.update(), "", "\n".join(log)
            continue
        step: str = {"video": "video loaded", "hands": f"{progress.frames_done}/{progress.num_frames} frames", "done": "done"}[progress.stage]
        log.append(f"{elapsed_s:5.1f} s  {step}, {progress.hands} hands")
        bar = gr.update(value=progress.frames_done, maximum=max(progress.num_frames, 1))
        yield recording, str(progress), bar, summary(progress, elapsed_s, stack, mesh_detail), "\n".join(log)


def build() -> gr.Blocks:
    """Input and Outputs tabs, Run, Stop and the status on the left; the streaming Rerun viewer on the right."""
    with gr.Blocks(title="EgoHand × Rerun") as demo:
        gr.Markdown(DESCRIPTION, elem_id="app-description")
        with gr.Row(elem_id="main-row", equal_height=False):
            with gr.Column(scale=1, min_width=360, elem_id="left-column"):
                with gr.Tabs(selected="input") as tabs:
                    with gr.Tab("Input", id="input"):
                        # format=None with audio kept: Gradio hands the upload over as it is (no transcode).
                        video = gr.Video(label="Source clip", sources=["upload"], format=None, include_audio=True, elem_id="source-video")
                        with gr.Accordion("Advanced settings", open=False):
                            stack = gr.Dropdown(label="Hand stack", choices=STACKS, value=default_stack())
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
                        gr.Examples(
                            examples=[video_path for video_path, _ in example_rows()], inputs=[video], examples_per_page=4,
                            cache_examples=False, elem_id="examples",
                        )
                    with gr.Tab("Outputs", id="outputs"):
                        progress_bar = gr.Slider(label="Frames processed", minimum=0, maximum=1, value=0, step=1, interactive=False)
                        run_summary = gr.Markdown("No run yet.", elem_id="run-summary")
                        run_log = gr.Textbox(label="Run log", lines=6, max_lines=6, autoscroll=True, interactive=False, elem_id="run-log")
                with gr.Row(elem_id="run-row"):
                    run = gr.Button("Run", variant="primary")
                    stop = gr.Button("Stop", variant="stop")
                status = gr.Markdown(READY, elem_id="run-status")
            with gr.Column(scale=3):
                viewer = Rerun(
                    streaming=True, panel_states={"time": "collapsed", "blueprint": "hidden", "selection": "hidden"}, height=VIEWER_HEIGHT
                )
        video.change(fill_intrinsics, inputs=[video], outputs=[fx, fy, cx, cy])
        video.change(example_stack, inputs=[video, stack], outputs=[stack])
        run_event = run.click(lambda: gr.Tabs(selected="outputs"), outputs=[tabs]).then(
            run_hands, inputs=[video, stack, max_frames, fx, fy, cx, cy, mesh_detail], outputs=[viewer, status, progress_bar, run_summary, run_log]
        )
        stop.click(lambda: "Stopped.", outputs=[status], cancels=[run_event], queue=False)
    return demo


def launch(demo: gr.Blocks, config: Config) -> None:
    """Serve ``demo``: one request at a time shares the GPU pipelines; forced dark like the 4DAnyone Space."""
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(
        server_name=config.server_name, server_port=config.server_port, ssr_mode=False, allowed_paths=[str(EXAMPLES)],
        css=APP_CSS, head=FORCE_DARK_HEAD, show_error=True,
    )
