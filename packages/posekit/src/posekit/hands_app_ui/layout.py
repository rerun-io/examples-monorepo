"""The app's widgets and how they connect to :mod:`posekit.hands_ui`."""

import gradio as gr
from gradio_rerun import Rerun

from posekit.apis.video_hands import EXAMPLES, VideoHandsSettings
from posekit.hands_app_ui.theme import APP_CSS, VIEWER_HEIGHT
from posekit.hands_ui import STACKS, Config, default_stack, example_rows, example_stack, fill_intrinsics, stream_hands
from posekit.track_ui_theme import FORCE_DARK_HEAD

DESCRIPTION: str = (
    "# EgoHand × [Rerun](https://rerun.io)\n"
    "A video of hands becomes 3D hands. A hand detector finds every hand, a 3D hand model reconstructs its joints and mesh "
    "in the camera, and each chunk streams into the Rerun viewer as it happens. Same code as `posekit-video-hands`."
)
READY: str = "Upload a clip, or pick an example, then press Run."


def build() -> gr.Blocks:
    """Inputs, examples, Run and Stop on the left; the streaming Rerun viewer on the right."""
    with gr.Blocks(title="EgoHand × Rerun") as demo:
        gr.Markdown(DESCRIPTION, elem_id="app-description")
        with gr.Row(elem_id="main-row", equal_height=False):
            with gr.Column(scale=1, min_width=360):
                # format=None with audio kept: Gradio hands the upload over as it is (no transcode).
                video = gr.Video(label="Source clip", sources=["upload"], format=None, include_audio=True, elem_id="source-video")
                with gr.Accordion("Advanced settings", open=False):
                    stack = gr.Dropdown(label="Hand stack", choices=STACKS, value=default_stack())
                    max_frames = gr.Number(label="Max frames (0 = all)", value=0, precision=0)
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
        run_event = run.click(stream_hands, inputs=[video, stack, max_frames, fx, fy, cx, cy], outputs=[viewer, status])
        stop.click(lambda: "Stopped.", outputs=[status], cancels=[run_event], queue=False)
    return demo


def launch(demo: gr.Blocks, config: Config) -> None:
    """Serve ``demo``: one request at a time shares the GPU pipelines; forced dark like the 4DAnyone Space."""
    demo.queue(max_size=1, default_concurrency_limit=1)
    demo.launch(
        server_name=config.server_name, server_port=config.server_port, ssr_mode=False, allowed_paths=[str(EXAMPLES)],
        css=APP_CSS, head=FORCE_DARK_HEAD, show_error=True,
    )
