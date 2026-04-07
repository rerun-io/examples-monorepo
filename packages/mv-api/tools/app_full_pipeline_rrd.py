import os

import gradio as gr

from mv_api.gradio_ui.full_pipeline_rrd_ui import EXAMPLE_RRD_PATH, build_full_pipeline_rrd_block

title = "# Full Exo/Ego Pipeline (RRD Input)"
GRADIO_SERVER_NAME: str = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT: int = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_ROOT_PATH: str | None = os.environ.get("GRADIO_ROOT_PATH") or None

with gr.Blocks() as demo:
    gr.Markdown(title)
    build_full_pipeline_rrd_block()


if __name__ == "__main__":
    launch_kwargs: dict[str, object] = {
        "server_name": GRADIO_SERVER_NAME,
        "server_port": GRADIO_SERVER_PORT,
    }
    if EXAMPLE_RRD_PATH.exists():
        launch_kwargs["allowed_paths"] = [str(EXAMPLE_RRD_PATH.parent)]
    if GRADIO_ROOT_PATH:
        launch_kwargs["root_path"] = GRADIO_ROOT_PATH
    demo.queue().launch(**launch_kwargs)
