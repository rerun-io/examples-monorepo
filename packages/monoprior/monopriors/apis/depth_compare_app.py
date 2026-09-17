"""Gradio depth-comparison app: side-by-side relative/metric depth models in an embedded Rerun viewer."""

from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from monopriors.gradio_ui.depth_compare_ui import EXAMPLE_DATA_DIR, relative_compare_block
from monopriors.gradio_ui.depth_inference_ui import depth_inference_block


@dataclass
class DepthCompareAppConfig:
    """Server settings for the depth-comparison Gradio app."""

    server_name: str = "127.0.0.1"
    """Interface to bind. Keep loopback and front it with ``tailscale serve --https=<port>`` for the tailnet."""

    port: int = 7861
    """Port to bind."""

    root_path: str = ""
    """Root path for reverse-proxy deployments that mount the app under a sub-path (e.g. ``/depth-compare``).
    Leave empty when the proxy forwards ``/`` unchanged, as ``tailscale serve --https=<port>`` does."""

    allowed_paths: tuple[Path, ...] = ()
    """Extra directories Gradio may serve files from; the example-image directory is always allowed."""


def build_demo() -> gr.Blocks:
    """Assemble the two-tab app (comparison + single-model inference)."""
    with gr.Blocks(title="Depth Comparison") as demo:
        gr.Markdown("# Depth Comparison")
        gr.Markdown("Demo to help compare different depth models. Including both Scale | Shift Invariant and Metric Depth types.")
        gr.Markdown(
            "Invariant models mean they have no true scale and are only relative, "
            "where as Metric models have a true scale and are absolute (meters)."
        )
        gr.Markdown(
            "Checkout the [Github Repo](https://github.com/pablovela5620/monoprior) "
            "[![GitHub Repo stars](https://img.shields.io/github/stars/pablovela5620/monoprior)](https://github.com/pablovela5620/monoprior)"
        )
        gr.Markdown("### Depth Prediction demo")
        with gr.Tab(label="Depth Comparison"):
            relative_compare_block.render()
        with gr.Tab(label="Depth Inference"):
            depth_inference_block.render()
    return demo


def main(config: DepthCompareAppConfig) -> None:
    demo = build_demo()
    demo.queue().launch(
        server_name=config.server_name,
        server_port=config.port,
        root_path=config.root_path or None,
        allowed_paths=[str(EXAMPLE_DATA_DIR), *(str(p) for p in config.allowed_paths)],
        ssr_mode=False,
    )
