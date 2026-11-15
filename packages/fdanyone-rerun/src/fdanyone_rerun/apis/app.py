"""``fdanyone-app``: serve the Gradio app."""

from taehv import TAEHV

from fdanyone_rerun.apis.run import PREVIEW_CHECKPOINT
from fdanyone_rerun.assets import model_dir
from fdanyone_rerun.gradio_ui.app import Config, build, launch
from fdanyone_rerun.previews import load_preview_decoder


def main(config: Config) -> None:
    """Load the preview decoder once, then serve."""
    decoder: TAEHV = load_preview_decoder(model_dir() / PREVIEW_CHECKPOINT, "cuda")
    launch(build(decoder), config)
