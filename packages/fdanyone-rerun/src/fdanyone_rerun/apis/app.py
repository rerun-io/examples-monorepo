"""``fdanyone-app``: serve the Gradio app."""

from fdanyone_rerun.gradio_ui.app import Config, build, launch


def main(config: Config) -> None:
    """Serve the app."""
    launch(build(), config)
