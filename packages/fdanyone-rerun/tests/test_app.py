"""The Gradio app builds without a GPU and refuses a run without a video."""

import gradio as gr

from fdanyone_rerun.gradio_ui.app import build, run_app


def test_app_builds() -> None:
    assert isinstance(build(examples=[]), gr.Blocks)


def test_run_without_a_video_asks_for_one() -> None:
    assert list(run_app(None, 6, 42)) == [(None, "Choose a video first.")]
