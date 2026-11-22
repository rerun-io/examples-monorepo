"""The Gradio app builds without a GPU and refuses a run without a video."""

import gradio as gr

from fdanyone_rerun.gradio_ui.app import APP_PERSON, build, run_app
from fdanyone_rerun.person4d import PersonConfig


def test_app_builds() -> None:
    assert isinstance(build(examples=[]), gr.Blocks)


def test_run_without_a_video_asks_for_one() -> None:
    assert list(run_app(None, 6, 42, True, 5000)) == [(None, "Choose a video first.")]


def test_the_app_preset_is_short_and_runs_no_extra_estimator() -> None:
    # Pablo's demo-app rule: no MoGe-2 in the app; the CLI default keeps the prior and the long run.
    assert APP_PERSON.prior is False and APP_PERSON.train.steps == 5_000
    assert PersonConfig().prior is True and PersonConfig().train.steps == 25_000
    assert APP_PERSON.max_side == PersonConfig().max_side and APP_PERSON.asog == PersonConfig().asog
