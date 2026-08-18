"""Smoke test: verify the rerun_prompt_da package can be imported."""


def test_import_rerun_prompt_da() -> None:
    """Import the package to catch broken package metadata or init logic."""

    import rerun_prompt_da  # noqa: F401


def test_import_prompt_da_gradio_ui_without_example_data() -> None:
    """Import the Gradio UI even when optional sample data has not been downloaded."""
    import pytest

    pytest.importorskip("gradio", reason="Gradio is optional outside the PromptDA demo lane")
    import rerun_prompt_da.gradio_ui.prompt_da_ui  # noqa: F401
