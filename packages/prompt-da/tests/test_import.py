"""Smoke test: verify the rerun_prompt_da package can be imported."""


def test_import_rerun_prompt_da() -> None:
    """Import the package to catch broken package metadata or init logic."""

    import rerun_prompt_da  # noqa: F401
