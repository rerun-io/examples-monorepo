"""Smoke test: verify the pyvrs_viewer package can be imported."""


def test_import_pyvrs_viewer() -> None:
    """Import the package to catch broken package metadata or init logic."""
    import pyvrs_viewer  # noqa: F401
