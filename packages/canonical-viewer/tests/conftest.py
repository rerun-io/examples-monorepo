"""Shared fixtures. No module-level workspace imports: pytest-beartype instruments canonical_viewer after this loads."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def canonical_viewer() -> Path:
    """The built rerun-canonical binary; skips when it has not been built."""
    from canonical_viewer import CANONICAL_VIEWER

    if not CANONICAL_VIEWER.is_file():
        pytest.skip(f"{CANONICAL_VIEWER} is missing: run `pixi run -e canonical-viewer-dev canonical-viewer-build`")
    return CANONICAL_VIEWER
