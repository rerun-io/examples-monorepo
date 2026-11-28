"""Fixtures shared by the viewer tests."""

from pathlib import Path

import pytest

# The canonical viewer's frame probe (packages/canonical-viewer/rust/crates/viewer-probe). The tests-integration task
# builds it first, with `canonical-viewer-build`.
VIEWER_PROBE: Path = Path(__file__).resolve().parents[2] / "canonical-viewer/target/release/viewer-probe"


@pytest.fixture(scope="module")
def viewer_probe() -> Path:
    """The release `viewer-probe`: headless canonical-viewer frames with the gsplat extension's counters."""
    if not VIEWER_PROBE.is_file():
        pytest.skip(f"{VIEWER_PROBE} is missing: run `pixi run -e canonical-viewer-dev canonical-viewer-build`")
    return VIEWER_PROBE
