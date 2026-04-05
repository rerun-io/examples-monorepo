"""Shared fixtures for mast3r-slam tests."""

from pathlib import Path

import pytest

PACKAGE_DIR: Path = Path(__file__).parent.parent
FIXTURES_DIR: Path = PACKAGE_DIR / "tests" / "fixtures"
BASELINE_RRD: Path = FIXTURES_DIR / "baseline_30frames.rrd"


@pytest.fixture(scope="session")
def baseline_rrd_path() -> Path:
    """Path to the pre-generated baseline .rrd for regression tests.

    Generate with: pixi run -e mast3r-slam --frozen _generate-test-baseline
    """
    if not BASELINE_RRD.exists():
        pytest.skip(f"Baseline RRD not found at {BASELINE_RRD}. Run _generate-test-baseline first.")
    return BASELINE_RRD
