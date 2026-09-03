"""Slow equivalence against the fork-recorded Aria fixture."""

from pathlib import Path

import numpy as np
import pytest
from simplecv.rerun_log_utils import RerunTyroConfig

from lamptrack.apis.lamp_replay import Config, fixture_path, load_snippets, replay

FIXTURE_DIR = Path(__file__).parents[1] / "data" / "fixtures" / "test-library"


@pytest.mark.slow
def test_fixture_lifter_and_smoothing_equivalence() -> None:
    """CPU fp32 lifting is exact and smoothed joints stay within 0.1 mm."""
    try:
        path = fixture_path(FIXTURE_DIR)
    except FileNotFoundError as exc:
        pytest.skip(f"fork-recorded LAMP fixture absent: {exc}")
    _, expected_lifter, expected_smoothed = load_snippets(path)
    if expected_lifter is None or expected_smoothed is None:
        pytest.skip(f"fixture {path} has no upstream equivalence arrays")
    lifted, smoothed = replay(
        Config(fixture_dir=FIXTURE_DIR, device="cpu", rr_config=RerunTyroConfig(application_id="lamp_fixture_test", headless=True))
    )
    assert np.array_equal(lifted, expected_lifter)
    assert np.allclose(smoothed, expected_smoothed, atol=1e-4, rtol=0.0)
