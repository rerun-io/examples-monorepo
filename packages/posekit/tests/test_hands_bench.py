"""Stage bookkeeping of the hands speed benchmark."""

import pytest

from posekit.apis.hands_bench import StageTimes


def test_stage_summary_reports_median_and_mean_per_frame() -> None:
    times = StageTimes()
    for total, detect in ((10.0, 2.0), (30.0, 4.0), (20.0, 3.0)):
        times.add("total", total)
        times.add("detection", detect)
    summary = times.summary()
    assert summary["total"].median_ms == pytest.approx(20.0)
    assert summary["total"].mean_ms == pytest.approx(20.0)
    assert summary["detection"].median_ms == pytest.approx(3.0)
    assert summary["detection"].frames == 3
