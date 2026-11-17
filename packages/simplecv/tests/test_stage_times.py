"""Per-frame stage times: the summary the benchmarks write and the line the replay prints."""

import pytest

from simplecv.stage_times import StageTimes


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


def test_the_line_gives_each_stage_its_median_and_mean_and_a_short_column_its_count() -> None:
    times = StageTimes()
    times.add("decode", 1.0)
    times.add("decode", 3.0)
    times.add("estimator", 5.0)
    assert times.line("frameset") == "ms per frameset, median / mean over 2: decode 2.0 / 2.0, estimator 5.0 / 5.0 (n=1)"
    assert StageTimes().line("frame") == "ms per frame: nothing timed"


def test_the_line_keeps_the_order_the_stages_were_first_seen() -> None:
    times = StageTimes()
    for decode, track in ((10.0, 30.0), (20.0, 40.0), (60.0, 50.0)):
        times.add("decode", decode)
        times.add("track", track)
    assert times.line("frameset") == "ms per frameset, median / mean over 3: decode 20.0 / 30.0, track 40.0 / 40.0"
