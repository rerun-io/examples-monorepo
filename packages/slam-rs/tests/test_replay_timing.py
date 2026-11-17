"""The replay's per-stage wall times: one column per stage, summarised per frameset."""

from slam_rs.vio_log import StageTimes


def test_the_summary_gives_each_stage_its_median_and_mean_in_insertion_order() -> None:
    times: StageTimes = StageTimes()
    for decode, track in ((10.0, 30.0), (20.0, 40.0), (60.0, 50.0)):
        times.add("decode", decode)
        times.add("track", track)
    assert times.summary() == "ms per frameset, median / mean over 3: decode 20.0 / 30.0, track 40.0 / 40.0"


def test_a_stage_seen_on_fewer_framesets_says_how_many() -> None:
    times: StageTimes = StageTimes()
    times.add("decode", 1.0)
    times.add("decode", 3.0)
    times.add("estimator", 5.0)
    assert times.summary() == "ms per frameset, median / mean over 2: decode 2.0 / 2.0, estimator 5.0 / 5.0 (n=1)"


def test_nothing_recorded_is_said_plainly() -> None:
    assert StageTimes().summary() == "ms per frameset: nothing timed"
