"""The bench harness's schedule and its best-of-medians rule.

Both are the protocol rather than the plumbing: the interleave is what cancels
host drift, and taking the best of the per-round medians is what makes a row
reproduce. Neither needs a GPU, a dump or a core, so they are checked here in
milliseconds.
"""

import numpy as np
import pytest

from slam_rs.apis.bench_track import Lane, LaneRound, best_median_ms, interleave


def test_the_schedule_runs_every_lane_once_per_round() -> None:
    """Round-major, lanes in the given order: drift hits both lanes in a round."""
    schedule: list[tuple[int, Lane]] = interleave(("cpu", "gpu"), 3)
    assert schedule == [
        (1, "cpu"),
        (1, "gpu"),
        (2, "cpu"),
        (2, "gpu"),
        (3, "cpu"),
        (3, "gpu"),
    ]


def test_every_round_holds_the_whole_lane_set_in_the_same_order() -> None:
    """The property drift-cancelling needs, over three lanes and four rounds.

    Rounds come out in order and each one is the full lane list, so no lane can
    be measured only early or only late in a sweep.
    """
    lanes: tuple[Lane, ...] = ("cpu", "gpu", "cpu")
    schedule: list[tuple[int, Lane]] = interleave(lanes, 4)
    assert len(schedule) == 12
    rounds: list[int] = [entry[0] for entry in schedule]
    assert rounds == sorted(rounds)
    for round_index in range(1, 5):
        assert [entry[1] for entry in schedule if entry[0] == round_index] == list(lanes)


def _round(lane: Lane, index: int, samples: list[float]) -> LaneRound:
    return LaneRound(
        lane=lane,
        round_index=index,
        track_ms=np.asarray(samples, dtype=np.float64),
        wall_s=1.0,
        cpu_pct=np.asarray([99.8], dtype=np.float64),
    )


def test_the_reported_figure_is_the_lowest_round_median() -> None:
    """Not the mean of the rounds and not the minimum call: the best median."""
    rounds: list[LaneRound] = [
        _round("gpu", 1, [6.0, 6.2, 6.4]),
        _round("gpu", 2, [5.8, 6.0, 6.2]),
        _round("gpu", 3, [1.0, 9.0, 9.4]),
    ]
    assert best_median_ms(rounds) == pytest.approx(6.0)


def test_a_lane_with_no_rounds_is_refused() -> None:
    """An empty lane has no figure, and saying so beats reporting zero."""
    with pytest.raises(ValueError, match="no rounds"):
        best_median_ms([])
