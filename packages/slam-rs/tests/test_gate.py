"""Ground-truth gate thresholds through the public verdict function."""

from hypothesis import given
from hypothesis import strategies as st

from slam_rs.reference import Baseline, gate_failures
from slam_rs.trajectory import MIN_ASSOCIATED_POSES


@given(
    error=st.sampled_from([10.9, 11.0, 11.1]),
    speed=st.sampled_from([21.9, 22.0, 22.1]),
    lost=st.integers(0, 1),
    associated=st.sampled_from([MIN_ASSOCIATED_POSES - 1, MIN_ASSOCIATED_POSES]),
    finite=st.booleans(),
    extent=st.sampled_from([9.9, 10.0, 10.1]),
    same_host=st.booleans(),
)
def test_each_clause_is_required(error: float, speed: float, lost: int, associated: int, finite: bool, extent: float, same_host: bool) -> None:
    baseline: Baseline = Baseline("fast", "gpu", "baseline-host", "0" * 64, 100, 10.0, 20.0, "2026-09-10")
    failures: list[str] = gate_failures(
        framesets=100,
        tracked=100,
        lost=lost,
        associated=associated,
        gt_rmse_cm=error,
        extent_m=extent,
        truth_extent_m=1.0,
        poses_finite=finite,
        baseline=baseline,
        median_tracker_ms=speed,
        hostname="baseline-host" if same_host else "other",
        lane="gpu",
        profile="fast",
    )
    expected: set[str] = set()
    if error > 11.0:
        expected.add("accuracy")
    if same_host and speed > 22.0:
        expected.add("speed")
    if lost:
        expected.add("lost")
    if associated < MIN_ASSOCIATED_POSES:
        expected.add("associated")
    if not finite:
        expected.add("finite")
    if extent > 10.0:
        expected.add("divergence")
    assert {message.split(":")[0] for message in failures} == expected
