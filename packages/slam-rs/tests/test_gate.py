"""Ground-truth gate thresholds through the public verdict function."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from slam_rs.reference import Baseline, Measurement, ReferenceManifest, gate_failures
from slam_rs.trajectory import MIN_ASSOCIATED_POSES


@given(
    error=st.one_of(st.sampled_from([10.9, 11.0, 11.1]), st.floats(min_value=10.0, max_value=12.0)),
    speed=st.one_of(st.sampled_from([21.9, 22.0, 22.1]), st.floats(min_value=20.0, max_value=24.0)),
    lost=st.integers(0, 1),
    associated=st.sampled_from([MIN_ASSOCIATED_POSES - 1, MIN_ASSOCIATED_POSES]),
    finite=st.booleans(),
    extent=st.sampled_from([9.9, 10.0, 10.1]),
    same_host=st.booleans(),
)
def test_each_clause_is_required(error: float, speed: float, lost: int, associated: int, finite: bool, extent: float, same_host: bool) -> None:
    baseline: Baseline = Baseline("fast", "gpu", "baseline-host", "0" * 64, 100, 10.0, 20.0, "2026-09-10")
    failures: list[str] = gate_failures(Measurement(
        framesets=100,
        tracked=100,
        lost=lost,
        associated=associated,
        gt_rmse_cm=error,
        extent_m=extent,
        truth_extent_m=1.0,
        poses_finite=finite,
        median_tracker_ms=speed,
        hostname="baseline-host" if same_host else "other",
        lane="gpu",
        profile="fast",
    ), baseline)
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


@given(matching=st.booleans(), same_host=st.booleans(), has_baseline=st.booleans())
def test_missing_or_other_lane_baseline_does_not_gate_accuracy_or_speed(matching: bool, same_host: bool, has_baseline: bool) -> None:
    baseline: Baseline | None = (
        Baseline("fast", "gpu" if matching else "cpu", "baseline-host", "0" * 64, 100, 10.0, 20.0, "2026-09-10") if has_baseline else None
    )
    failures: list[str] = gate_failures(Measurement(
        framesets=100,
        tracked=100,
        lost=0,
        associated=100,
        gt_rmse_cm=100.0,
        extent_m=1.0,
        truth_extent_m=1.0,
        poses_finite=True,
        median_tracker_ms=100.0,
        hostname="baseline-host" if same_host else "other",
        lane="gpu",
        profile="fast",
    ), baseline if matching else None)
    clauses: set[str] = {message.split(":")[0] for message in failures}
    assert ("accuracy" in clauses) == (matching and has_baseline)
    assert ("speed" in clauses) == (matching and same_host and has_baseline)


@given(value=st.sampled_from([float("nan"), float("inf"), float("-inf")]))
def test_nonfinite_measurements_fail_without_a_baseline(value: float) -> None:
    baseline: Baseline | None = None
    failures: list[str] = gate_failures(Measurement(
        framesets=100,
        tracked=100,
        lost=0,
        associated=100,
        gt_rmse_cm=value,
        extent_m=1.0,
        truth_extent_m=1.0,
        poses_finite=True,
        median_tracker_ms=2.0,
        hostname="host",
        lane="gpu",
        profile="fast",
    ), baseline)
    assert any(message.startswith("finite:") for message in failures)


@pytest.mark.slow
def test_catalog_smoke_gate(manifest: ReferenceManifest) -> None:
    from slam_rs.apis.fleet_check import ClipResult, measure

    for segment in manifest.in_tier("smoke"):
        result: ClipResult = measure(manifest, segment)
        assert not result.failures, result.verdict
