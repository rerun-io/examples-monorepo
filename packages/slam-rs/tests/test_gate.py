"""Ground-truth gate boundary and monotonicity properties."""

from dataclasses import replace
from typing import Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from slam_rs.reference import Baseline, Measurement, ReferenceManifest, gate_failures

BASELINE: Baseline = Baseline("fast", "gpu", "baseline-host", "0" * 64, 100, 10.0, 20.0, "2026-09-10")
PASSING: Measurement = Measurement(100, 100, 0, 100, 10.0, 1.0, 1.0, True, 20.0, "baseline-host", "gpu", "fast")


@pytest.mark.parametrize(
    ("accepted", "refused", "clause"),
    [
        (replace(PASSING, tracked=10), replace(PASSING, tracked=9), "tracked"),
        (replace(PASSING, associated=10), replace(PASSING, associated=9), "associated"),
        (PASSING, replace(PASSING, lost=1), "lost"),
        (replace(PASSING, gt_rmse_cm=11.0), replace(PASSING, gt_rmse_cm=11.000000000000002), "accuracy"),
        (replace(PASSING, median_tracker_ms=22.0), replace(PASSING, median_tracker_ms=22.000000000000004), "speed"),
        (replace(PASSING, extent_m=10.0), replace(PASSING, extent_m=10.000000000000002), "divergence"),
        (replace(PASSING, truth_extent_m=0.1), replace(PASSING, truth_extent_m=0.09999999999999999), "divergence"),
        (PASSING, replace(PASSING, poses_finite=False), "finite"),
    ],
)
def test_clause_flips_at_boundary(accepted: Measurement, refused: Measurement, clause: str) -> None:
    assert gate_failures(accepted, BASELINE) == []
    failures: list[str] = gate_failures(refused, BASELINE)
    assert len(failures) == 1
    assert failures[0].startswith(f"{clause}:")


@given(low=st.integers(0, 1000), extra=st.integers(0, 1000))
def test_more_tracked_or_associated_poses_cannot_add_failures(low: int, extra: int) -> None:
    for before, after in (
        (replace(PASSING, tracked=low), replace(PASSING, tracked=low + extra)),
        (replace(PASSING, associated=low), replace(PASSING, associated=low + extra)),
    ):
        assert bool(gate_failures(after, BASELINE)) <= bool(gate_failures(before, BASELINE))


@given(low=st.floats(min_value=0.0, max_value=100.0), extra=st.floats(min_value=0.0, max_value=100.0))
def test_larger_error_cost_or_extent_cannot_remove_failures(low: float, extra: float) -> None:
    for before, after in (
        (replace(PASSING, gt_rmse_cm=low), replace(PASSING, gt_rmse_cm=low + extra)),
        (replace(PASSING, median_tracker_ms=low), replace(PASSING, median_tracker_ms=low + extra)),
        (replace(PASSING, extent_m=low), replace(PASSING, extent_m=low + extra)),
    ):
        assert bool(gate_failures(after, BASELINE)) >= bool(gate_failures(before, BASELINE))


@given(lane=st.sampled_from(["cpu", "gpu"]), profile=st.sampled_from(["reference", "fast"]))
def test_baseline_resolution_matches_both_lane_and_profile(
    manifest: ReferenceManifest, lane: Literal["cpu", "gpu"], profile: Literal["reference", "fast"]
) -> None:
    segment = replace(manifest.segments[0], baseline=(BASELINE,))
    measurement: Measurement = replace(PASSING, lane=lane, profile=profile, gt_rmse_cm=100.0, median_tracker_ms=100.0)
    failures: list[str] = gate_failures(measurement, segment.baseline_for(lane, profile))
    if lane == "gpu" and profile == "fast":
        assert [message.split(":")[0] for message in failures] == ["accuracy", "speed"]
    else:
        assert failures == []


def test_other_host_gates_accuracy_but_not_speed() -> None:
    failures: list[str] = gate_failures(replace(PASSING, hostname="other", gt_rmse_cm=100.0, median_tracker_ms=100.0), BASELINE)
    assert len(failures) == 1
    assert failures[0].startswith("accuracy:")


@given(value=st.sampled_from([float("nan"), float("inf"), float("-inf")]))
def test_each_nonfinite_measurement_names_finite_clause(value: float) -> None:
    for measurement in (
        replace(PASSING, gt_rmse_cm=value), replace(PASSING, extent_m=value),
        replace(PASSING, truth_extent_m=value), replace(PASSING, median_tracker_ms=value),
    ):
        assert "finite: poses and measurements must be finite" in gate_failures(measurement, None)


@pytest.mark.slow
def test_catalog_smoke_gate(manifest: ReferenceManifest) -> None:
    from slam_rs.apis.fleet_check import ClipResult, measure

    for segment in manifest.in_tier("smoke"):
        result: ClipResult = measure(manifest, segment)
        assert not result.failures, result.verdict


@pytest.mark.parametrize("baseline", [replace(BASELINE, lane="cpu"), replace(BASELINE, profile="reference")])
def test_mismatched_baseline_is_a_caller_error(baseline: Baseline) -> None:
    with pytest.raises(ValueError, match=f"baseline {baseline.lane}/{baseline.profile}.*measurement gpu/fast"):
        gate_failures(PASSING, baseline)
