"""Relative acceptance bands and explicit missing metrics."""

import pytest

from posekit.eval.hand_accuracy_bands import error_quantiles, within_accuracy_band


def test_lower_is_better_band_includes_five_percent() -> None:
    assert within_accuracy_band(100.0, 105.0, higher_is_better=False)
    assert not within_accuracy_band(100.0, 105.01, higher_is_better=False)


def test_higher_is_better_band_includes_five_percent() -> None:
    assert within_accuracy_band(0.8, 0.76, higher_is_better=True)
    assert not within_accuracy_band(0.8, 0.759, higher_is_better=True)


def test_missing_metrics_cannot_pass() -> None:
    assert not within_accuracy_band(None, 1.0, higher_is_better=True)
    assert not within_accuracy_band(1.0, None, higher_is_better=False)
    assert not within_accuracy_band(1.0, float("nan"), higher_is_better=False)


def test_errors_report_median_and_p95_without_discarding_failures() -> None:
    assert error_quantiles([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx((3.0, 4.8))
    assert error_quantiles([]) == (None, None)
    with pytest.raises(ValueError, match="finite"):
        error_quantiles([1.0, float("nan")])
