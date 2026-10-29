"""Synthetic clock alignment contracts."""

import numpy as np
import pytest

from selfcap_curate.links import SpeedSeries, correlate


def test_recovers_submillisecond_lag() -> None:
    times = np.arange(0.0, 30.0, 0.001)
    signal = np.sin(times * 2.3) + np.sin(times**1.4) + np.cos(times * 7.1)
    result = correlate(SpeedSeries(times, signal), SpeedSeries(times + 0.0374, signal), half_width_s=0.25, fs=1000.0)
    assert result.lag_s == pytest.approx(0.0374, abs=0.001)


def test_search_edge_is_missing() -> None:
    times = np.arange(0.0, 30.0, 0.001)
    speed = np.sin(times**1.5) + np.cos(times * 4.3)
    result = correlate(SpeedSeries(times, speed), SpeedSeries(times + 0.247, speed), half_width_s=0.25, fs=1000.0)
    assert result.lag_s is None
    assert result.clipped
