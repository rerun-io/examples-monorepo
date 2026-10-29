"""Repeat clock inference on a known native-to-presented grid."""

import numpy as np

from selfcap_curate.links import fit_pattern


def test_repeat_pattern_recovers_native_clock_and_rejects_stillness() -> None:
    times = np.arange(5000, dtype=np.float64) / 50.0
    native = np.floor(times * 24.0 * (1.0 + 430e-6) + 0.68)
    differences = np.where(np.r_[False, np.diff(native) == 0.0], 0.0, 5.0)
    result = fit_pattern(times, differences)
    assert result is not None
    assert result.ppm_vs_24 == 430
    assert result.phi == 0.68
    assert result.agreement == 1.0
    assert fit_pattern(times, np.zeros_like(times)) is None
