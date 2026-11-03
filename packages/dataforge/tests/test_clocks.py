import numpy as np
import pytest

from dataforge.clocks import frame_times, nearest_framesets


@pytest.mark.parametrize("fps", [30, 60])
def test_frame_times_match_the_multiply_by_period_form(fps: int) -> None:
    # HO-Cap wrote k * (1e9 / fps) and Assembly101 k / fps * 1e9; the two differ by at most one ulp, and at 30 and
    # 60 fps the exact fractional nanosecond is 0, 1/3 or 2/3 (never 1/2), so rint lands on the same integer.
    frames = np.arange(10_000_000, dtype=np.int64)
    period_form = np.rint(np.arange(10_000_000, dtype=np.float64) * (1e9 / fps)).astype(np.int64)
    np.testing.assert_array_equal(frame_times(frames, fps), period_form)


def test_nearest_framesets_clamp_and_tie_goes_left() -> None:
    actual = nearest_framesets(np.array([10, 20, 30], dtype=np.int64), np.array([0, 10, 14, 15, 16, 25, 30, 40], dtype=np.int64))
    np.testing.assert_array_equal(actual, [0, 0, 0, 0, 1, 1, 2, 2])


def test_nearest_framesets_handles_irregular_float_samples_ties_and_boundaries() -> None:
    source = np.array([0.0, 0.25, 0.5])
    target = np.array([-1.0, 0.125, 0.38, 1.0])
    np.testing.assert_array_equal(nearest_framesets(source, target), [0, 0, 2, 2])
    assert nearest_framesets(source, np.empty(0)).dtype == np.int64


def test_nearest_framesets_preserves_unvalidated_nan_selection() -> None:
    np.testing.assert_array_equal(nearest_framesets(np.array([1.0]), np.array([np.nan, -2.0, 5.0])), [0, 0, 0])
