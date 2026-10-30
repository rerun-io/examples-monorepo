"""Nearest-sample matching retains clipped boundaries and left ties."""

import numpy as np

from simplecv.temporal import nearest_indices


def test_nearest_indices_handles_irregular_samples_ties_and_boundaries() -> None:
    source = np.array([0.0, 0.25, 0.5])
    target = np.array([-1.0, 0.125, 0.38, 1.0])
    np.testing.assert_array_equal(nearest_indices(source, target), [0, 0, 2, 2])
    assert nearest_indices(source, np.empty(0)).dtype == np.int64


def test_nearest_indices_preserves_unvalidated_nan_selection() -> None:
    np.testing.assert_array_equal(nearest_indices(np.array([1.0]), np.array([np.nan, -2.0, 5.0])), [0, 0, 0])
