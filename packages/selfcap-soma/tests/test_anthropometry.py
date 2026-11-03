"""Conservative anatomical guards leave plausible inputs unchanged."""

import numpy as np

from selfcap_soma.anthropometry import bounded_scale, plausible_ears
from selfcap_soma.shape_frames import select_shape_frames


def test_ear_span_is_inclusive_and_missing_pair_is_rejected() -> None:
    points = np.zeros((5, 133, 3), dtype=np.float64)
    points[:, 4, 0] = [0.109, 0.110, 0.150, 0.200, 0.201]
    np.testing.assert_array_equal(plausible_ears(points), [False, True, True, True, False])
    points[2, 3] = np.nan
    assert not plausible_ears(points)[2]


def test_scale_preserves_good_fit_and_bounds_outlier() -> None:
    assert bounded_scale(1.10) == 1.10
    assert bounded_scale(1.586) == 1.25
    assert bounded_scale(0.7) == 0.85


def test_torso_only_shape_fallback_when_all_ears_rejected() -> None:
    points = np.zeros((20, 133, 3), dtype=np.float64)
    weights = np.ones((20, 133), dtype=np.float64)
    weights[:, 3:5] = 0.0
    selection = select_shape_frames(np.arange(20, dtype=np.float64), points, weights, None, allow_torso_only=True)
    assert len(selection.indices) == 20
