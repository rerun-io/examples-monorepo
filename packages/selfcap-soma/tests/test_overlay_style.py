import numpy as np

from selfcap_soma.overlay_style import observed_links, observed_points


def test_only_confident_nearby_finite_detections_are_observed() -> None:
    projected = np.array([[[10.0, 10.0], [10.0, 10.0], [10.0, 10.0], [np.nan, np.nan]]])
    detected = np.array([[[20.0, 10.0], [30.0, 10.0], [10.0, 10.0], [10.0, 10.0]]])
    scores = np.array([[0.8, 0.9, 0.6, 1.0]])
    seen = observed_points(projected, detected, scores)
    np.testing.assert_array_equal(seen, [[True, False, False, False]])
    np.testing.assert_array_equal(observed_links(seen, np.array([[0, 1], [0, 0]])), [[False, True]])


def test_residual_tolerance_scales_with_depth_without_accepting_low_confidence() -> None:
    from selfcap_soma.overlay_style import residual_tolerance

    fitted = np.zeros((2, 2, 3), dtype=np.float64)
    target = fitted.copy()
    target[:, 0, 0] = 0.02
    target[:, 1] = np.nan
    depth = np.array([[1.0, 1.0], [2.0, 2.0]])
    tolerance = residual_tolerance(fitted, target, depth, 1000.0)
    np.testing.assert_allclose(tolerance, [[40.0, 15.0], [20.0, 15.0]])
    projected = np.zeros((2, 2, 2))
    detected = projected.copy()
    detected[:, 0, 0] = 30.0
    seen = observed_points(projected, detected, np.array([[0.9, 0.6], [0.9, 0.9]]), tolerance)
    np.testing.assert_array_equal(seen, [[True, False], [False, True]])
