import numpy as np

from selfcap_soma.overlay_style import observed_links, observed_points


def test_only_confident_nearby_finite_detections_are_observed() -> None:
    projected = np.array([[[10.0, 10.0], [10.0, 10.0], [10.0, 10.0], [np.nan, np.nan]]])
    detected = np.array([[[20.0, 10.0], [30.0, 10.0], [10.0, 10.0], [10.0, 10.0]]])
    scores = np.array([[0.8, 0.9, 0.6, 1.0]])
    seen = observed_points(projected, detected, scores)
    np.testing.assert_array_equal(seen, [[True, False, False, False]])
    np.testing.assert_array_equal(observed_links(seen, np.array([[0, 1], [0, 0]])), [[False, True]])
