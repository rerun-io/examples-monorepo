import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

from exo_calib.correspondences import ObservationSet, build_observations, under_supported_view_indices


def test_build_observations_requires_a_strong_cross_view_pair() -> None:
    kp_xy: Float64[ndarray, "3 1 1 2"] = np.array([[[[10.0, 20.0]]], [[[11.0, 20.0]]], [[[12.0, 20.0]]]], dtype=np.float64)
    conf: Float64[ndarray, "3 1 1"] = np.array([[[0.4]], [[1.0]], [[1.0]]], dtype=np.float64)

    observations: ObservationSet = build_observations(kp_xy, conf, min_pair_conf=0.9, min_views=2)

    # sqrt(0.4 * 1.0) = 0.63 fails the pair rule; the two confident views form the point.
    np.testing.assert_array_equal(observations.point_frame_idx, np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(observations.point_joint_idx, np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(observations.obs_view_idx, np.array([1, 2], dtype=np.int64))
    np.testing.assert_allclose(observations.obs_xy, np.array([[11.0, 20.0], [12.0, 20.0]]))
    np.testing.assert_allclose(observations.obs_conf, np.array([1.0, 1.0]))
    assert build_observations(kp_xy, conf, min_pair_conf=0.9, min_views=3).point_frame_idx.size == 0


def test_under_supported_view_indices_uses_relative_median_support() -> None:
    view_indices: Int64[ndarray, " n"] = np.repeat(np.arange(6, dtype=np.int64), np.array([200, 180, 170, 15, 1, 0]))

    fixed_views: tuple[int, ...] = under_supported_view_indices(view_indices, n_views=6, minimum_ratio=0.1)

    assert fixed_views == (3, 4, 5)
