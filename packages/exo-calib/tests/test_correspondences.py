import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

from exo_calib.correspondences import ObservationSet, build_observations


def test_build_observations_requires_a_strong_cross_view_pair() -> None:
    kp_xy_vtj2: Float64[ndarray, "3 1 2 2"] = np.array(
        [[[[10.0, 20.0], [30.0, 40.0]]], [[[11.0, 20.0], [31.0, 40.0]]], [[[12.0, 20.0], [32.0, 40.0]]]], dtype=np.float64
    )
    conf_vtj: Float64[ndarray, "3 1 2"] = np.array([[[1.0, 0.9]], [[0.5, 0.9]], [[0.5, 0.1]]], dtype=np.float64)

    observations: ObservationSet = build_observations(kp_xy_vtj2, conf_vtj, min_pair_conf=0.75)

    np.testing.assert_array_equal(observations.point_frame_idx, np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(observations.point_joint_idx, np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(observations.obs_point_idx, np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(observations.obs_view_idx, np.array([0, 1], dtype=np.int64))
    np.testing.assert_allclose(observations.obs_xy, np.array([[30.0, 40.0], [31.0, 40.0]], dtype=np.float64))
    np.testing.assert_allclose(observations.obs_conf, np.array([0.9, 0.9], dtype=np.float64))


def test_build_observations_applies_stride_budget_and_seed_deterministically() -> None:
    kp_xy_vtj2: Float64[ndarray, "3 6 2 2"] = np.zeros((3, 6, 2, 2), dtype=np.float64)
    view_grid_v111: Float64[ndarray, "3 1 1 1"] = np.arange(3, dtype=np.float64).reshape(3, 1, 1, 1)
    frame_grid_1t11: Float64[ndarray, "1 6 1 1"] = np.arange(6, dtype=np.float64).reshape(1, 6, 1, 1)
    kp_xy_vtj2[..., :1] = view_grid_v111 + frame_grid_1t11
    conf_vtj: Float64[ndarray, "3 6 2"] = np.full((3, 6, 2), 0.9, dtype=np.float64)

    first: ObservationSet = build_observations(kp_xy_vtj2, conf_vtj, frame_stride=2, max_points=3, seed=7)
    second: ObservationSet = build_observations(kp_xy_vtj2, conf_vtj, frame_stride=2, max_points=3, seed=7)

    assert first.point_frame_idx.size == 3
    assert np.all(first.point_frame_idx % 2 == 0)
    obs_per_point_n: Int64[ndarray, "n_points"] = np.bincount(first.obs_point_idx, minlength=first.point_frame_idx.size)
    assert np.all(obs_per_point_n >= 2)
    np.testing.assert_array_equal(first.point_frame_idx, second.point_frame_idx)
    np.testing.assert_array_equal(first.point_joint_idx, second.point_joint_idx)
    np.testing.assert_array_equal(first.obs_point_idx, second.obs_point_idx)
    np.testing.assert_array_equal(first.obs_view_idx, second.obs_view_idx)
    np.testing.assert_allclose(first.obs_xy, second.obs_xy)
    np.testing.assert_allclose(first.obs_conf, second.obs_conf)
