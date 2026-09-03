import numpy as np
from conftest import observe_points, ring_rig
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.correspondences import ObservationSet
from exo_calib.triangulation import RobustTriangulationResult, triangulate_points, triangulate_robust


def test_triangulate_points_recovers_exact_synthetic_scene() -> None:
    n_views: int = 4
    n_points: int = 20
    cam_T_world: Float64[ndarray, "4 4 4"] = ring_rig(n_views, radius_m=4.0)
    camera_intrinsics: Float64[ndarray, "3 3"] = np.array([[800.0, 0.0, 320.0], [0.0, 820.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    intrinsics: Float64[ndarray, "4 3 3"] = np.repeat(camera_intrinsics[None], n_views, axis=0)
    rng: np.random.Generator = np.random.default_rng(4)
    points_xyz: Float64[ndarray, "20 3"] = rng.uniform(-0.5, 0.5, size=(n_points, 3)).astype(np.float64)
    observations: ObservationSet = observe_points(points_xyz, intrinsics, cam_T_world)

    actual_xyz: Float64[ndarray, "20 3"] = triangulate_points(observations, intrinsics, cam_T_world)

    np.testing.assert_allclose(actual_xyz, points_xyz, atol=1e-6)


def test_triangulate_robust_drops_one_gross_outlier_and_recovers_point() -> None:
    n_views: int = 4
    cam_T_world: Float64[ndarray, "4 4 4"] = ring_rig(n_views, radius_m=4.0)
    camera_intrinsics: Float64[ndarray, "3 3"] = np.array([[800.0, 0.0, 320.0], [0.0, 820.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    intrinsics: Float64[ndarray, "4 3 3"] = np.repeat(camera_intrinsics[None], n_views, axis=0)
    points_xyz: Float64[ndarray, "5 3"] = np.array(
        [[-0.3, -0.2, 0.1], [0.2, -0.1, -0.4], [0.1, 0.3, 0.2], [-0.2, 0.4, -0.1], [0.4, 0.1, 0.3]], dtype=np.float64
    )
    observations: ObservationSet = observe_points(points_xyz, intrinsics, cam_T_world)
    outlier_obs_idx: int = 2 * n_views + 3
    observations.obs_xy[outlier_obs_idx, 0] += 50.0

    robust: RobustTriangulationResult = triangulate_robust(observations, intrinsics, cam_T_world, reproj_threshold_px=10.0)

    refined_xyz: Float64[ndarray, "5 3"] = robust.points_xyz
    pruned_observations: ObservationSet = robust.obs
    assert pruned_observations.obs_point_idx.size == observations.obs_point_idx.size - 1
    assert np.count_nonzero(pruned_observations.obs_point_idx == 2) == 3
    np.testing.assert_allclose(refined_xyz, points_xyz, atol=1e-6)
