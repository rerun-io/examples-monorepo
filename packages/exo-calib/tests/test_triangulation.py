import numpy as np
from conftest import observe_points, project_points, ring_rig
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.ops.correspondences import ObservationSet
from exo_calib.ops.triangulation import RobustTriangulationResult, triangulate_frame_keypoints, triangulate_points, triangulate_robust


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


def test_triangulate_frame_keypoints_preserves_joint_rows_and_confidence() -> None:
    points_xyz: Float64[ndarray, "3 3"] = np.array(
        [[0.0, 0.0, 5.0], [0.5, 0.2, 4.0], [-0.2, 0.1, 6.0]], dtype=np.float64
    )
    intrinsics: Float64[ndarray, "3 3 3"] = np.repeat(
        np.array([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]], dtype=np.float64), 3, axis=0
    )
    cam_T_world: Float64[ndarray, "3 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    cam_T_world[1, 0, 3] = -1.0
    cam_T_world[2, 1, 3] = -1.0
    kp_xy: Float64[ndarray, "3 3 2"] = project_points(points_xyz, intrinsics, cam_T_world)
    conf: Float64[ndarray, "3 3"] = np.full((3, 3), 0.9, dtype=np.float64)
    conf[1:, 2] = 0.1

    triangulated_xyz, triangulated_conf = triangulate_frame_keypoints(
        kp_xy,
        conf,
        intrinsics,
        cam_T_world,
        min_confidence=0.5,
        reproj_threshold_px=1.0,
        view_gate_px=1.0,
    )

    np.testing.assert_allclose(triangulated_xyz[:2], points_xyz[:2], atol=1e-8)
    assert np.isnan(triangulated_xyz[2]).all()
    np.testing.assert_allclose(triangulated_conf, np.array([0.9, 0.9, 0.0], dtype=np.float64))


def test_a_view_holding_another_person_does_not_move_the_tracked_skeleton() -> None:
    """Six cameras; view 3 holds a skeleton 1 m to its side (100 px). Without the view gate the least-squares point spreads that
    error over every view (13-55 px each, under the 100 px observation gate) and the skeleton moves 43 cm."""
    intrinsics: Float64[ndarray, "6 3 3"] = np.repeat(np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])[None], 6, axis=0)
    cam_T_world: Float64[ndarray, "6 4 4"] = ring_rig(6, 4.0)
    skeleton: Float64[ndarray, "17 3"] = np.random.default_rng(3).uniform(-0.3, 0.3, (17, 3))

    kp_xy: Float64[ndarray, "6 17 2"] = project_points(skeleton, intrinsics, cam_T_world)
    kp_xy[3] = project_points(skeleton + cam_T_world[3, 0, :3], intrinsics, cam_T_world)[3]  # view 3's image-right axis, 1 m

    triangulated_xyz, _ = triangulate_frame_keypoints(
        kp_xy, np.full((6, 17), 0.95), intrinsics, cam_T_world, min_confidence=0.15, reproj_threshold_px=100.0, view_gate_px=24.0
    )

    np.testing.assert_allclose(triangulated_xyz, skeleton, atol=1e-6)
