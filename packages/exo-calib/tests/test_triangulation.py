import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

from exo_calib.correspondences import ObservationSet
from exo_calib.triangulation import triangulate_points, triangulate_robust


def test_triangulate_points_recovers_exact_synthetic_scene() -> None:
    n_views: int = 4
    n_points: int = 20
    angles_v: Float64[ndarray, "4"] = np.arange(n_views, dtype=np.float64) * (np.pi / 2.0)
    centers_v3: Float64[ndarray, "4 3"] = np.column_stack(
        (4.0 * np.cos(angles_v), 4.0 * np.sin(angles_v), np.zeros(n_views, dtype=np.float64))
    )
    cam_T_world_v44: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], n_views, axis=0)
    world_up_3: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    view_idx: int
    for view_idx in range(n_views):
        forward_3: Float64[ndarray, "3"] = -centers_v3[view_idx] / np.linalg.norm(centers_v3[view_idx])
        right_3: Float64[ndarray, "3"] = np.cross(forward_3, world_up_3)
        down_3: Float64[ndarray, "3"] = np.cross(forward_3, right_3)
        rotation_33: Float64[ndarray, "3 3"] = np.stack((right_3, down_3, forward_3))
        cam_T_world_v44[view_idx, :3, :3] = rotation_33
        cam_T_world_v44[view_idx, :3, 3] = -rotation_33 @ centers_v3[view_idx]

    k_33: Float64[ndarray, "3 3"] = np.array([[800.0, 0.0, 320.0], [0.0, 820.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    k_v33: Float64[ndarray, "4 3 3"] = np.repeat(k_33[None], n_views, axis=0)
    rng: np.random.Generator = np.random.default_rng(4)
    points_xyz_n3: Float64[ndarray, "20 3"] = rng.uniform(-0.5, 0.5, size=(n_points, 3)).astype(np.float64)
    points_homo_n4: Float64[ndarray, "20 4"] = np.column_stack((points_xyz_n3, np.ones(n_points, dtype=np.float64)))
    points_cam_vn3: Float64[ndarray, "4 20 3"] = np.einsum("vij,nj->vni", cam_T_world_v44[:, :3], points_homo_n4)
    projected_vn3: Float64[ndarray, "4 20 3"] = np.einsum("vij,vnj->vni", k_v33, points_cam_vn3)
    projected_xy_vn2: Float64[ndarray, "4 20 2"] = projected_vn3[:, :, :2] / projected_vn3[:, :, 2:3]
    obs_point_idx_m: Int64[ndarray, "80"] = np.repeat(np.arange(n_points, dtype=np.int64), n_views)
    obs_view_idx_m: Int64[ndarray, "80"] = np.tile(np.arange(n_views, dtype=np.int64), n_points)
    obs_xy_m2: Float64[ndarray, "80 2"] = projected_xy_vn2.transpose(1, 0, 2).reshape(-1, 2)
    observations: ObservationSet = ObservationSet(
        point_frame_idx=np.arange(n_points, dtype=np.int64),
        point_joint_idx=np.zeros(n_points, dtype=np.int64),
        obs_point_idx=obs_point_idx_m,
        obs_view_idx=obs_view_idx_m,
        obs_xy=obs_xy_m2,
        obs_conf=np.ones(obs_point_idx_m.size, dtype=np.float64),
    )

    actual_xyz_n3: Float64[ndarray, "20 3"] = triangulate_points(observations, k_v33, cam_T_world_v44)

    np.testing.assert_allclose(actual_xyz_n3, points_xyz_n3, atol=1e-6)


def test_triangulate_robust_drops_one_gross_outlier_and_recovers_point() -> None:
    n_views: int = 4
    n_points: int = 5
    angles_v: Float64[ndarray, "4"] = np.arange(n_views, dtype=np.float64) * (np.pi / 2.0)
    centers_v3: Float64[ndarray, "4 3"] = np.column_stack(
        (4.0 * np.cos(angles_v), 4.0 * np.sin(angles_v), np.zeros(n_views, dtype=np.float64))
    )
    cam_T_world_v44: Float64[ndarray, "4 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], n_views, axis=0)
    world_up_3: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    view_idx: int
    for view_idx in range(n_views):
        forward_3: Float64[ndarray, "3"] = -centers_v3[view_idx] / np.linalg.norm(centers_v3[view_idx])
        right_3: Float64[ndarray, "3"] = np.cross(forward_3, world_up_3)
        down_3: Float64[ndarray, "3"] = np.cross(forward_3, right_3)
        rotation_33: Float64[ndarray, "3 3"] = np.stack((right_3, down_3, forward_3))
        cam_T_world_v44[view_idx, :3, :3] = rotation_33
        cam_T_world_v44[view_idx, :3, 3] = -rotation_33 @ centers_v3[view_idx]

    k_33: Float64[ndarray, "3 3"] = np.array([[800.0, 0.0, 320.0], [0.0, 820.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    k_v33: Float64[ndarray, "4 3 3"] = np.repeat(k_33[None], n_views, axis=0)
    points_xyz_n3: Float64[ndarray, "5 3"] = np.array(
        [[-0.3, -0.2, 0.1], [0.2, -0.1, -0.4], [0.1, 0.3, 0.2], [-0.2, 0.4, -0.1], [0.4, 0.1, 0.3]], dtype=np.float64
    )
    points_homo_n4: Float64[ndarray, "5 4"] = np.column_stack((points_xyz_n3, np.ones(n_points, dtype=np.float64)))
    points_cam_vn3: Float64[ndarray, "4 5 3"] = np.einsum("vij,nj->vni", cam_T_world_v44[:, :3], points_homo_n4)
    projected_vn3: Float64[ndarray, "4 5 3"] = np.einsum("vij,vnj->vni", k_v33, points_cam_vn3)
    projected_xy_vn2: Float64[ndarray, "4 5 2"] = projected_vn3[:, :, :2] / projected_vn3[:, :, 2:3]
    obs_point_idx_m: Int64[ndarray, "20"] = np.repeat(np.arange(n_points, dtype=np.int64), n_views)
    obs_view_idx_m: Int64[ndarray, "20"] = np.tile(np.arange(n_views, dtype=np.int64), n_points)
    obs_xy_m2: Float64[ndarray, "20 2"] = projected_xy_vn2.transpose(1, 0, 2).reshape(-1, 2)
    outlier_obs_idx: int = 2 * n_views + 3
    obs_xy_m2[outlier_obs_idx, 0] += 50.0
    observations: ObservationSet = ObservationSet(
        point_frame_idx=np.arange(n_points, dtype=np.int64),
        point_joint_idx=np.zeros(n_points, dtype=np.int64),
        obs_point_idx=obs_point_idx_m,
        obs_view_idx=obs_view_idx_m,
        obs_xy=obs_xy_m2,
        obs_conf=np.ones(obs_point_idx_m.size, dtype=np.float64),
    )

    robust_result: tuple[Float64[ndarray, "5 3"], ObservationSet] = triangulate_robust(
        observations, k_v33, cam_T_world_v44, reproj_threshold_px=10.0
    )

    refined_xyz_n3: Float64[ndarray, "5 3"] = robust_result[0]
    pruned_observations: ObservationSet = robust_result[1]
    assert pruned_observations.obs_point_idx.size == observations.obs_point_idx.size - 1
    assert np.count_nonzero(pruned_observations.obs_point_idx == 2) == 3
    np.testing.assert_allclose(refined_xyz_n3, points_xyz_n3, atol=1e-6)
