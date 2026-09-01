import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.ba import BaResult, mean_reprojection_error_px, refine_extrinsics
from exo_calib.correspondences import ObservationSet
from exo_calib.triangulation import triangulate_points


def test_mean_reprojection_error_px_uses_confidence_weights_and_ignores_invalid() -> None:
    observations: ObservationSet = ObservationSet(
        point_frame_idx=np.array([0], dtype=np.int64),
        point_joint_idx=np.array([0], dtype=np.int64),
        obs_point_idx=np.array([0, 0, 0], dtype=np.int64),
        obs_view_idx=np.array([0, 1, 1], dtype=np.int64),
        obs_xy=np.array([[0.0, 0.0], [6.0, 0.0], [np.nan, 0.0]], dtype=np.float64),
        obs_conf=np.array([1.0, 0.5, 1.0], dtype=np.float64),
    )
    points_xyz_n3: Float64[ndarray, "1 3"] = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    k_v33: Float64[ndarray, "2 3 3"] = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    cam_T_world_v44: Float64[ndarray, "2 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)

    actual_error_px: float = mean_reprojection_error_px(observations, points_xyz_n3, k_v33, cam_T_world_v44)

    assert actual_error_px == 2.0


def test_refine_extrinsics_recovers_perturbed_synthetic_rig() -> None:
    n_views: int = 8
    n_points: int = 200
    angles_v: Float64[ndarray, "8"] = np.arange(n_views, dtype=np.float64) * (2.0 * np.pi / float(n_views))
    centers_v3: Float64[ndarray, "8 3"] = np.column_stack(
        (4.0 * np.cos(angles_v), 4.0 * np.sin(angles_v), np.zeros(n_views, dtype=np.float64))
    )
    true_cam_T_world_v44: Float64[ndarray, "8 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], n_views, axis=0)
    world_up_3: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    view_idx: int
    for view_idx in range(n_views):
        forward_3: Float64[ndarray, "3"] = -centers_v3[view_idx] / np.linalg.norm(centers_v3[view_idx])
        right_3: Float64[ndarray, "3"] = np.cross(forward_3, world_up_3)
        down_3: Float64[ndarray, "3"] = np.cross(forward_3, right_3)
        rotation_33: Float64[ndarray, "3 3"] = np.stack((right_3, down_3, forward_3))
        true_cam_T_world_v44[view_idx, :3, :3] = rotation_33
        true_cam_T_world_v44[view_idx, :3, 3] = -rotation_33 @ centers_v3[view_idx]

    k_v33: Float64[ndarray, "8 3 3"] = np.zeros((n_views, 3, 3), dtype=np.float64)
    k_v33[:, 0, 0] = 750.0 + 10.0 * np.arange(n_views, dtype=np.float64)
    k_v33[:, 1, 1] = 770.0 + 8.0 * np.arange(n_views, dtype=np.float64)
    k_v33[:, 0, 2] = 320.0
    k_v33[:, 1, 2] = 240.0
    k_v33[:, 2, 2] = 1.0
    rng: np.random.Generator = np.random.default_rng(12)
    true_points_xyz_n3: Float64[ndarray, "200 3"] = rng.uniform(-0.6, 0.6, size=(n_points, 3)).astype(np.float64)
    true_points_homo_n4: Float64[ndarray, "200 4"] = np.column_stack((true_points_xyz_n3, np.ones(n_points, dtype=np.float64)))
    true_points_cam_vn3: Float64[ndarray, "8 200 3"] = np.einsum("vij,nj->vni", true_cam_T_world_v44[:, :3], true_points_homo_n4)
    projected_vn3: Float64[ndarray, "8 200 3"] = np.einsum("vij,vnj->vni", k_v33, true_points_cam_vn3)
    projected_xy_vn2: Float64[ndarray, "8 200 2"] = projected_vn3[:, :, :2] / projected_vn3[:, :, 2:3]
    observations: ObservationSet = ObservationSet(
        point_frame_idx=np.arange(n_points, dtype=np.int64),
        point_joint_idx=np.zeros(n_points, dtype=np.int64),
        obs_point_idx=np.repeat(np.arange(n_points, dtype=np.int64), n_views),
        obs_view_idx=np.tile(np.arange(n_views, dtype=np.int64), n_points),
        obs_xy=projected_xy_vn2.transpose(1, 0, 2).reshape(-1, 2),
        obs_conf=np.ones(n_points * n_views, dtype=np.float64),
    )

    initial_cam_T_world_v44: Float64[ndarray, "8 4 4"] = true_cam_T_world_v44.copy()
    angle_rad: float = np.deg2rad(2.0)
    for view_idx in range(1, n_views):
        axis_3: Float64[ndarray, "3"] = rng.normal(size=3).astype(np.float64)
        axis_3 /= np.linalg.norm(axis_3)
        skew_33: Float64[ndarray, "3 3"] = np.array(
            [[0.0, -axis_3[2], axis_3[1]], [axis_3[2], 0.0, -axis_3[0]], [-axis_3[1], axis_3[0], 0.0]], dtype=np.float64
        )
        delta_rotation_33: Float64[ndarray, "3 3"] = (
            np.eye(3, dtype=np.float64) + np.sin(angle_rad) * skew_33 + (1.0 - np.cos(angle_rad)) * (skew_33 @ skew_33)
        )
        initial_cam_T_world_v44[view_idx, :3, :3] = delta_rotation_33 @ true_cam_T_world_v44[view_idx, :3, :3]
        initial_cam_T_world_v44[view_idx, :3, 3] += rng.normal(scale=0.05, size=3)

    initial_points_xyz_n3: Float64[ndarray, "200 3"] = triangulate_points(observations, k_v33, initial_cam_T_world_v44)

    result: BaResult = refine_extrinsics(
        initial_cam_T_world_v44,
        k_v33,
        observations,
        initial_points_xyz_n3,
        rounds=2,
        robust="huber",
        robust_scale_px=2.0,
        fixed_pose_indices=(0,),
        max_iterations=30,
    )

    rotation_error_deg_v: Float64[ndarray, "8"] = np.empty(n_views, dtype=np.float64)
    for view_idx in range(n_views):
        relative_rotation_33: Float64[ndarray, "3 3"] = result.cam_T_world_v44[view_idx, :3, :3] @ true_cam_T_world_v44[view_idx, :3, :3].T
        cosine: float = float(np.clip((np.trace(relative_rotation_33) - 1.0) / 2.0, -1.0, 1.0))
        rotation_error_deg_v[view_idx] = np.rad2deg(np.arccos(cosine))

    refined_centers_v3: Float64[ndarray, "8 3"] = -np.einsum(
        "vji,vj->vi", result.cam_T_world_v44[:, :3, :3], result.cam_T_world_v44[:, :3, 3]
    )
    true_centers_v3: Float64[ndarray, "8 3"] = -np.einsum(
        "vji,vj->vi", true_cam_T_world_v44[:, :3, :3], true_cam_T_world_v44[:, :3, 3]
    )
    refined_baselines_v3: Float64[ndarray, "8 3"] = refined_centers_v3 - refined_centers_v3[0]
    true_baselines_v3: Float64[ndarray, "8 3"] = true_centers_v3 - true_centers_v3[0]
    alignment_scale: float = float(np.sum(refined_baselines_v3 * true_baselines_v3) / np.sum(refined_baselines_v3**2))
    aligned_centers_v3: Float64[ndarray, "8 3"] = true_centers_v3[0] + alignment_scale * refined_baselines_v3
    translation_error_m_v: Float64[ndarray, "8"] = np.linalg.norm(aligned_centers_v3 - true_centers_v3, axis=1)

    assert np.max(rotation_error_deg_v) < 0.1
    assert np.max(translation_error_m_v) < 0.005
    assert len(result.mean_reproj_px_per_round) == 2
    assert np.all(np.diff(result.mean_reproj_px_per_round) <= 1e-12)
    assert result.converged
