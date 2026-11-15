import numpy as np
from conftest import observe_points, ring_rig
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.ba import BaResult, mean_reprojection_error_px, refine_extrinsics, revert_large_pose_updates
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
    points_xyz: Float64[ndarray, "1 3"] = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    intrinsics: Float64[ndarray, "2 3 3"] = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    cam_T_world: Float64[ndarray, "2 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)

    actual_error_px: float = mean_reprojection_error_px(observations, points_xyz, intrinsics, cam_T_world)

    assert actual_error_px == 2.0


def test_refine_extrinsics_recovers_perturbed_synthetic_rig() -> None:
    n_views: int = 8
    n_points: int = 200
    true_cam_T_world: Float64[ndarray, "8 4 4"] = ring_rig(n_views, radius_m=4.0)
    intrinsics: Float64[ndarray, "8 3 3"] = np.zeros((n_views, 3, 3), dtype=np.float64)
    intrinsics[:, 0, 0] = 750.0 + 10.0 * np.arange(n_views, dtype=np.float64)
    intrinsics[:, 1, 1] = 770.0 + 8.0 * np.arange(n_views, dtype=np.float64)
    intrinsics[:, 0, 2] = 320.0
    intrinsics[:, 1, 2] = 240.0
    intrinsics[:, 2, 2] = 1.0
    rng: np.random.Generator = np.random.default_rng(12)
    true_points_xyz: Float64[ndarray, "200 3"] = rng.uniform(-0.6, 0.6, size=(n_points, 3)).astype(np.float64)
    observations: ObservationSet = observe_points(true_points_xyz, intrinsics, true_cam_T_world)

    initial_cam_T_world: Float64[ndarray, "8 4 4"] = true_cam_T_world.copy()
    angle_rad: float = np.deg2rad(2.0)
    for view_idx in range(1, n_views):
        axis: Float64[ndarray, "3"] = rng.normal(size=3).astype(np.float64)
        axis /= np.linalg.norm(axis)
        skew: Float64[ndarray, "3 3"] = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=np.float64
        )
        delta_rotation: Float64[ndarray, "3 3"] = (
            np.eye(3, dtype=np.float64) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)
        )
        initial_cam_T_world[view_idx, :3, :3] = delta_rotation @ true_cam_T_world[view_idx, :3, :3]
        initial_cam_T_world[view_idx, :3, 3] += rng.normal(scale=0.05, size=3)

    initial_points_xyz: Float64[ndarray, "200 3"] = triangulate_points(observations, intrinsics, initial_cam_T_world)

    result: BaResult = refine_extrinsics(
        initial_cam_T_world,
        intrinsics,
        observations,
        initial_points_xyz,
        rounds=2,
        robust="huber",
        robust_scale_px=2.0,
        fixed_pose_indices=(0,),
        max_iterations=30,
    )

    rotation_error_deg: Float64[ndarray, "8"] = np.empty(n_views, dtype=np.float64)
    for view_idx in range(n_views):
        relative_rotation: Float64[ndarray, "3 3"] = result.cam_T_world[view_idx, :3, :3] @ true_cam_T_world[view_idx, :3, :3].T
        cosine: float = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
        rotation_error_deg[view_idx] = np.rad2deg(np.arccos(cosine))

    refined_centers: Float64[ndarray, "8 3"] = -np.einsum(
        "vji,vj->vi", result.cam_T_world[:, :3, :3], result.cam_T_world[:, :3, 3]
    )
    true_centers: Float64[ndarray, "8 3"] = -np.einsum(
        "vji,vj->vi", true_cam_T_world[:, :3, :3], true_cam_T_world[:, :3, 3]
    )
    refined_baselines: Float64[ndarray, "8 3"] = refined_centers - refined_centers[0]
    true_baselines: Float64[ndarray, "8 3"] = true_centers - true_centers[0]
    alignment_scale: float = float(np.sum(refined_baselines * true_baselines) / np.sum(refined_baselines**2))
    aligned_centers: Float64[ndarray, "8 3"] = true_centers[0] + alignment_scale * refined_baselines
    translation_error_m: Float64[ndarray, "8"] = np.linalg.norm(aligned_centers - true_centers, axis=1)

    assert np.max(rotation_error_deg) < 0.1
    assert np.max(translation_error_m) < 0.005
    assert len(result.mean_reproj_px_per_round) == 2
    assert np.all(np.diff(result.mean_reproj_px_per_round) <= 1e-12)
    assert result.converged


def test_revert_large_pose_updates_restores_only_implausible_updates() -> None:
    initial_cam_T_world: Float64[ndarray, "2 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    refined_cam_T_world: Float64[ndarray, "2 4 4"] = initial_cam_T_world.copy()
    small_angle_rad: float = np.deg2rad(5.0)
    large_angle_rad: float = np.deg2rad(20.0)
    refined_cam_T_world[0, :2, :2] = np.array(
        [[np.cos(small_angle_rad), -np.sin(small_angle_rad)], [np.sin(small_angle_rad), np.cos(small_angle_rad)]]
    )
    refined_cam_T_world[1, :2, :2] = np.array(
        [[np.cos(large_angle_rad), -np.sin(large_angle_rad)], [np.sin(large_angle_rad), np.cos(large_angle_rad)]]
    )
    refined_cam_T_world[:, :3, 3] = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)

    guarded_cam_T_world, reverted = revert_large_pose_updates(refined_cam_T_world, initial_cam_T_world, maximum_rotation_deg=10.0)

    np.testing.assert_allclose(guarded_cam_T_world[0], refined_cam_T_world[0])
    np.testing.assert_allclose(guarded_cam_T_world[1], initial_cam_T_world[1])
    assert reverted == (1,)
