"""Bundle adjustment for multi-camera calibration."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float64, Int64
from kornia_rs.k3d import bundle_adjust  # pyrefly: ignore[missing-import]  # Compiled extension exposes k3d only at runtime.
from numpy import ndarray

from exo_calib.correspondences import ObservationSet
from exo_calib.triangulation import reprojection_errors_px, triangulate_points, valid_observation_mask


@dataclass(slots=True)
class BaResult:
    """Refined calibration and per-round diagnostics."""

    cam_T_world_v44: Float64[ndarray, "v 4 4"]
    """Refined Float64 world-to-camera transforms with shape ``(v, 4, 4)``."""
    points_xyz_n3: Float64[ndarray, "n_points 3"]
    """Re-triangulated Float64 world points with shape ``(n_points, 3)``."""
    mean_reproj_px_per_round: list[float]
    """Confidence-weighted pixel error after each refinement round."""
    converged: bool
    """Whether bundle adjustment converged in every round."""


def mean_reprojection_error_px(
    obs: ObservationSet,
    points_xyz_n3: Float64[ndarray, "n_points 3"],
    k_v33: Float64[ndarray, "v 3 3"],
    cam_T_world_v44: Float64[ndarray, "v 4 4"],
) -> float:
    """Compute confidence-weighted mean reprojection error in pixels.

    Args:
        obs: Sparse pixel observations for the points.
        points_xyz_n3: Float64 world points with shape ``(n_points, 3)``.
        k_v33: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world_v44: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.

    Returns:
        Confidence-weighted mean pixel error, or NaN when no observation is valid.
    """
    error_m: Float64[ndarray, "n_obs"] = reprojection_errors_px(obs, points_xyz_n3, k_v33, cam_T_world_v44)
    finite_m: Bool[ndarray, "n_obs"] = np.isfinite(error_m)
    if not finite_m.any():
        return float("nan")
    weight_r: Float64[ndarray, "r"] = obs.obs_conf[finite_m]
    return float(np.sum(weight_r * error_m[finite_m]) / np.sum(weight_r))


def refine_extrinsics(
    cam_T_world_v44: Float64[ndarray, "v 4 4"],
    k_v33: Float64[ndarray, "v 3 3"],
    obs: ObservationSet,
    points_xyz_n3: Float64[ndarray, "n_points 3"],
    *,
    rounds: int = 2,
    robust: Literal["identity", "huber", "cauchy", "tukey"] = "huber",
    robust_scale_px: float = 2.0,
    fixed_pose_indices: tuple[int, ...] = (0,),
    max_iterations: int = 50,
    pose_prior_sigma_m: float | None = None,
) -> BaResult:
    """Refine camera extrinsics with normalized-intrinsics bundle adjustment.

    Args:
        cam_T_world_v44: Float64 initial world-to-camera transforms with shape ``(v, 4, 4)``.
        k_v33: Float64 per-camera intrinsics with shape ``(v, 3, 3)``.
        obs: Sparse pixel observations for the points.
        points_xyz_n3: Float64 initial world points with shape ``(n_points, 3)``.
        rounds: Number of bundle-adjustment and re-triangulation rounds.
        robust: Robust loss used by ``kornia_rs``.
        robust_scale_px: Robust loss scale in pixels before focal normalization.
        fixed_pose_indices: Camera poses held fixed to remove gauge freedom.
        max_iterations: Maximum solver iterations in each round.
        pose_prior_sigma_m: Soft prior (meters) anchoring every camera center to
            its initial estimate; ``None`` disables the priors.

    Returns:
        Refined poses, re-triangulated points, diagnostics, and convergence state.
    """
    n_views: int = cam_T_world_v44.shape[0]
    n_points: int = points_xyz_n3.shape[0]
    current_cam_T_world_v44: Float64[ndarray, "v 4 4"] = cam_T_world_v44.copy()
    current_points_xyz_n3: Float64[ndarray, "n_points 3"] = points_xyz_n3.copy()
    mean_reproj_px_per_round: list[float] = []
    all_converged: bool = True
    inverse_k_v33: Float64[ndarray, "v 3 3"] = np.linalg.inv(k_v33)
    focal_v: Float64[ndarray, "v"] = (k_v33[:, 0, 0] + k_v33[:, 1, 1]) / 2.0
    mean_focal: float = float(np.mean(focal_v))
    normalized_robust_scale: float = robust_scale_px / mean_focal
    fixed_pose_index_list: list[int] = list(fixed_pose_indices)
    # Soft anchors on the INITIAL camera centers. Reprojection-only BA leaves a
    # 7-DOF similarity gauge (fixing one pose still leaves global scale free);
    # weak center priors pin every gauge direction without fighting real signal,
    # and keep the refinement a *refinement* of the Stage A metric rig.
    prior_centers_v3: Float64[ndarray, "v 3"] | None = None
    prior_sigmas_v: Float64[ndarray, "v"] | None = None
    if pose_prior_sigma_m is not None:
        init_rotations_v33: Float64[ndarray, "v 3 3"] = cam_T_world_v44[:, :3, :3]
        init_translations_v3: Float64[ndarray, "v 3"] = cam_T_world_v44[:, :3, 3]
        prior_centers_v3 = -np.einsum("vji,vj->vi", init_rotations_v33, init_translations_v3)
        prior_sigmas_v = np.full(n_views, pose_prior_sigma_m, dtype=np.float64)

    _round_idx: int
    for _round_idx in range(rounds):
        round_start_cam_T_world_v44: Float64[ndarray, "v 4 4"] = current_cam_T_world_v44.copy()
        round_start_points_xyz_n3: Float64[ndarray, "n_points 3"] = current_points_xyz_n3.copy()
        valid_reference_m: Bool[ndarray, "n_obs"] = valid_observation_mask(obs, n_views, n_points=n_points)
        valid_reference_idx_q: Int64[ndarray, "q"] = np.flatnonzero(valid_reference_m).astype(np.int64)
        if valid_reference_idx_q.size == 0:
            all_converged = False
            break
        referenced_point_idx_q: Int64[ndarray, "q"] = obs.obs_point_idx[valid_reference_idx_q]
        finite_point_q: Bool[ndarray, "q"] = np.isfinite(current_points_xyz_n3[referenced_point_idx_q]).all(axis=1)
        valid_obs_idx_m: Int64[ndarray, "m_valid"] = valid_reference_idx_q[finite_point_q]
        if valid_obs_idx_m.size == 0:
            all_converged = False
            break

        old_point_idx_m: Int64[ndarray, "m_valid"] = obs.obs_point_idx[valid_obs_idx_m]
        dense_old_point_idx_d: Int64[ndarray, "n_valid"] = np.unique(old_point_idx_m).astype(np.int64)
        old_to_dense_n: Int64[ndarray, "n_points"] = np.full(n_points, -1, dtype=np.int64)
        old_to_dense_n[dense_old_point_idx_d] = np.arange(dense_old_point_idx_d.size, dtype=np.int64)
        dense_point_idx_m: Int64[ndarray, "m_valid"] = old_to_dense_n[old_point_idx_m]
        view_idx_m: Int64[ndarray, "m_valid"] = obs.obs_view_idx[valid_obs_idx_m]
        pixel_homo_m3: Float64[ndarray, "m_valid 3"] = np.column_stack(
            (obs.obs_xy[valid_obs_idx_m], np.ones(valid_obs_idx_m.size, dtype=np.float64))
        )
        normalized_homo_m3: Float64[ndarray, "m_valid 3"] = np.einsum("mij,mj->mi", inverse_k_v33[view_idx_m], pixel_homo_m3)
        normalized_xy_m2: Float64[ndarray, "m_valid 2"] = normalized_homo_m3[:, :2]
        ba_observations_m4: Float64[ndarray, "m_valid 4"] = np.column_stack(
            (view_idx_m.astype(np.float64), dense_point_idx_m.astype(np.float64), normalized_xy_m2)
        )
        rotations_v33: Float64[ndarray, "v 3 3"] = current_cam_T_world_v44[:, :3, :3].copy()
        translations_v3: Float64[ndarray, "v 3"] = current_cam_T_world_v44[:, :3, 3].copy()
        dense_points_xyz_d3: Float64[ndarray, "n_valid 3"] = current_points_xyz_n3[dense_old_point_idx_d].copy()
        identity_k_33: Float64[ndarray, "3 3"] = np.eye(3, dtype=np.float64)
        ba_output: tuple[
            Float64[ndarray, "v 3 3"], Float64[ndarray, "v 3"], Float64[ndarray, "n_valid 3"], int, bool
        ] = bundle_adjust(
            rotations_v33,
            translations_v3,
            dense_points_xyz_d3,
            ba_observations_m4,
            identity_k_33,
            fixed_pose_indices=fixed_pose_index_list,
            fix_all_points=False,
            max_iterations=max_iterations,
            robust=robust,
            robust_scale=normalized_robust_scale,
            # "schur" eliminates the 3N point parameters into a camera-only
            # reduced system; dense "lm" builds O((6V+3N)^2) normal equations
            # and stalls for minutes at a few thousand points. It is also the
            # only solver honoring pose priors.
            solver="schur",
            pose_prior_centers=None if prior_centers_v3 is None else prior_centers_v3.astype(np.float32),
            pose_prior_sigmas=None if prior_sigmas_v is None else prior_sigmas_v.astype(np.float32),
        )
        optimized_rotations_v33: Float64[ndarray, "v 3 3"] = ba_output[0]
        optimized_translations_v3: Float64[ndarray, "v 3"] = ba_output[1]
        optimized_dense_points_xyz_d3: Float64[ndarray, "n_valid 3"] = ba_output[2]
        round_converged: bool = ba_output[4]
        current_cam_T_world_v44[:, :3, :3] = optimized_rotations_v33
        current_cam_T_world_v44[:, :3, 3] = optimized_translations_v3
        current_points_xyz_n3[dense_old_point_idx_d] = optimized_dense_points_xyz_d3
        round_mean_reproj_px: float = mean_reprojection_error_px(obs, current_points_xyz_n3, k_v33, current_cam_T_world_v44)
        if mean_reproj_px_per_round and round_mean_reproj_px > mean_reproj_px_per_round[-1]:
            current_cam_T_world_v44 = round_start_cam_T_world_v44
            current_points_xyz_n3 = round_start_points_xyz_n3
            mean_reproj_px_per_round.append(mean_reproj_px_per_round[-1])
            all_converged = all_converged and round_converged
            continue
        mean_reproj_px_per_round.append(round_mean_reproj_px)
        current_points_xyz_n3 = triangulate_points(obs, k_v33, current_cam_T_world_v44)
        all_converged = all_converged and round_converged

    return BaResult(
        cam_T_world_v44=current_cam_T_world_v44,
        points_xyz_n3=current_points_xyz_n3,
        mean_reproj_px_per_round=mean_reproj_px_per_round,
        converged=all_converged,
    )
