"""Bundle adjustment for multi-camera calibration."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float64, Int64
from kornia_rs.k3d import bundle_adjust  # pyrefly: ignore[missing-import]  # Compiled extension exposes k3d only at runtime.
from numpy import ndarray

from exo_calib.correspondences import ObservationSet
from exo_calib.triangulation import reprojection_errors_px, triangulate_points, valid_observation_mask


@dataclass(slots=True, frozen=True)
class PosePrior:
    """Soft anchors on the camera centres.

    Reprojection-only BA leaves a 7-DOF similarity gauge (fixing one pose still
    leaves global scale free); weak centre priors pin every gauge direction
    without fighting real signal, and keep the refinement a *refinement* of the
    Stage A metric rig.
    """

    centers: Float64[ndarray, "v 3"]
    """Float64 world-frame centres the cameras are anchored to, shape ``(v, 3)``."""
    sigma_m: float
    """Prior standard deviation in metres, shared by every camera."""


@dataclass(slots=True)
class BaResult:
    """Refined calibration and per-round diagnostics."""

    cam_T_world: Float64[ndarray, "v 4 4"]
    """Refined Float64 world-to-camera transforms with shape ``(v, 4, 4)``."""
    points_xyz: Float64[ndarray, "n_points 3"]
    """Re-triangulated Float64 world points with shape ``(n_points, 3)``."""
    mean_reproj_px_per_round: list[float]
    """Confidence-weighted pixel error after each refinement round."""
    converged: bool
    """Whether bundle adjustment converged in every round."""


def mean_reprojection_error_px(
    obs: ObservationSet,
    points_xyz: Float64[ndarray, "n_points 3"],
    intrinsics: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
) -> float:
    """Compute confidence-weighted mean reprojection error in pixels.

    Args:
        obs: Sparse pixel observations for the points.
        points_xyz: Float64 world points with shape ``(n_points, 3)``.
        intrinsics: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.

    Returns:
        Confidence-weighted mean pixel error, or NaN when no observation is valid.
    """
    reprojection_error: Float64[ndarray, "n_obs"] = reprojection_errors_px(obs, points_xyz, intrinsics, cam_T_world)
    finite_mask: Bool[ndarray, "n_obs"] = np.isfinite(reprojection_error)
    if not finite_mask.any():
        return float("nan")
    weights: Float64[ndarray, "r"] = obs.obs_conf[finite_mask]
    return float(np.sum(weights * reprojection_error[finite_mask]) / np.sum(weights))


def refine_extrinsics(
    cam_T_world: Float64[ndarray, "v 4 4"],
    intrinsics: Float64[ndarray, "v 3 3"],
    obs: ObservationSet,
    points_xyz: Float64[ndarray, "n_points 3"],
    *,
    rounds: int,
    robust: Literal["huber", "cauchy"],
    robust_scale_px: float,
    max_iterations: int,
    fixed_pose_indices: tuple[int, ...] = (0,),
    pose_prior: PosePrior | None = None,
) -> BaResult:
    """Refine camera extrinsics with normalized-intrinsics bundle adjustment.

    Args:
        cam_T_world: Float64 initial world-to-camera transforms with shape ``(v, 4, 4)``.
        intrinsics: Float64 per-camera intrinsics with shape ``(v, 3, 3)``.
        obs: Sparse pixel observations for the points.
        points_xyz: Float64 initial world points with shape ``(n_points, 3)``.
        rounds: Number of bundle-adjustment and re-triangulation rounds.
        robust: Robust loss used by ``kornia_rs``.
        robust_scale_px: Robust loss scale in pixels before focal normalization.
        fixed_pose_indices: Camera poses held fixed to remove gauge freedom.
        max_iterations: Maximum solver iterations in each round.
        pose_prior: Soft camera-centre anchors; ``None`` runs reprojection-only BA.

    Returns:
        Refined poses, re-triangulated points, diagnostics, and convergence state.
    """
    n_views: int = cam_T_world.shape[0]
    n_points: int = points_xyz.shape[0]
    current_cam_T_world: Float64[ndarray, "v 4 4"] = cam_T_world.copy()
    current_points_xyz: Float64[ndarray, "n_points 3"] = points_xyz.copy()
    mean_reproj_px_per_round: list[float] = []
    all_converged: bool = True
    inverse_intrinsics: Float64[ndarray, "v 3 3"] = np.linalg.inv(intrinsics)
    focal: Float64[ndarray, "v"] = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2.0
    mean_focal: float = float(np.mean(focal))
    normalized_robust_scale: float = robust_scale_px / mean_focal
    fixed_pose_index_list: list[int] = list(fixed_pose_indices)
    prior_centers: Float64[ndarray, "v 3"] | None = None
    prior_sigmas: Float64[ndarray, "v"] | None = None
    if pose_prior is not None:
        prior_centers = pose_prior.centers
        prior_sigmas = np.full(n_views, pose_prior.sigma_m, dtype=np.float64)

    _round_idx: int
    for _round_idx in range(rounds):
        round_start_cam_T_world: Float64[ndarray, "v 4 4"] = current_cam_T_world.copy()
        round_start_points_xyz: Float64[ndarray, "n_points 3"] = current_points_xyz.copy()
        valid_reference_mask: Bool[ndarray, "n_obs"] = valid_observation_mask(obs, n_views, n_points=n_points)
        valid_reference_idx: Int64[ndarray, "q"] = np.flatnonzero(valid_reference_mask).astype(np.int64)
        if valid_reference_idx.size == 0:
            all_converged = False
            break
        referenced_point_idx: Int64[ndarray, "q"] = obs.obs_point_idx[valid_reference_idx]
        finite_point_mask: Bool[ndarray, "q"] = np.isfinite(current_points_xyz[referenced_point_idx]).all(axis=1)
        valid_obs_idx: Int64[ndarray, "m_valid"] = valid_reference_idx[finite_point_mask]
        if valid_obs_idx.size == 0:
            all_converged = False
            break

        old_point_idx: Int64[ndarray, "m_valid"] = obs.obs_point_idx[valid_obs_idx]
        dense_old_point_idx: Int64[ndarray, "n_valid"] = np.unique(old_point_idx).astype(np.int64)
        old_to_dense: Int64[ndarray, "n_points"] = np.full(n_points, -1, dtype=np.int64)
        old_to_dense[dense_old_point_idx] = np.arange(dense_old_point_idx.size, dtype=np.int64)
        dense_point_idx: Int64[ndarray, "m_valid"] = old_to_dense[old_point_idx]
        view_idx: Int64[ndarray, "m_valid"] = obs.obs_view_idx[valid_obs_idx]
        pixel_homo: Float64[ndarray, "m_valid 3"] = np.column_stack(
            (obs.obs_xy[valid_obs_idx], np.ones(valid_obs_idx.size, dtype=np.float64))
        )
        normalized_homo: Float64[ndarray, "m_valid 3"] = np.einsum("mij,mj->mi", inverse_intrinsics[view_idx], pixel_homo)
        normalized_xy: Float64[ndarray, "m_valid 2"] = normalized_homo[:, :2]
        ba_observations: Float64[ndarray, "m_valid 4"] = np.column_stack(
            (view_idx.astype(np.float64), dense_point_idx.astype(np.float64), normalized_xy)
        )
        rotations: Float64[ndarray, "v 3 3"] = current_cam_T_world[:, :3, :3].copy()
        translations: Float64[ndarray, "v 3"] = current_cam_T_world[:, :3, 3].copy()
        dense_points_xyz: Float64[ndarray, "n_valid 3"] = current_points_xyz[dense_old_point_idx].copy()
        identity_intrinsics: Float64[ndarray, "3 3"] = np.eye(3, dtype=np.float64)
        ba_output: tuple[
            Float64[ndarray, "v 3 3"], Float64[ndarray, "v 3"], Float64[ndarray, "n_valid 3"], int, bool
        ] = bundle_adjust(
            rotations,
            translations,
            dense_points_xyz,
            ba_observations,
            identity_intrinsics,
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
            pose_prior_centers=None if prior_centers is None else prior_centers.astype(np.float32),
            pose_prior_sigmas=None if prior_sigmas is None else prior_sigmas.astype(np.float32),
        )
        optimized_rotations: Float64[ndarray, "v 3 3"] = ba_output[0]
        optimized_translations: Float64[ndarray, "v 3"] = ba_output[1]
        optimized_dense_points_xyz: Float64[ndarray, "n_valid 3"] = ba_output[2]
        round_converged: bool = ba_output[4]
        current_cam_T_world[:, :3, :3] = optimized_rotations
        current_cam_T_world[:, :3, 3] = optimized_translations
        current_points_xyz[dense_old_point_idx] = optimized_dense_points_xyz
        round_mean_reproj_px: float = mean_reprojection_error_px(obs, current_points_xyz, intrinsics, current_cam_T_world)
        if mean_reproj_px_per_round and round_mean_reproj_px > mean_reproj_px_per_round[-1]:
            current_cam_T_world = round_start_cam_T_world
            current_points_xyz = round_start_points_xyz
            mean_reproj_px_per_round.append(mean_reproj_px_per_round[-1])
            all_converged = all_converged and round_converged
            continue
        mean_reproj_px_per_round.append(round_mean_reproj_px)
        current_points_xyz = triangulate_points(obs, intrinsics, current_cam_T_world)
        all_converged = all_converged and round_converged

    return BaResult(
        cam_T_world=current_cam_T_world,
        points_xyz=current_points_xyz,
        mean_reproj_px_per_round=mean_reproj_px_per_round,
        converged=all_converged,
    )


def revert_large_pose_updates(
    refined_cam_T_world: Float64[ndarray, "v 4 4"],
    initial_cam_T_world: Float64[ndarray, "v 4 4"],
    *,
    maximum_rotation_deg: float,
) -> tuple[Float64[ndarray, "v 4 4"], tuple[int, ...]]:
    """Restore initial poses whose BA rotation update exceeds a data-independent bound.

    Args:
        refined_cam_T_world: Float64 refined world-to-camera transforms, shape ``(v, 4, 4)``.
        initial_cam_T_world: Float64 initial world-to-camera transforms, shape ``(v, 4, 4)``.
        maximum_rotation_deg: Largest accepted rotation change per camera.

    Returns:
        The guarded transforms and the indices of the reverted cameras.
    """
    if maximum_rotation_deg < 0.0:
        raise ValueError("maximum_rotation_deg must be nonnegative")
    relative_rotation: Float64[ndarray, "v 3 3"] = np.einsum(
        "vij,vkj->vik", refined_cam_T_world[:, :3, :3], initial_cam_T_world[:, :3, :3]
    )
    cosine: Float64[ndarray, "v"] = np.clip(
        (np.trace(relative_rotation, axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    update_deg: Float64[ndarray, "v"] = np.rad2deg(np.arccos(cosine))
    reverted: tuple[int, ...] = tuple(int(view_idx) for view_idx in np.flatnonzero(update_deg > maximum_rotation_deg))
    guarded_cam_T_world: Float64[ndarray, "v 4 4"] = refined_cam_T_world.copy()
    if reverted:
        guarded_cam_T_world[np.asarray(reverted, dtype=np.int64)] = initial_cam_T_world[np.asarray(reverted, dtype=np.int64)]
    return guarded_cam_T_world, reverted
