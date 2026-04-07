from itertools import combinations

import numpy as np
from einops import rearrange
from jaxtyping import Float, Int
from numpy import ndarray


def batch_triangulate(
    keypoints_2d: Float[ndarray, "n_views n_kpts 3"],
    projection_matrices: Float[ndarray, "n_views 3 4"],
    min_views: int = 2,
) -> Float[ndarray, "n_kpts 4"]:
    """
    Camera MUST be in OPENCV convention
    """
    num_joints: int = keypoints_2d.shape[1]

    # Count views where each joint is visible
    visibility_count: Int[ndarray, "n_kpts"] = (keypoints_2d[:, :, -1] > 0).sum(axis=0)  # noqa: UP037
    valid_joints = np.where(visibility_count >= min_views)[0]

    # Filter keypoints by valid joints
    filtered_keypoints: Float[ndarray, "n_views n_kpts 3"] = keypoints_2d[:, valid_joints]
    conf3d = filtered_keypoints[:, :, -1].sum(axis=0) / visibility_count[valid_joints]

    P0: Float[ndarray, "1 n_views 4"] = projection_matrices[None, :, 0, :]
    P1: Float[ndarray, "1 n_views 4"] = projection_matrices[None, :, 1, :]
    P2: Float[ndarray, "1 n_views 4"] = projection_matrices[None, :, 2, :]

    # x-coords homogenous
    u: Float[ndarray, "n_kpts n_views 1"] = rearrange(filtered_keypoints[..., 0], "c j -> j c 1")
    uP2: Float[ndarray, "n_kpts n_views 4"] = u * P2

    # y-coords homogenous
    v: Float[ndarray, "n_kpts n_views 1"] = rearrange(filtered_keypoints[..., 1], "c j -> j c 1")
    vP2: Float[ndarray, "n_kpts n_views 4"] = v * P2

    confidences: Float[ndarray, "n_kpts n_views 1"] = rearrange(filtered_keypoints[..., 2], "c j -> j c 1")

    Au: Float[ndarray, "n_kpts n_views 4"] = confidences * (uP2 - P0)
    Av: Float[ndarray, "n_kpts n_views 4"] = confidences * (vP2 - P1)
    A: Float[ndarray, "n_kpts _ 4"] = np.hstack([Au, Av])

    # Solve using SVD
    _, _, Vh = np.linalg.svd(A)
    triangulated_points = Vh[:, -1, :]
    triangulated_points /= triangulated_points[:, 3, None]

    # Construct result
    result: Float[ndarray, "n_kpts 4"] = np.zeros((num_joints, 4))
    # convert from homogenous to euclidean and add confidence
    result[valid_joints, :3] = triangulated_points[:, :3]
    result[valid_joints, 3] = conf3d

    return result


def _project_points(
    xyz1: Float[ndarray, "4"],
    P: Float[ndarray, "3 4"],
) -> tuple[Float[ndarray, "2"], float]:
    """Project a single homogeneous 3D point with a single projection matrix.

    Returns (uv, depth) where depth is the third homogeneous component before
    normalization. Camera MUST be in OPENCV convention.
    """
    x_proj: Float[ndarray, "3"] = (P @ xyz1)  # type: ignore[assignment]
    z: float = float(x_proj[2])
    uv: Float[ndarray, "2"] = (x_proj[:2] / x_proj[2]).astype(np.float64)
    return uv, z


def robust_batch_triangulate(
    keypoints_2d: Float[ndarray, "n_views n_kpts 3"],
    projection_matrices: Float[ndarray, "n_views 3 4"],
    min_views: int = 2,
    reproj_error_thresh: float = 3.0,
    init_subset: int = 3,
) -> Float[ndarray, "n_kpts 4"]:
    """Robustly triangulate 3D keypoints across multiple views.

    - Uses a RANSAC-like strategy per keypoint:
      1) enumerate small view subsets (2- or 3-view) as hypotheses
      2) score by reprojection error over all visible views
      3) re-triangulate with inliers and return the refined result

    Args:
        keypoints_2d: (n_views, n_kpts, 3) with (u, v, conf)
        projection_matrices: (n_views, 3, 4) projection matrices (OPENCV)
        min_views: minimum number of inlier views required to accept a 3D point
        reproj_error_thresh: inlier pixel threshold for reprojection error
        init_subset: initial hypothesis subset size (2 or 3 are typical)

    Returns:
        xyzc: (n_kpts, 4) with (x,y,z,confidence)
    """
    n_views: int = keypoints_2d.shape[0]
    n_kpts: int = keypoints_2d.shape[1]

    init_subset = max(2, min(init_subset, n_views))

    result: Float[ndarray, "n_kpts 4"] = np.zeros((n_kpts, 4), dtype=np.float64)

    # Precompute which views see each keypoint
    visibility: Int[ndarray, "n_views n_kpts"] = (keypoints_2d[..., 2] > 0).astype(np.int32)

    for j in range(n_kpts):
        visible_views_idx: Int[ndarray, "n_vis"] = np.where(visibility[:, j] > 0)[0]
        if visible_views_idx.size < min_views:
            # Not enough views; leave as zeros
            continue

        # Prepare per-view arrays for this keypoint
        uv_all: Float[ndarray, "n_vis 2"] = keypoints_2d[visible_views_idx, j, :2].astype(np.float64)
        conf_all: Float[ndarray, "n_vis"] = keypoints_2d[visible_views_idx, j, 2].astype(np.float64)
        P_all: Float[ndarray, "n_vis 3 4"] = projection_matrices[visible_views_idx]

        best_inliers: list[int] | None = None
        best_X: Float[ndarray, "4"] | None = None

        # Enumerate hypotheses from small subsets of views
        subset_size: int = min(init_subset, visible_views_idx.size)
        for subset in combinations(range(visible_views_idx.size), subset_size):
            # Triangulate using the subset via the existing batch_triangulate
            sub_idx = np.array(subset, dtype=np.int32)
            kpts_subset: Float[ndarray, "m 1 3"] = np.zeros((sub_idx.size, 1, 3), dtype=np.float64)
            kpts_subset[:, 0, :2] = uv_all[sub_idx]
            kpts_subset[:, 0, 2] = conf_all[sub_idx]

            P_subset: Float[ndarray, "m 3 4"] = P_all[sub_idx]
            Xj: Float[ndarray, "4"] = batch_triangulate(kpts_subset, P_subset, min_views=subset_size)[0]
            # If degenerate (confidence == 0), skip
            if not np.isfinite(Xj[:3]).all():
                continue
            Xh: Float[ndarray, "4"] = np.concatenate([Xj[:3], np.array([1.0])]).astype(np.float64)  # homogeneous

            # Score over all visible views
            errors: list[float] = []
            inliers: list[int] = []
            for vi, P in enumerate(P_all):
                uv_pred, depth = _project_points(Xh, P)
                # Ignore points behind the camera for inlier counting
                if depth <= 0:
                    errors.append(np.inf)
                    continue
                err = float(np.linalg.norm(uv_pred - uv_all[vi]))
                errors.append(err)
                if err <= reproj_error_thresh:
                    inliers.append(vi)

            if len(inliers) >= (best_inliers or []).__len__():
                best_inliers = inliers
                best_X = Xh

        # If no good hypothesis found, fallback to naive triangulation with all visible views
        if best_inliers is None or len(best_inliers) < min_views:
            kpts_all: Float[ndarray, "m 1 3"] = np.zeros((visible_views_idx.size, 1, 3), dtype=np.float64)
            kpts_all[:, 0, :2] = uv_all
            kpts_all[:, 0, 2] = conf_all
            X_naive: Float[ndarray, "4"] = batch_triangulate(kpts_all, P_all, min_views=min_views)[0]
            if np.isfinite(X_naive[:3]).all():
                result[j, :3] = X_naive[:3]
                # average conf of visible views
                result[j, 3] = float(conf_all.mean())
            continue

        # Refine using inliers
        inlier_idx = np.array(best_inliers, dtype=np.int32)
        kpts_in: Float[ndarray, "m 1 3"] = np.zeros((inlier_idx.size, 1, 3), dtype=np.float64)
        kpts_in[:, 0, :2] = uv_all[inlier_idx]
        kpts_in[:, 0, 2] = conf_all[inlier_idx]
        P_in: Float[ndarray, "m 3 4"] = P_all[inlier_idx]

        X_refined: Float[ndarray, "4"] = batch_triangulate(kpts_in, P_in, min_views=min_views)[0]
        if np.isfinite(X_refined[:3]).all():
            result[j, :3] = X_refined[:3]
            result[j, 3] = float(conf_all[inlier_idx].mean())
        else:
            # Fallback to best hypothesis if refinement failed
            assert best_X is not None
            result[j, :3] = best_X[:3]
            result[j, 3] = float(conf_all[inlier_idx].mean())

    return result
