"""Confidence processing for multi-view keypoints."""

import warnings

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray


def modulate_crop_edge_confidence(
    kp_xy_n2: Float64[ndarray, "n 2"], conf_n: Float64[ndarray, "n"], crop_wh: tuple[int, int], margin_px: float
) -> Float64[ndarray, "n"]:
    """Reduce confidence near or outside crop edges.

    Args:
        kp_xy_n2: Float64 pixel coordinates with shape ``(n, 2)``.
        conf_n: Float64 detector confidences with shape ``(n,)``.
        crop_wh: Crop width and height in pixels.
        margin_px: Width of the confidence falloff in pixels.

    Returns:
        Float64 modulated confidences with shape ``(n,)``.
    """
    crop_w: int = crop_wh[0]
    crop_h: int = crop_wh[1]
    x_n: Float64[ndarray, "n"] = kp_xy_n2[:, 0]
    y_n: Float64[ndarray, "n"] = kp_xy_n2[:, 1]
    edge_scale_n: Float64[ndarray, "n"] = np.minimum.reduce(
        (x_n / margin_px, (float(crop_w) - x_n) / margin_px, y_n / margin_px, (float(crop_h) - y_n) / margin_px, np.ones_like(x_n))
    )
    edge_scale_n = np.where(np.isfinite(kp_xy_n2).all(axis=1), np.maximum(edge_scale_n, 0.0), 0.0)
    modulated_conf_n: Float64[ndarray, "n"] = np.where(edge_scale_n > 0.0, conf_n * edge_scale_n, 0.0)
    return modulated_conf_n


def threshold_rescale_confidence(conf_n: Float64[ndarray, "n"], tau: float = 0.15) -> Float64[ndarray, "n"]:
    """Threshold and linearly rescale detector confidences.

    Args:
        conf_n: Float64 detector confidences with shape ``(n,)``.
        tau: Inclusive confidence cutoff mapped to zero.

    Returns:
        Float64 rescaled confidences with shape ``(n,)``.
    """
    rescaled_conf_n: Float64[ndarray, "n"] = np.where(conf_n >= tau, (conf_n - tau) / (1.0 - tau), 0.0)
    return rescaled_conf_n


def median_filter_keypoints(kp_xy_tn2: Float64[ndarray, "t n 2"], window: int = 5) -> Float64[ndarray, "t n 2"]:
    """Apply a centered temporal median to keypoint coordinates.

    Args:
        kp_xy_tn2: Float64 keypoint coordinates with shape ``(t, n, 2)``.
        window: Odd temporal window size.

    Returns:
        Float64 filtered keypoint coordinates with shape ``(t, n, 2)``.
    """
    half_window: int = window // 2
    filtered_xy_tn2: Float64[ndarray, "t n 2"] = np.empty_like(kp_xy_tn2)
    frame_idx: int
    for frame_idx in range(kp_xy_tn2.shape[0]):
        start_idx: int = max(0, frame_idx - half_window)
        stop_idx: int = min(kp_xy_tn2.shape[0], frame_idx + half_window + 1)
        window_xy_tn2: Float64[ndarray, "window_t n 2"] = kp_xy_tn2[start_idx:stop_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            filtered_xy_tn2[frame_idx] = np.nanmedian(window_xy_tn2, axis=0)
    return filtered_xy_tn2


def kineo_point_confidence(
    points_xyz_n3: Float64[ndarray, "n 3"],
    kp_xy_vn2: Float64[ndarray, "v n 2"],
    conf_vn: Float64[ndarray, "v n"],
    k_v33: Float64[ndarray, "v 3 3"],
    cam_T_world_v44: Float64[ndarray, "v 4 4"],
    lambda_decay: float = 1.0,
) -> Float64[ndarray, "n"]:
    """Compute Kineo confidence from multi-view reprojection consistency.

    Args:
        points_xyz_n3: Float64 world points with shape ``(n, 3)``.
        kp_xy_vn2: Float64 observed pixels with shape ``(v, n, 2)``.
        conf_vn: Float64 detector confidences with shape ``(v, n)``.
        k_v33: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world_v44: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.
        lambda_decay: Exponential reprojection-error decay.

    Returns:
        Float64 point confidences with shape ``(n,)``.
    """
    n_points: int = points_xyz_n3.shape[0]
    n_views: int = kp_xy_vn2.shape[0]
    if n_views < 2:
        return np.zeros(n_points, dtype=np.float64)

    points_homo_n4: Float64[ndarray, "n 4"] = np.concatenate((points_xyz_n3, np.ones((n_points, 1), dtype=np.float64)), axis=1)
    points_cam_vn3: Float64[ndarray, "v n 3"] = np.einsum("vij,nj->vni", cam_T_world_v44[:, :3, :], points_homo_n4)
    projected_homo_vn3: Float64[ndarray, "v n 3"] = np.einsum("vij,vnj->vni", k_v33, points_cam_vn3)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_xy_vn2: Float64[ndarray, "v n 2"] = projected_homo_vn3[:, :, :2] / projected_homo_vn3[:, :, 2:3]

    focal_v: Float64[ndarray, "v"] = (k_v33[:, 0, 0] + k_v33[:, 1, 1]) / 2.0
    reproj_error_vn: Float64[ndarray, "v n"] = np.linalg.norm(projected_xy_vn2 - kp_xy_vn2, axis=2)
    normalized_error_vn: Float64[ndarray, "v n"] = reproj_error_vn / (focal_v[:, None] + 1e-8)
    valid_view_vn: Bool[ndarray, "v n"] = (
        (points_cam_vn3[:, :, 2] > 0.0)
        & np.isfinite(projected_xy_vn2).all(axis=2)
        & np.isfinite(kp_xy_vn2).all(axis=2)
        & np.isfinite(conf_vn)
        & (conf_vn > 0.0)
    )
    view_score_vn: Float64[ndarray, "v n"] = np.where(valid_view_vn, np.exp(-lambda_decay * normalized_error_vn), 0.0)
    valid_conf_vn: Float64[ndarray, "v n"] = np.where(valid_view_vn, conf_vn, 0.0)
    pair_sum_n: Float64[ndarray, "n"] = np.zeros(n_points, dtype=np.float64)
    first_view_idx: int
    second_view_idx: int
    for first_view_idx in range(n_views - 1):
        for second_view_idx in range(first_view_idx + 1, n_views):
            pair_score_n: Float64[ndarray, "n"] = np.sqrt(view_score_vn[first_view_idx] * view_score_vn[second_view_idx])
            pair_weight_n: Float64[ndarray, "n"] = np.sqrt(valid_conf_vn[first_view_idx] * valid_conf_vn[second_view_idx])
            pair_sum_n += pair_score_n * pair_weight_n

    n_pairs: int = n_views * (n_views - 1) // 2
    point_conf_n: Float64[ndarray, "n"] = pair_sum_n / float(n_pairs)
    point_conf_n[np.sum(valid_view_vn, axis=0) < 2] = 0.0
    return point_conf_n
