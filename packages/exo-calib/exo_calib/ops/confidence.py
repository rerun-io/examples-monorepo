"""Confidence processing for multi-view keypoints."""

import warnings

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from posekit.skeletons import COCO_17
from simplecv.data.skeleton.coco_133 import FACE_IDX

BODY_JOINTS: int = COCO_17.num_keypoints
CONFIDENT_JOINT: float = 0.5
MIN_BODY_JOINTS: int = 4
"""A view sees the body when at least this many COCO-17 joints pass ``CONFIDENT_JOINT``."""


def modulate_crop_edge_confidence(
    kp_xy: Float64[ndarray, "n 2"], conf: Float64[ndarray, "n"], crop_wh: tuple[int, int], margin_px: float
) -> Float64[ndarray, "n"]:
    """Reduce confidence near or outside crop edges.

    Args:
        kp_xy: Float64 pixel coordinates with shape ``(n, 2)``.
        conf: Float64 detector confidences with shape ``(n,)``.
        crop_wh: Crop width and height in pixels.
        margin_px: Width of the confidence falloff in pixels.

    Returns:
        Float64 modulated confidences with shape ``(n,)``.
    """
    crop_w: int = crop_wh[0]
    crop_h: int = crop_wh[1]
    x: Float64[ndarray, "n"] = kp_xy[:, 0]
    y: Float64[ndarray, "n"] = kp_xy[:, 1]
    edge_scale: Float64[ndarray, "n"] = np.minimum.reduce(
        (x / margin_px, (float(crop_w) - x) / margin_px, y / margin_px, (float(crop_h) - y) / margin_px, np.ones_like(x))
    )
    edge_scale = np.where(np.isfinite(kp_xy).all(axis=1), np.maximum(edge_scale, 0.0), 0.0)
    modulated_conf: Float64[ndarray, "n"] = np.where(edge_scale > 0.0, conf * edge_scale, 0.0)
    return modulated_conf


def threshold_rescale_confidence(conf: Float64[ndarray, "n"], *, tau: float) -> Float64[ndarray, "n"]:
    """Threshold and linearly rescale detector confidences.

    Args:
        conf: Float64 detector confidences with shape ``(n,)``.
        tau: Inclusive confidence cutoff mapped to zero.

    Returns:
        Float64 rescaled confidences with shape ``(n,)``.
    """
    rescaled_conf: Float64[ndarray, "n"] = np.where(conf >= tau, (conf - tau) / (1.0 - tau), 0.0)
    return rescaled_conf


def median_filter_keypoints(
    kp_xy: Float64[ndarray, "t n 2"], frame_index: Int64[ndarray, " t"], *, window: int
) -> Float64[ndarray, "t n 2"]:
    """Apply a centered temporal median over a window of VIDEO frames (AssemblyHands-X), not of records.

    Record ``t`` takes the median of the records whose frame index lies within ``window // 2`` video frames of its own,
    so a frame budget that samples sparser than that leaves every record as the model saw it, and the median never
    mixes poses seconds apart.

    Args:
        kp_xy: Float64 keypoint coordinates with shape ``(t, n, 2)``.
        frame_index: Video frame index of each record with shape ``(t,)``.
        window: Odd temporal window size in video frames.

    Returns:
        Float64 filtered keypoint coordinates with shape ``(t, n, 2)``.
    """
    filtered_xy: Float64[ndarray, "t n 2"] = np.empty_like(kp_xy)
    for t in range(kp_xy.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            filtered_xy[t] = np.nanmedian(kp_xy[np.abs(frame_index - frame_index[t]) <= window // 2], axis=0)
    return filtered_xy


def kineo_point_confidence(
    points_xyz: Float64[ndarray, "n 3"],
    kp_xy: Float64[ndarray, "v n 2"],
    conf: Float64[ndarray, "v n"],
    intrinsics: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
    *,
    lambda_decay: float,
) -> Float64[ndarray, "n"]:
    """Compute Kineo confidence from multi-view reprojection consistency.

    Args:
        points_xyz: Float64 world points with shape ``(n, 3)``.
        kp_xy: Float64 observed pixels with shape ``(v, n, 2)``.
        conf: Float64 detector confidences with shape ``(v, n)``.
        intrinsics: Float64 camera intrinsics with shape ``(v, 3, 3)``.
        cam_T_world: Float64 world-to-camera transforms with shape ``(v, 4, 4)``.
        lambda_decay: Exponential reprojection-error decay.

    Returns:
        Float64 point confidences with shape ``(n,)``.
    """
    n_points: int = points_xyz.shape[0]
    n_views: int = kp_xy.shape[0]
    if n_views < 2:
        return np.zeros(n_points, dtype=np.float64)

    points_homo: Float64[ndarray, "n 4"] = np.concatenate((points_xyz, np.ones((n_points, 1), dtype=np.float64)), axis=1)
    points_cam: Float64[ndarray, "v n 3"] = np.einsum("vij,nj->vni", cam_T_world[:, :3, :], points_homo)
    projected_homo: Float64[ndarray, "v n 3"] = np.einsum("vij,vnj->vni", intrinsics, points_cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_xy: Float64[ndarray, "v n 2"] = projected_homo[:, :, :2] / projected_homo[:, :, 2:3]

    focal: Float64[ndarray, "v"] = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2.0
    reproj_error: Float64[ndarray, "v n"] = np.linalg.norm(projected_xy - kp_xy, axis=2)
    normalized_error: Float64[ndarray, "v n"] = reproj_error / (focal[:, None] + 1e-8)
    valid_view: Bool[ndarray, "v n"] = (
        (points_cam[:, :, 2] > 0.0)
        & np.isfinite(projected_xy).all(axis=2)
        & np.isfinite(kp_xy).all(axis=2)
        & np.isfinite(conf)
        & (conf > 0.0)
    )
    view_score: Float64[ndarray, "v n"] = np.where(valid_view, np.exp(-lambda_decay * normalized_error), 0.0)
    valid_conf: Float64[ndarray, "v n"] = np.where(valid_view, conf, 0.0)
    pair_sum: Float64[ndarray, "n"] = np.zeros(n_points, dtype=np.float64)
    first_view_idx: int
    second_view_idx: int
    for first_view_idx in range(n_views - 1):
        for second_view_idx in range(first_view_idx + 1, n_views):
            pair_score: Float64[ndarray, "n"] = np.sqrt(view_score[first_view_idx] * view_score[second_view_idx])
            pair_weight: Float64[ndarray, "n"] = np.sqrt(valid_conf[first_view_idx] * valid_conf[second_view_idx])
            pair_sum += pair_score * pair_weight

    n_pairs: int = n_views * (n_views - 1) // 2
    point_conf: Float64[ndarray, "n"] = pair_sum / float(n_pairs)
    point_conf[np.sum(valid_view, axis=0) < 2] = 0.0
    return point_conf


def sees_body(conf: Float64[ndarray, "*batch 133"]) -> Bool[ndarray, "*batch"]:
    """Where a view sees the body: at least ``MIN_BODY_JOINTS`` COCO-17 joints above ``CONFIDENT_JOINT``."""
    return (conf[..., :BODY_JOINTS] > CONFIDENT_JOINT).sum(axis=-1) >= MIN_BODY_JOINTS


def drop_face_confidences(conf: Float64[ndarray, "v t 133"]) -> Float64[ndarray, "v t 133"]:
    """Zero the 68 COCO WholeBody face landmarks (23–90); body, feet, and hands stay."""
    selected: Float64[ndarray, "v t 133"] = conf.copy()
    selected[..., FACE_IDX] = 0.0
    return selected
