"""Person boxes: reprojection-driven tracking boxes, one person's detector box per view, and the pose model's crop rectangles."""

from itertools import combinations

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from posekit.ops.crops import CropSpec, bbox_xyxy_to_center_scale
from simplecv.ops.triangulate import triangulate_weighted

PERSON_FIT: float = 1.0
"""Before a person is tracked, a view supports a hypothesis that projects within this many box half-sizes of a candidate's centre."""


def boxes_from_projected_keypoints(
    points_xyz: Float64[ndarray, "j 3"],
    conf: Float64[ndarray, "j"],
    camera_intrinsics: Float64[ndarray, "3 3"],
    camera_T_world: Float64[ndarray, "4 4"],
    image_wh: tuple[int, int],
    *,
    pad: float,
    min_joints: int,
    min_confidence: float,
) -> Float64[ndarray, "4"]:
    """Project a 3D skeleton into one view and return a padded XYXY box.

    Args:
        points_xyz: Float64 world keypoints with shape ``(j, 3)``.
        conf: Float64 keypoint confidences with shape ``(j,)``.
        camera_intrinsics: Float64 camera intrinsics with shape ``(3, 3)``.
        camera_T_world: Float64 world-to-camera transform with shape ``(4, 4)``.
        image_wh: Image width and height in pixels.
        pad: Multiplicative width and height padding about the box center.
        min_joints: Minimum confident, projectable joints inside the image.
        min_confidence: Inclusive confidence threshold for projected joints.

    Returns:
        Float64 XYXY box with shape ``(4,)``, clipped to the image, or NaNs
        when too few joints can define a usable box.
    """
    if pad <= 0.0:
        raise ValueError("pad must be positive")
    if min_joints < 1:
        raise ValueError("min_joints must be positive")
    image_w: int = image_wh[0]
    image_h: int = image_wh[1]
    if image_w < 1 or image_h < 1:
        raise ValueError("image dimensions must be positive")

    points_homo: Float64[ndarray, "j 4"] = np.column_stack(
        (points_xyz, np.ones(points_xyz.shape[0], dtype=np.float64))
    )
    points_cam: Float64[ndarray, "j 3"] = np.einsum("ij,nj->ni", camera_T_world[:3], points_homo)
    projected_homo: Float64[ndarray, "j 3"] = np.einsum("ij,nj->ni", camera_intrinsics, points_cam)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_xy: Float64[ndarray, "j 2"] = projected_homo[:, :2] / projected_homo[:, 2:3]
    usable: Bool[ndarray, "j"] = (
        np.isfinite(points_xyz).all(axis=1)
        & np.isfinite(conf)
        & (conf >= min_confidence)
        & (points_cam[:, 2] > 0.0)
        & np.isfinite(projected_xy).all(axis=1)
        & (projected_xy[:, 0] >= 0.0)
        & (projected_xy[:, 0] < float(image_w))
        & (projected_xy[:, 1] >= 0.0)
        & (projected_xy[:, 1] < float(image_h))
    )
    if int(usable.sum()) < min_joints:
        return np.full(4, np.nan, dtype=np.float64)

    usable_xy: Float64[ndarray, "m 2"] = projected_xy[usable]
    minimum_xy: Float64[ndarray, "2"] = usable_xy.min(axis=0)
    maximum_xy: Float64[ndarray, "2"] = usable_xy.max(axis=0)
    center_xy: Float64[ndarray, "2"] = (minimum_xy + maximum_xy) / 2.0
    padded_size_wh: Float64[ndarray, "2"] = (maximum_xy - minimum_xy) * pad
    box_xyxy: Float64[ndarray, "4"] = np.concatenate(
        (center_xy - padded_size_wh / 2.0, center_xy + padded_size_wh / 2.0)
    )
    box_xyxy[[0, 2]] = np.clip(box_xyxy[[0, 2]], 0.0, float(image_w))
    box_xyxy[[1, 3]] = np.clip(box_xyxy[[1, 3]], 0.0, float(image_h))
    if box_xyxy[2] <= box_xyxy[0] or box_xyxy[3] <= box_xyxy[1]:
        return np.full(4, np.nan, dtype=np.float64)
    return box_xyxy


def boxes_needing_detection(
    boxes_xyxy: Float64[ndarray, "v 4"], image_wh: tuple[int, int]
) -> Bool[ndarray, "v"]:
    """Mark missing or image-boundary projected boxes for detector re-anchoring.

    Args:
        boxes_xyxy: Float64 projected XYXY boxes with shape ``(v, 4)``.
        image_wh: Image width and height in pixels.

    Returns:
        Boolean mask with shape ``(v,)``. A clipped box touching any image
        boundary is marked because its unpadded projection left the image.
    """
    image_w: int = image_wh[0]
    image_h: int = image_wh[1]
    finite_mask: Bool[ndarray, "v"] = np.isfinite(boxes_xyxy).all(axis=1)
    touches_boundary: Bool[ndarray, "v"] = (
        (boxes_xyxy[:, 0] <= 0.0)
        | (boxes_xyxy[:, 1] <= 0.0)
        | (boxes_xyxy[:, 2] >= float(image_w))
        | (boxes_xyxy[:, 3] >= float(image_h))
    )
    return ~finite_mask | touches_boundary


def box_iou(first_xyxy: Float64[ndarray, "n 4"], second_xyxy: Float64[ndarray, "n 4"]) -> Float64[ndarray, " n"]:
    """Row-wise intersection over union of two XYXY box arrays; NaN where either box is missing (NaN) or both are empty."""
    overlap: Float64[ndarray, " n"] = np.clip(np.minimum(first_xyxy[:, 2:], second_xyxy[:, 2:]) - np.maximum(first_xyxy[:, :2], second_xyxy[:, :2]), 0.0, None).prod(axis=1)
    union: Float64[ndarray, " n"] = (first_xyxy[:, 2:] - first_xyxy[:, :2]).prod(axis=1) + (second_xyxy[:, 2:] - second_xyxy[:, :2]).prod(axis=1) - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        return overlap / union


def extrapolate_keypoints(
    latest_points_xyz: Float64[ndarray, "j 3"],
    previous_points_xyz: Float64[ndarray, "j 3"],
    *,
    step_ratio: float = 1.0,
) -> Float64[ndarray, "j 3"]:
    """Extrapolate a skeleton with constant velocity from its last two states.

    Args:
        latest_points_xyz: Float64 latest world keypoints with shape ``(j, 3)``.
        previous_points_xyz: Float64 preceding world keypoints with shape ``(j, 3)``.
        step_ratio: Prediction interval divided by the two-state interval.

    Returns:
        Float64 predicted world keypoints with shape ``(j, 3)``. Joints with
        only a latest state are carried forward; joints absent from the latest
        state remain NaN.
    """
    if step_ratio < 0.0:
        raise ValueError("step_ratio must be nonnegative")
    predicted_points_xyz: Float64[ndarray, "j 3"] = latest_points_xyz.copy()
    velocity_available: Bool[ndarray, "j"] = np.isfinite(latest_points_xyz).all(axis=1) & np.isfinite(previous_points_xyz).all(axis=1)
    predicted_points_xyz[velocity_available] = latest_points_xyz[velocity_available] + step_ratio * (
        latest_points_xyz[velocity_available] - previous_points_xyz[velocity_available]
    )
    return predicted_points_xyz


def vote_person(
    boxes_xyxy: Float64[ndarray, "n 4"], scores: Float64[ndarray, " n"], view_of: Int64[ndarray, " n"], projection: Float64[ndarray, "v 3 4"]
) -> Float64[ndarray, "3"] | None:
    """The person to track when none is tracked yet: the 3D point that the most views' candidate box centres agree on (within
    ``PERSON_FIT`` half-sizes), triangulated from every pair of candidates (pinhole pixels; ``view_of`` names each row's view)
    in two views; None when no pair supports any point.

    Among equal support the tighter fit wins (two people at the cameras' height share every epipolar line, so a point between
    them fits both loosely), then the higher summed score. Picking the top-scoring box per view instead puts different people
    in different views.
    """
    n_views: int = projection.shape[0]
    centers: Float64[ndarray, "n 2"] = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2.0
    pairs = [(a, b) for va, vb in combinations(range(n_views), 2) for a in np.flatnonzero(view_of == va).tolist() for b in np.flatnonzero(view_of == vb).tolist()]
    if not pairs:
        return None
    first, second = np.array(pairs, dtype=np.int64).T
    hypotheses: Int64[ndarray, " h"] = np.arange(len(pairs))
    xy: Float64[ndarray, "v h 2"] = np.full((n_views, len(pairs), 2), np.nan)
    xy[view_of[first], hypotheses], xy[view_of[second], hypotheses] = centers[first], centers[second]
    points, valid = triangulate_weighted(torch.from_numpy(xy), torch.from_numpy(np.isfinite(xy[..., 0]).astype(np.float64)), torch.from_numpy(projection))
    points_xyz: Float64[ndarray, "h 3"] = points[valid].numpy()
    rows, distances = nearest_candidates(points_xyz, boxes_xyxy, view_of, projection)
    fits: Bool[ndarray, "h v"] = distances < PERSON_FIT
    support = [(int(fit.sum()), -float(distance[fit].sum()), float(scores[row[fit]].sum())) for fit, distance, row in zip(fits, distances, rows, strict=True)]
    best: int = max(range(len(support)), key=support.__getitem__, default=-1)
    return points_xyz[best] if best >= 0 and support[best][0] > 0 else None


def skeleton_centre(points_xyz: Float64[ndarray, "j 3"], conf: Float64[ndarray, " j"], min_confidence: float) -> Float64[ndarray, "3"] | None:
    """The median of a skeleton's finite joints at or above ``min_confidence``; None without any."""
    usable: Bool[ndarray, " j"] = (conf >= min_confidence) & np.isfinite(points_xyz).all(axis=1)
    return np.median(points_xyz[usable], axis=0) if usable.any() else None


def nearest_candidates(
    points_xyz: Float64[ndarray, "h 3"], boxes_xyxy: Float64[ndarray, "n 4"], view_of: Int64[ndarray, " n"], projection: Float64[ndarray, "v 3 4"]
) -> tuple[Int64[ndarray, "h v"], Float64[ndarray, "h v"]]:
    """Per 3D point, each view's candidate box whose centre lies nearest the point's projection, in box half-sizes: its row (-1:
    no candidate, or the point is behind the camera) and that distance (inf).

    With the tracked person as the point this is the one-person rule, and nearest wins with no cut: the rig is what is being
    calibrated, so a badly placed camera's projection may miss the person's own box, yet the person is still the nearer one
    there (a hard cut starved Ego-Exo4D's top-down camera of every detection). A view that sees only another person takes that
    person; the tracker's view gate and refine's robust steps weigh such views down.
    """
    n_points, n_views = len(points_xyz), projection.shape[0]
    centers: Float64[ndarray, "n 2"] = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) / 2.0
    half_sizes: Float64[ndarray, "n 2"] = (boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]) / 2.0
    uvw: Float64[ndarray, "h n 3"] = np.einsum("vij,hj->hvi", projection, np.column_stack((points_xyz, np.ones(n_points))))[:, view_of]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance: Float64[ndarray, "h n"] = np.linalg.norm((centers - uvw[..., :2] / uvw[..., 2:]) / half_sizes, axis=-1)
    distance[uvw[..., 2] <= 0.0] = np.inf
    rows: Int64[ndarray, "h v"] = np.full((n_points, n_views), -1, dtype=np.int64)
    distances: Float64[ndarray, "h v"] = np.full((n_points, n_views), np.inf)
    for view in range(n_views):
        mine: Int64[ndarray, " k"] = np.flatnonzero(view_of == view)
        if mine.size:
            best: Int64[ndarray, " h"] = distance[:, mine].argmin(axis=1)
            distances[:, view] = distance[:, mine][np.arange(n_points), best]
            rows[:, view] = np.where(np.isfinite(distances[:, view]), mine[best], -1)
    return rows, distances


def crop_rectangles(bbox_xyxy: Float32[ndarray, "t 4"], crop_spec: CropSpec) -> tuple[Float32[ndarray, "t 2"], Float32[ndarray, "t 2"]]:
    """Crop rectangles exactly as posekit prepares them (center/scale convention); NaN rows where there is no box.

    Args:
        bbox_xyxy: Per-frame person boxes; NaN where no detection.
        crop_spec: The pose model's crop input size and padding.

    Returns:
        Image-space crop origins and sizes, index-aligned with ``bbox_xyxy``.
    """
    valid_box: Bool[ndarray, "t"] = np.isfinite(bbox_xyxy).all(axis=1)
    crop_origin_xy: Float32[ndarray, "t 2"] = np.full((bbox_xyxy.shape[0], 2), np.nan, dtype=np.float32)
    crop_size_wh: Float32[ndarray, "t 2"] = np.full((bbox_xyxy.shape[0], 2), np.nan, dtype=np.float32)
    if valid_box.any():
        crop_w, crop_h = crop_spec.input_size
        centers, scales = bbox_xyxy_to_center_scale(
            torch.as_tensor(bbox_xyxy[valid_box]), aspect_wh=float(crop_w) / float(crop_h), padding=crop_spec.padding
        )
        crop_scale_wh: Float32[ndarray, "m 2"] = scales.cpu().numpy()
        crop_origin_xy[valid_box] = centers.cpu().numpy() - crop_scale_wh / 2.0
        crop_size_wh[valid_box] = crop_scale_wh
    return crop_origin_xy, crop_size_wh
