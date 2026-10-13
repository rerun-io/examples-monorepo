"""Person boxes: reprojection-driven tracking boxes and the pose model's crop rectangles."""

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from posekit.ops.crops import CropSpec, bbox_xyxy_to_center_scale


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
