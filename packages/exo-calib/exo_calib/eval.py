"""Similarity alignment and error metrics for camera rigs."""

from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment

from exo_calib.cameras import camera_centers

RigMode: TypeAlias = Literal["se3", "sim3"]
RIG_MODES: tuple[RigMode, ...] = get_args(RigMode)


@dataclass(slots=True)
class CameraErrors:
    """Per-camera pose errors; aggregate with plain numpy at the call site."""

    rotation_error_deg: Float64[ndarray, "v"]
    """Per-camera geodesic rotation errors in degrees."""
    translation_error_cm: Float64[ndarray, "v"]
    """Per-camera center errors in centimeters."""
    alignment_scale: float
    """Scale of the alignment applied before measuring errors."""


def align_rigs(
    pred_cam_T_world: Float64[ndarray, "v 4 4"],
    gt_cam_T_world: Float64[ndarray, "v 4 4"],
    *,
    with_scale: bool = False,
) -> SimilarityTransform:
    """Align predicted camera centers to ground truth with Umeyama's method.

    Args:
        pred_cam_T_world: Float64 predicted world-to-camera transforms with shape ``(v, 4, 4)``.
        gt_cam_T_world: Float64 ground-truth world-to-camera transforms with shape ``(v, 4, 4)``.
        with_scale: Estimate a similarity scale when true.

    Returns:
        Transform mapping predicted world coordinates into the ground-truth world frame.
    """
    return umeyama_alignment(camera_centers(pred_cam_T_world), camera_centers(gt_cam_T_world), allow_scaling=with_scale)


def camera_errors(
    pred_cam_T_world: Float64[ndarray, "v 4 4"],
    gt_cam_T_world: Float64[ndarray, "v 4 4"],
    *,
    gt_T_pred: SimilarityTransform,
) -> CameraErrors:
    """Measure aligned camera orientation and center errors.

    Args:
        pred_cam_T_world: Float64 predicted world-to-camera transforms with shape ``(v, 4, 4)``.
        gt_cam_T_world: Float64 ground-truth world-to-camera transforms with shape ``(v, 4, 4)``.
        gt_T_pred: Similarity mapping predicted world coordinates into the ground-truth world.

    Returns:
        Per-camera errors, summaries, and the applied scale.
    """
    pred_rotation: Float64[ndarray, "v 3 3"] = pred_cam_T_world[:, :3, :3]
    gt_rotation: Float64[ndarray, "v 3 3"] = gt_cam_T_world[:, :3, :3]
    aligned_pred_rotation: Float64[ndarray, "v 3 3"] = np.einsum("vij,jk->vik", pred_rotation, gt_T_pred.dst_R_src.T)
    relative_rotation: Float64[ndarray, "v 3 3"] = np.einsum(
        "vij,vkj->vik", gt_rotation, aligned_pred_rotation
    )
    rotation_cosine: Float64[ndarray, "v"] = np.clip((np.trace(relative_rotation, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation_cosine[np.isclose(rotation_cosine, 1.0, atol=1e-12)] = 1.0
    rotation_error_deg: Float64[ndarray, "v"] = np.rad2deg(np.arccos(rotation_cosine))

    pred_centers: Float64[ndarray, "v 3"] = camera_centers(pred_cam_T_world)
    gt_centers: Float64[ndarray, "v 3"] = camera_centers(gt_cam_T_world)
    aligned_pred_centers: Float64[ndarray, "v 3"] = gt_T_pred.apply(pred_centers)
    translation_error_cm: Float64[ndarray, "v"] = np.linalg.norm(gt_centers - aligned_pred_centers, axis=1) * 100.0
    translation_error_cm[np.isclose(translation_error_cm, 0.0, atol=1e-10)] = 0.0
    return CameraErrors(
        rotation_error_deg=rotation_error_deg,
        translation_error_cm=translation_error_cm,
        alignment_scale=gt_T_pred.scale,
    )


def evaluate_rig(
    pred_cam_T_world: Float64[ndarray, "v 4 4"], gt_cam_T_world: Float64[ndarray, "v 4 4"], mode: RigMode = "se3"
) -> CameraErrors:
    """Align and evaluate a predicted camera rig.

    Args:
        pred_cam_T_world: Float64 predicted world-to-camera transforms with shape ``(v, 4, 4)``.
        gt_cam_T_world: Float64 ground-truth world-to-camera transforms with shape ``(v, 4, 4)``.
        mode: Rigid ``"se3"`` or similarity ``"sim3"`` alignment.

    Returns:
        Per-camera aligned pose errors and summaries.
    """
    gt_T_pred: SimilarityTransform = align_rigs(pred_cam_T_world, gt_cam_T_world, with_scale=mode == "sim3")
    return camera_errors(pred_cam_T_world, gt_cam_T_world, gt_T_pred=gt_T_pred)


@dataclass(slots=True, frozen=True)
class RigGeometry:
    """Self-contained rig geometry in the Stage A floor frame (``z = 0`` is the fitted floor); meaningful without ground truth."""

    camera_centers_m: Float64[ndarray, "v 3"]
    """World-frame camera centres in metres."""
    pairwise_distance_m: Float64[ndarray, "v v"]
    """Distance between every pair of camera centres in metres."""
    height_above_floor_m: Float64[ndarray, "v"]
    """Camera height above the fitted floor in metres."""


def rig_geometry(cam_T_world: Float64[ndarray, "v 4 4"]) -> RigGeometry:
    """Measure the rig in its own floor frame."""
    centers: Float64[ndarray, "v 3"] = camera_centers(cam_T_world)
    pairwise_distance: Float64[ndarray, "v v"] = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    return RigGeometry(camera_centers_m=centers, pairwise_distance_m=pairwise_distance, height_above_floor_m=centers[:, 2])
