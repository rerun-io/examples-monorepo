"""Similarity alignment and error metrics for camera rigs."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

RigMode: TypeAlias = Literal["se3", "sim3"]


@dataclass(slots=True, frozen=True)
class RigAlignment:
    """Similarity transform mapping predicted world coordinates to ground truth."""

    rotation_33: Float64[ndarray, "3 3"]
    """Float64 alignment rotation with shape ``(3, 3)``."""
    translation_3: Float64[ndarray, "3"]
    """Float64 alignment translation with shape ``(3,)``."""
    scale: float
    """Alignment scale, or one for rigid alignment."""


@dataclass(slots=True)
class CameraErrors:
    """Per-camera pose errors; aggregate with plain numpy at the call site."""

    rotation_error_deg_v: Float64[ndarray, "v"]
    """Per-camera geodesic rotation errors in degrees."""
    translation_error_cm_v: Float64[ndarray, "v"]
    """Per-camera center errors in centimeters."""
    alignment_scale: float
    """Scale of the alignment applied before measuring errors."""


def align_rigs(
    pred_cam_T_world_v44: Float64[ndarray, "v 4 4"],
    gt_cam_T_world_v44: Float64[ndarray, "v 4 4"],
    *,
    with_scale: bool = False,
) -> RigAlignment:
    """Align predicted camera centers to ground truth with Umeyama's method.

    Args:
        pred_cam_T_world_v44: Float64 predicted world-to-camera transforms with shape ``(v, 4, 4)``.
        gt_cam_T_world_v44: Float64 ground-truth world-to-camera transforms with shape ``(v, 4, 4)``.
        with_scale: Estimate a similarity scale when true.

    Returns:
        Transform mapping predicted world coordinates into the ground-truth world frame.
    """
    pred_centers_v3: Float64[ndarray, "v 3"] = -np.einsum(
        "vji,vj->vi", pred_cam_T_world_v44[:, :3, :3], pred_cam_T_world_v44[:, :3, 3]
    )
    gt_centers_v3: Float64[ndarray, "v 3"] = -np.einsum("vji,vj->vi", gt_cam_T_world_v44[:, :3, :3], gt_cam_T_world_v44[:, :3, 3])
    pred_mean_3: Float64[ndarray, "3"] = np.mean(pred_centers_v3, axis=0)
    gt_mean_3: Float64[ndarray, "3"] = np.mean(gt_centers_v3, axis=0)
    pred_centered_v3: Float64[ndarray, "v 3"] = pred_centers_v3 - pred_mean_3
    gt_centered_v3: Float64[ndarray, "v 3"] = gt_centers_v3 - gt_mean_3
    covariance_33: Float64[ndarray, "3 3"] = gt_centered_v3.T @ pred_centered_v3 / float(pred_centers_v3.shape[0])
    svd_result: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"], Float64[ndarray, "3 3"]] = np.linalg.svd(covariance_33)
    correction_33: Float64[ndarray, "3 3"] = np.eye(3, dtype=np.float64)
    if np.linalg.det(svd_result[0] @ svd_result[2]) < 0.0:
        correction_33[2, 2] = -1.0
    rotation_33: Float64[ndarray, "3 3"] = svd_result[0] @ correction_33 @ svd_result[2]
    if with_scale:
        pred_variance: float = float(np.mean(np.sum(pred_centered_v3**2, axis=1)))
        scale: float = float(np.sum(svd_result[1] * np.diag(correction_33)) / pred_variance)
    else:
        scale = 1.0
    translation_3: Float64[ndarray, "3"] = gt_mean_3 - scale * (rotation_33 @ pred_mean_3)
    return RigAlignment(rotation_33=rotation_33, translation_3=translation_3, scale=scale)


def camera_errors(
    pred_cam_T_world_v44: Float64[ndarray, "v 4 4"],
    gt_cam_T_world_v44: Float64[ndarray, "v 4 4"],
    *,
    alignment: RigAlignment,
) -> CameraErrors:
    """Measure aligned camera orientation and center errors.

    Args:
        pred_cam_T_world_v44: Float64 predicted world-to-camera transforms with shape ``(v, 4, 4)``.
        gt_cam_T_world_v44: Float64 ground-truth world-to-camera transforms with shape ``(v, 4, 4)``.
        alignment: Similarity mapping from predicted world to ground-truth world.

    Returns:
        Per-camera errors, summaries, and the applied scale.
    """
    pred_rotation_v33: Float64[ndarray, "v 3 3"] = pred_cam_T_world_v44[:, :3, :3]
    gt_rotation_v33: Float64[ndarray, "v 3 3"] = gt_cam_T_world_v44[:, :3, :3]
    aligned_pred_rotation_v33: Float64[ndarray, "v 3 3"] = np.einsum("vij,jk->vik", pred_rotation_v33, alignment.rotation_33.T)
    relative_rotation_v33: Float64[ndarray, "v 3 3"] = np.einsum(
        "vij,vkj->vik", gt_rotation_v33, aligned_pred_rotation_v33
    )
    rotation_cosine_v: Float64[ndarray, "v"] = np.clip((np.trace(relative_rotation_v33, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation_cosine_v[np.isclose(rotation_cosine_v, 1.0, atol=1e-12)] = 1.0
    rotation_error_deg_v: Float64[ndarray, "v"] = np.rad2deg(np.arccos(rotation_cosine_v))

    pred_centers_v3: Float64[ndarray, "v 3"] = -np.einsum(
        "vji,vj->vi", pred_rotation_v33, pred_cam_T_world_v44[:, :3, 3]
    )
    gt_centers_v3: Float64[ndarray, "v 3"] = -np.einsum("vji,vj->vi", gt_rotation_v33, gt_cam_T_world_v44[:, :3, 3])
    aligned_pred_centers_v3: Float64[ndarray, "v 3"] = (
        alignment.scale * np.einsum("ij,vj->vi", alignment.rotation_33, pred_centers_v3) + alignment.translation_3
    )
    translation_error_cm_v: Float64[ndarray, "v"] = np.linalg.norm(gt_centers_v3 - aligned_pred_centers_v3, axis=1) * 100.0
    translation_error_cm_v[np.isclose(translation_error_cm_v, 0.0, atol=1e-10)] = 0.0
    return CameraErrors(
        rotation_error_deg_v=rotation_error_deg_v,
        translation_error_cm_v=translation_error_cm_v,
        alignment_scale=alignment.scale,
    )


def evaluate_rig(
    pred_cam_T_world_v44: Float64[ndarray, "v 4 4"], gt_cam_T_world_v44: Float64[ndarray, "v 4 4"], mode: RigMode = "se3"
) -> CameraErrors:
    """Align and evaluate a predicted camera rig.

    Args:
        pred_cam_T_world_v44: Float64 predicted world-to-camera transforms with shape ``(v, 4, 4)``.
        gt_cam_T_world_v44: Float64 ground-truth world-to-camera transforms with shape ``(v, 4, 4)``.
        mode: Rigid ``"se3"`` or similarity ``"sim3"`` alignment.

    Returns:
        Per-camera aligned pose errors and summaries.
    """
    alignment: RigAlignment = align_rigs(pred_cam_T_world_v44, gt_cam_T_world_v44, with_scale=mode == "sim3")
    return camera_errors(pred_cam_T_world_v44, gt_cam_T_world_v44, alignment=alignment)
