"""3D hand metrics as the EgoHand paper defines them (arXiv 2610.30293, Sec. 6.1).

- MPJPE: wrist-relative joint error in the camera frame, no rotation or scale alignment.
- PA-MPJPE: joint error after a per-hand similarity (scale, rotation, translation) alignment.
- RTE: wrist (root) distance in camera coordinates.
- 2D EPE: pixel distance between projected predicted joints and the ground-truth projections.
Each metric is averaged over a hand's valid joints, then summarised over hands as mean and p90.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Float64
from numpy import ndarray
from serde import serde
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment


@dataclass(frozen=True, slots=True)
class HandErrors:
    """Errors of one predicted hand against its ground truth."""

    mpjpe_mm: float
    """Mean wrist-relative joint error, millimetres."""
    pa_mpjpe_mm: float
    """Mean joint error after similarity alignment, millimetres."""
    rte_mm: float
    """Wrist distance in camera coordinates, millimetres."""
    epe_2d_px: float
    """Mean projected joint error, pixels."""


def hand_errors(
    pred_xyz: Float[ndarray, "21 3"],
    gt_xyz: Float[ndarray, "21 3"],
    pred_uv: Float[ndarray, "21 2"],
    gt_uv: Float[ndarray, "21 2"],
    valid: Bool[ndarray, "21"],
) -> HandErrors:
    """Score one hand; joint 0 is the wrist (root) of both skeletons.

    Args:
        pred_xyz: Predicted joints in the camera frame, metres.
        gt_xyz: Ground-truth joints in the camera frame, metres.
        pred_uv: Predicted joints in source pixels.
        gt_uv: Ground-truth joints in source pixels.
        valid: Joints to score (both 3D and 2D).

    Returns:
        The four per-hand errors.
    """
    pred: Float64[ndarray, "21 3"] = pred_xyz.astype(np.float64)
    gt: Float64[ndarray, "21 3"] = gt_xyz.astype(np.float64)
    relative_error: Float64[ndarray, "j"] = np.linalg.norm((pred - pred[0])[valid] - (gt - gt[0])[valid], axis=1)
    alignment: SimilarityTransform = umeyama_alignment(pred[valid], gt[valid], allow_scaling=True)
    aligned_error: Float64[ndarray, "j"] = np.linalg.norm(alignment.apply(pred[valid]) - gt[valid], axis=1)
    pixel_error: Float64[ndarray, "j"] = np.linalg.norm(pred_uv.astype(np.float64)[valid] - gt_uv.astype(np.float64)[valid], axis=1)
    return HandErrors(
        mpjpe_mm=float(relative_error.mean() * 1000.0),
        pa_mpjpe_mm=float(aligned_error.mean() * 1000.0),
        rte_mm=float(np.linalg.norm(pred[0] - gt[0]) * 1000.0),
        epe_2d_px=float(pixel_error.mean()),
    )


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Summary:
    """Mean and 90th percentile of a per-hand error; ``None`` when no hand was scored."""

    mean: float | None
    """Mean over hands."""
    p90: float | None
    """90th percentile over hands (linear interpolation)."""
    count: int
    """Number of hands."""


def summarize(values: Float[ndarray, "n"]) -> Summary:
    """Mean and p90 of per-hand errors."""
    if not len(values):
        return Summary(None, None, 0)
    return Summary(float(np.mean(values)), float(np.quantile(values, 0.9)), int(len(values)))


def box_iou(a: Float[ndarray, "4"], b: Float[ndarray, "4"]) -> float:
    """Intersection over union of two ``xyxy`` boxes."""
    low: Float64[ndarray, "2"] = np.maximum(a[:2], b[:2]).astype(np.float64)
    high: Float64[ndarray, "2"] = np.minimum(a[2:], b[2:]).astype(np.float64)
    intersection: float = float(np.prod(np.clip(high - low, 0.0, None)))
    union: float = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - intersection
    return intersection / union if union > 0.0 else 0.0
