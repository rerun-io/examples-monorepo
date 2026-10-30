"""Per-view evidence classification; inferred geometry remains available."""

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray


def observed_points(
    projected: Float64[ndarray, "n j 2"],
    detected: Float64[ndarray, "n j 2"],
    scores: Float64[ndarray, "n j"],
    max_distance_px: float | Float64[ndarray, "n j"] = 15.0,
) -> Bool[ndarray, "n j"]:
    """A nearby confident image measurement supports an observed marker."""
    return (
        (scores >= 0.7)
        & np.isfinite(projected).all(-1)
        & np.isfinite(detected).all(-1)
        & (np.linalg.norm(projected - detected, axis=-1) <= max_distance_px)
    )


def observed_links(observed: Bool[ndarray, "n j"], edges: Int64[ndarray, "e 2"]) -> Bool[ndarray, "n e"]:
    """A link is observed only when both of its endpoints are supported."""
    return observed[:, edges].all(-1)


def residual_tolerance(
    fitted: Float64[ndarray, "n j 3"],
    reference: Float64[ndarray, "n j 3"],
    depth: Float64[ndarray, "n j"],
    focal_px: float,
) -> Float64[ndarray, "n j"]:
    """Project twice each joint's median 3D residual at its camera depth.

    Use the larger pinhole focal length for a conservative image-plane radius.
    Missing residuals fall back to 15 px; missing detections never become evidence.
    """
    distances: Float64[ndarray, "n j"] = np.linalg.norm(fitted - reference, axis=-1)
    medians: Float64[ndarray, "j"] = np.zeros(fitted.shape[1], dtype=np.float64)
    for joint in range(fitted.shape[1]):
        valid: Float64[ndarray, "n"] = distances[np.isfinite(distances[:, joint]), joint]
        if len(valid):
            medians[joint] = np.median(valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius: Float64[ndarray, "n j"] = 2.0 * focal_px * medians[None, :] / depth
    return np.where(np.isfinite(radius) & (depth > 0.001), np.maximum(15.0, radius), 15.0)
