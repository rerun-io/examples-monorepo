"""Per-view evidence classification; inferred geometry remains available."""

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray


def observed_points(
    projected: Float64[ndarray, "n j 2"], detected: Float64[ndarray, "n j 2"], scores: Float64[ndarray, "n j"], max_distance_px: float = 15.0
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
