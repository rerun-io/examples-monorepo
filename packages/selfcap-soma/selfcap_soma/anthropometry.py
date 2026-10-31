"""Broad adult bounds used to reject impossible input geometry."""

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray

SCALE_MIN = 0.85
SCALE_MAX = 1.25


def plausible_ears(points: Float64[ndarray, "n 133 3"]) -> Bool[ndarray, "n"]:
    """Accept a pair only when its finite distance is within 110–200 mm."""
    span = np.linalg.norm(points[:, 3] - points[:, 4], axis=-1)
    return np.isfinite(span) & (span >= 0.110) & (span <= 0.200)



def usable_ears(points: Float64[ndarray, "n 133 3"], confidence: Float64[ndarray, "n 133"]) -> Bool[ndarray, "n"]:
    """Plausible ear pairs with at least one ear detected."""
    return plausible_ears(points) & (confidence[:, 3:5] > 0.0).any(1)

def bounded_scale(scale: float) -> float:
    """Preserve an interior scale exactly; clamp only outside adult bounds."""
    return min(SCALE_MAX, max(SCALE_MIN, scale))
