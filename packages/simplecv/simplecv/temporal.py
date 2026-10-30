"""Sample selection shared by temporal readers without imposing tolerance policy."""

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray


def nearest_indices(source: Float64[ndarray, "m"], target: Float64[ndarray, "n"]) -> Int64[ndarray, "n"]:
    """Select the closest sample, clipping at boundaries and choosing the left tie."""
    right = np.clip(np.searchsorted(source, target), 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(source[left] - target) <= abs(source[right] - target), left, right)
