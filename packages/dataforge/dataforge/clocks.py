"""Source clocks: nominal frame times and nearest-frame association for secondary timelines."""

import numpy as np
from jaxtyping import Int64
from numpy import ndarray


def frame_times(frames: Int64[ndarray, "n"], fps: int) -> Int64[ndarray, "n"]:
    """Nearest nanosecond of each frame index on a nominal-rate clock."""
    return np.rint(frames / fps * 1e9).astype(np.int64)


def nearest_framesets(primary: Int64[ndarray, "f"], times_ns: Int64[ndarray, "n"]) -> Int64[ndarray, "n"]:
    """Associate labels with primary frames for the secondary timeline; ties go left."""
    right: Int64[ndarray, "n"] = np.clip(np.searchsorted(primary, times_ns), 0, len(primary) - 1)
    left: Int64[ndarray, "n"] = np.maximum(right - 1, 0)
    return np.where(np.abs(times_ns - primary[left]) <= np.abs(times_ns - primary[right]), left, right).astype(np.int64)
