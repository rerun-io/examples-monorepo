"""Bracket measured timestamps once, then interpolate positions and rotations."""

from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp


class Brackets(NamedTuple):
    """The adjacent samples and gap policy for each requested time."""

    left: Int64[ndarray, "m"]
    right: Int64[ndarray, "m"]
    alpha: Float64[ndarray, "m"]
    valid: Bool[ndarray, "m"]


def bracket(
    stamps: Float64[ndarray, "n"],
    times: Float64[ndarray, "m"],
    max_gap: float,
    *,
    inclusive: bool = True,
    denominator_floor: float = 1e-12,
    clip: bool = False,
) -> Brackets:
    """Find neighbouring samples without crossing a tracking gap.

    Args:
        stamps: Float64[n] source seconds, in nondecreasing order.
        times: Float64[m] requested seconds; exact hits use the preceding interval.
        max_gap: Largest accepted source interval in seconds.
        inclusive: Accept an interval equal to max_gap; False keeps a strict limit.
        denominator_floor: Preserve the caller's minimum interpolation denominator.
        clip: Clamp alpha to [0, 1]; validity still rejects times outside the track.

    Returns:
        Int64[m] left/right indices, Float64[m] alpha and Bool[m] validity.
        Zero-length intervals and tracks with fewer than two samples are invalid.
    """
    if len(stamps) < 2:
        return Brackets(
            np.zeros(len(times), dtype=np.int64),
            np.zeros(len(times), dtype=np.int64),
            np.zeros(len(times), dtype=np.float64),
            np.zeros(len(times), dtype=bool),
        )
    right: Int64[ndarray, "m"] = np.clip(np.searchsorted(stamps, times), 1, len(stamps) - 1)
    left: Int64[ndarray, "m"] = right - 1
    gap: Float64[ndarray, "m"] = stamps[right] - stamps[left]
    alpha: Float64[ndarray, "m"] = np.divide(
        times - stamps[left], np.maximum(gap, denominator_floor), out=np.zeros(len(times), dtype=np.float64), where=gap > 0.0
    )
    valid: Bool[ndarray, "m"] = (times >= stamps[0]) & (times <= stamps[-1]) & (gap > 0.0) & ((gap <= max_gap) if inclusive else (gap < max_gap))
    return Brackets(left, right, np.clip(alpha, 0.0, 1.0) if clip else alpha, valid)


def lerp(values: Float32[ndarray, "n *value"] | Float64[ndarray, "n *value"], samples: Brackets) -> Float64[ndarray, "m *value"]:
    """Interpolate numeric values; the caller applies samples.valid to its output.

    Args:
        values: Float32 or Float64[n, *value] samples, with arbitrary trailing axes.
        samples: Shared time brackets, including Float64[m] alpha.
    """
    if not len(values):
        return np.full((len(samples.alpha), *values.shape[1:]), np.nan, dtype=np.float64)
    alpha: Float64[ndarray, "m *broadcast"] = samples.alpha.reshape((-1,) + (1,) * (values.ndim - 1))
    return (1.0 - alpha) * values[samples.left] + alpha * values[samples.right]


def slerp(stamps: Float64[ndarray, "n"], rotations: Rotation, times: Float64[ndarray, "m"]) -> Float64[ndarray, "m 3 3"]:
    """Interpolate rotations at endpoint-clamped times; callers retain their gap gates.

    Args:
        stamps: Float64[n] source seconds, in nondecreasing order.
        rotations: Source rotations, one per stamp.
        times: Float64[m] requested seconds.

    Returns:
        Float64[m, 3, 3] matrices. Repeated stamps keep their first rotation.
    """
    if not len(stamps):
        return np.tile(np.eye(3), (len(times), 1, 1))
    keep: Bool[ndarray, "n"] = np.r_[True, np.diff(stamps) > 0.0]
    unique: Float64[ndarray, "p"] = stamps[keep]
    if len(unique) == 1:
        return np.repeat(rotations.as_matrix()[keep], len(times), axis=0)
    return Slerp(unique, rotations[keep])(np.clip(times, unique[0], unique[-1])).as_matrix()
