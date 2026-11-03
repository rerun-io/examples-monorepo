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
) -> Brackets:
    """Find neighbouring samples without crossing a tracking gap.

    Args:
        stamps: Float64[n] source seconds, in nondecreasing order.
        times: Float64[m] requested seconds; exact hits use the preceding interval.
        max_gap: Largest accepted source interval in seconds.

    Returns:
        Int64[m] left/right indices, Float64[m] alpha and Bool[m] validity.
        Gaps equal to max_gap are accepted. A zero-length gap uses its left sample.
        Outside times and tracks with fewer than two samples are invalid; alpha is never clamped.
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
        times - stamps[left], gap, out=np.zeros(len(times), dtype=np.float64), where=gap != 0.0
    )
    valid: Bool[ndarray, "m"] = (times >= stamps[0]) & (times <= stamps[-1]) & (gap >= 0.0) & (gap <= max_gap)
    return Brackets(left, right, alpha, valid)


def lerp(values: Float32[ndarray, "n *value"] | Float64[ndarray, "n *value"], samples: Brackets) -> Float64[ndarray, "m *value"]:
    """Interpolate numeric values; the caller applies samples.valid to its output.

    Args:
        values: Float32 or Float64[n, *value] samples, with arbitrary trailing axes.
        samples: Shared time brackets, including Float64[m] alpha.
    """
    if not len(values):
        return np.full((len(samples.alpha), *values.shape[1:]), np.nan, dtype=np.float64)
    alpha: Float64[ndarray, "m *broadcast"] = samples.alpha.reshape((-1,) + (1,) * (values.ndim - 1))
    result = (1.0 - alpha) * values[samples.left] + alpha * values[samples.right]
    return np.where(alpha == 0.0, values[samples.left], result)


def slerp(stamps: Float64[ndarray, "n"], rotations: Rotation, times: Float64[ndarray, "m"]) -> Float64[ndarray, "m 3 3"]:
    """Interpolate Float64[m,3,3] rotations without extrapolation; duplicate stamps keep the first rotation."""
    result: Float64[ndarray, "m 3 3"] = np.full((len(times), 3, 3), np.nan)
    if not len(stamps):
        return result
    keep: Bool[ndarray, "n"] = np.r_[True, np.diff(stamps) > 0.0]
    unique: Float64[ndarray, "p"] = stamps[keep]
    inside: Bool[ndarray, "m"] = (times >= unique[0]) & (times <= unique[-1])
    if len(unique) == 1:
        result[inside] = rotations.as_matrix()[keep][0]
    elif inside.any():
        result[inside] = Slerp(unique, rotations[keep])(times[inside]).as_matrix()
    return result


def interpolate_positions(
    stamps: Float64[ndarray, "n"], values: Float32[ndarray, "n *value"] | Float64[ndarray, "n *value"],
    times: Float64[ndarray, "m"], max_gap: float, valid: Bool[ndarray, "n"] | None = None,
) -> Float64[ndarray, "m *value"]:
    """Interpolate numeric samples with the shared gap policy; invalid rows are NaN.

    Args:
        stamps: Float64[n] source seconds in nondecreasing order.
        values: Float32 or Float64[n, *value] source samples.
        times: Float64[m] requested seconds.
        max_gap: Largest accepted interval in seconds, inclusive.
        valid: Bool[n] source tracking mask, or None for all samples.
    """
    samples = bracket(stamps, times, max_gap)
    usable = samples.valid
    if valid is not None and len(stamps) >= 2:
        usable = usable & valid[samples.left] & (valid[samples.right] | (stamps[samples.left] == stamps[samples.right]))
    result = lerp(values, samples)
    result[~usable] = np.nan
    return result


class PoseSamples(NamedTuple):
    """Interpolated rotations and positions, with one shared validity mask."""

    rotations: Float64[ndarray, "m 3 3"]
    """Interpolated rotations; invalid rows use identity placeholders."""
    positions: Float64[ndarray, "m 3"]
    """Interpolated positions; invalid rows are NaN."""
    valid: Bool[ndarray, "m"]
    """Times within the track whose source samples pass the gap and tracking gates."""

    @property
    def transforms(self) -> Float64[ndarray, "m 4 4"]:
        """Homogeneous transforms, with NaN in every cell of invalid rows."""
        result = np.tile(np.eye(4), (len(self.valid), 1, 1))
        result[:, :3, :3], result[:, :3, 3] = self.rotations, self.positions
        result[~self.valid] = np.nan
        return result


def interpolate_poses(
    stamps: Float64[ndarray, "n"], positions: Float32[ndarray, "n 3"] | Float64[ndarray, "n 3"],
    rotations: Rotation, times: Float64[ndarray, "m"], max_gap: float, valid: Bool[ndarray, "n"] | None = None,
) -> PoseSamples:
    """Interpolate Float32/Float64[n,3] positions and rotations on one gap-gated clock.

    Invalid rows carry NaN positions and identity rotation placeholders; use ``valid`` before fitting them.
    """
    samples = bracket(stamps, times, max_gap)
    usable = samples.valid
    if valid is not None and len(stamps) >= 2:
        usable = usable & valid[samples.left] & (valid[samples.right] | (stamps[samples.left] == stamps[samples.right]))
    translated = lerp(positions, samples)
    rotated = slerp(stamps, rotations, times)
    translated[~usable], rotated[~usable] = np.nan, np.eye(3)
    return PoseSamples(rotated, translated, usable)
