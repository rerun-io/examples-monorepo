"""Choose shape evidence using geometric support rather than detector score scale."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import serde


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ShapeFrames:
    indices: Int64[ndarray, "n"]
    """Chronological frame indices selected for shared shape fitting."""
    used_fallback: bool
    """Whether selection needed the whole segment."""


def select_shape_frames(
    times: Float64[ndarray, "n"],
    xyz: Float64[ndarray, "n 133 3"],
    confidence: Float64[ndarray, "n 133"],
    support: Int64[ndarray, "n 133"] | None,
) -> ShapeFrames:
    """Prefer early geometrically supported anchors, then the best whole-segment evidence.

    Confidence only ranks accepted finite evidence; it is not a detector threshold.
    Bridged points retain their reduced loss weights and rank below direct support.
    """
    anchors = [3, 4, 5, 6, 11, 12]
    finite = np.isfinite(xyz[:, anchors]).all((1, 2)) & np.isfinite(confidence[:, anchors]).all(1) & (confidence[:, anchors] > 0.0).all(1)
    direct = finite if support is None else finite & (support[:, anchors] >= 2).all(1)
    early = np.flatnonzero(direct & (times < 25.0))[::8]
    if len(early) >= 64:
        return ShapeFrames(early[:96].astype(np.int64), False)
    candidates = np.flatnonzero(direct)
    if len(candidates) < 10:
        candidates = np.flatnonzero(finite)
    if len(candidates) < 3:
        raise ValueError("shape fitting needs at least three finite, positively weighted ear/torso frames across the segment")
    quality = confidence[:, anchors].mean(1)
    if support is not None:
        quality += (support[:, anchors] >= 2).sum(1)
    best = candidates[np.argsort(-quality[candidates], kind="stable")[:96]]
    return ShapeFrames(np.sort(best).astype(np.int64), True)
