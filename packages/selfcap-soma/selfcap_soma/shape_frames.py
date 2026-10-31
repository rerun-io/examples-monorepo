"""Choose shape evidence using geometric support rather than detector score scale."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import serde


class InsufficientBodyObservation(ValueError):
    """Input limitation: too few usable body-anchor observations to initialize a fit."""


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
    allow_torso_only: bool = False,
) -> ShapeFrames:
    """Prefer early geometrically supported anchors, then the best whole-segment evidence.

    Confidence only ranks accepted finite evidence; it is not a detector threshold.
    Bridged points retain their reduced loss weights and rank below direct support.
    """
    anchors = [3, 4, 5, 6, 11, 12]
    available = np.isfinite(xyz[:, anchors]).all(2) & np.isfinite(confidence[:, anchors]) & (confidence[:, anchors] > 0.0)
    supported = available if support is None else available & (support[:, anchors] >= 2)
    # Two shoulders, at least one hip and at least one ear constrain the
    # upper body without requiring an occluded contralateral hip.
    finite = available[:, :2].any(1) & available[:, 2:4].all(1) & available[:, 4:].any(1)
    direct = supported[:, :2].any(1) & supported[:, 2:4].all(1) & supported[:, 4:].any(1)
    if allow_torso_only:
        finite = available[:, 2:4].all(1) & available[:, 4:].any(1)
        direct = supported[:, 2:4].all(1) & supported[:, 4:].any(1)
    early = np.flatnonzero(direct & (times < 25.0))[::8]
    if len(early) >= 64:
        return ShapeFrames(early[:96].astype(np.int64), False)
    candidates = np.flatnonzero(direct)
    if len(candidates) < 10:
        candidates = np.flatnonzero(finite)
    if len(candidates) < 3:
        raise InsufficientBodyObservation("fewer than three frames with two shoulders, one hip and one ear across the segment")
    quality = np.where(available, confidence[:, anchors], 0.0).sum(1) / len(anchors)
    if support is not None:
        quality += (support[:, anchors] >= 2).sum(1)
    best = candidates[np.argsort(-quality[candidates], kind="stable")[:96]]
    return ShapeFrames(np.sort(best).astype(np.int64), True)
