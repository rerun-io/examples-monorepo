"""Observation-count invariant robust objective blocks for exo calibration."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import serde


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ObjectiveShares:
    headset: float = 1.0
    """Relative total weight of headset anchors, equally split between views."""
    phones: float = .25
    """Relative total weight of static phone anchors, equally split between views."""
    hands: float = 1.5
    """Relative total weight of Quest/image hand matches, equally split between views."""
    body: float = .5
    """Relative total weight of image body matches, equally split between views."""

    def __post_init__(self) -> None:
        values = (self.headset, self.phones, self.hands, self.body)
        if not all(np.isfinite(x) and x >= 0.0 for x in values) or sum(values) <= 0.0:
            raise ValueError('Objective shares must be finite, nonnegative, and nonzero in total')


def block_weights(views: Int64[ndarray, "n"], confidence: Float64[ndarray, "n"], share: float) -> Float64[ndarray, "n"]:
    """Assign a fixed block share, with equal view shares and normalized confidence."""
    output = np.zeros(len(views))
    active = [view for view in np.unique(views) if np.sum(confidence[views == view]) > 0.0]
    for view in active:
        chosen = views == view
        output[chosen] = share / len(active) * confidence[chosen] / confidence[chosen].sum()
    return output


def robust_loss(weights: Float64[ndarray, "n"], kind: str = 'cauchy') -> Callable[[Float64[ndarray, "n"]], Float64[ndarray, "3 n"]]:
    """Weight robust costs and derivatives, preserving the pixel-scale outlier threshold."""
    def loss(z: Float64[ndarray, "n"]) -> Float64[ndarray, "3 n"]:
        if kind == 'huber':
            large = z > 1.0
            safe = np.maximum(z, 1.0)
            rho = np.stack([np.where(large, 2.0 * np.sqrt(safe) - 1.0, z),
                            np.where(large, 1.0 / np.sqrt(safe), 1.0),
                            np.where(large, -.5 / safe**1.5, 0.0)])
        else:
            rho = np.stack([np.log1p(z), 1.0 / (1.0 + z), -1.0 / (1.0 + z)**2])
        return rho * weights
    return loss
