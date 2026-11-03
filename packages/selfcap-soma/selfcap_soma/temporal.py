"""Timestamp-aware 24 Hz second-difference diagnostics for fitted joint groups."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float32, Float64
from numpy import ndarray
from serde import serde


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class GroupJitter:
    name: str
    """Joint group name."""
    median_mm: float
    """Median 24 Hz equivalent second-difference magnitude."""
    p90_mm: float
    """90th percentile magnitude."""


def second_difference_mm(points: Float32[ndarray, "n j 3"], times: Float64[ndarray, "n"]) -> Float64[ndarray, "m j"]:
    """Return acceleration scaled to one 24 Hz second difference; ignore gaps."""
    dt = np.diff(times)
    velocity = np.diff(points.astype(np.float64), axis=0) / np.maximum(dt[:, None, None], 1e-6)
    acceleration = 2.0 * np.diff(velocity, axis=0) / np.maximum((dt[1:] + dt[:-1])[:, None, None], 1e-6)
    result = np.linalg.norm(acceleration, axis=-1) * (1000.0 / 24.0**2)
    valid = (dt[1:] > 0) & (dt[1:] <= 0.1) & (dt[:-1] > 0) & (dt[:-1] <= 0.1)
    result[~valid] = np.nan
    return result


def group_jitter(points: Float32[ndarray, "n j 3"], times: Float64[ndarray, "n"], names: tuple[str, ...]) -> list[GroupJitter]:
    """Summarize root, spine, shoulder chain, arms and finger positions separately."""
    delta = second_difference_mm(points, times)
    groups = {
        "pelvis": ("Hips",),
        "spine_neck": ("Spine1", "Spine2", "Chest", "Neck1", "Neck2"),
        "shoulders": ("LeftShoulder", "RightShoulder", "LeftArm", "RightArm"),
        "arms_wrists": ("LeftForeArm", "RightForeArm", "LeftHand", "RightHand"),
        "fingers": tuple(name for name in names if "Hand" in name and name not in ("LeftHand", "RightHand")),
    }
    rows = []
    for label, members in groups.items():
        values = delta[:, [names.index(name) for name in members]]
        rows.append(GroupJitter(label, float(np.nanmedian(values)), float(np.nanpercentile(values, 90))))
    return rows
