"""Foot contact evidence and soft anatomical constraints in a Y-up world."""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Float64
from numpy import ndarray
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FootContact:
    floor_height_m: float | None
    """Robust height of low, stationary, confident observed foot landmarks."""
    active: Bool[ndarray, "n 6"]
    """Contact is supported only near the floor and below 8 cm/s."""


def contact_plan(
    times: Float64[ndarray, "n"], points: Float64[ndarray, "n 6 3"], scores: Float64[ndarray, "n 6"], floor_height_m: float | None = None
) -> FootContact:
    """Estimate the low stationary surface, then gate contact without filling gaps."""
    valid = np.isfinite(points).all(-1) & (scores >= 0.7)
    speeds = np.full(valid.shape, np.inf)
    if len(times) >= 2:
        dt = np.diff(times)
        edge_valid = valid[1:] & valid[:-1] & (dt[:, None] > 0.0) & (dt[:, None] <= 0.5)
        velocity = np.linalg.norm(np.diff(points, axis=0), axis=-1) / np.maximum(dt[:, None], 1e-9)
        velocity = np.where(edge_valid, velocity, np.inf)
        speeds[0], speeds[-1] = velocity[0], velocity[-1]
        if len(times) > 2:
            speeds[1:-1] = np.maximum(velocity[:-1], velocity[1:])
    still = valid & (speeds <= 0.08)
    heights = points[..., 1][still]
    floor = floor_height_m
    if floor is None and heights.size:
        low = float(np.quantile(heights, 0.15))
        support = heights[np.abs(heights - low) <= 0.02]
        floor = float(np.median(support)) if support.size else low
    active = np.zeros_like(valid) if floor is None else still & (np.abs(points[..., 1] - floor) <= 0.03)
    return FootContact(floor, active)


def foot_pose_penalty(poses: Float[Tensor, "n 2 3"], axes: Float[Tensor, "2 3 3"], toe_joints: bool = False) -> Float[Tensor, ""]:
    """Soft local rotation-vector bounds; toes allow flex -20/+60° and yaw/roll ±10°."""
    angles = torch.einsum("nfi,fij->nfj", poses, axes)
    lower = torch.deg2rad(poses.new_tensor([-20.0, -10.0, -10.0] if toe_joints else [-35.0, -35.0, -25.0]))
    upper = torch.deg2rad(poses.new_tensor([60.0, 10.0, 10.0] if toe_joints else [50.0, 35.0, 25.0]))
    excess = torch.relu(lower - angles) + torch.relu(angles - upper)
    return 0.04 * (angles / 0.5).square().mean() + (excess / 0.1).square().mean()


def symmetry_penalty(lengths: Float[Tensor, "n 8"]) -> Float[Tensor, ""]:
    """Match paired upper arm, forearm, thigh and shin lengths in 1 cm units."""
    return ((lengths[:, 0::2] - lengths[:, 1::2]) / 0.01).square().mean()
