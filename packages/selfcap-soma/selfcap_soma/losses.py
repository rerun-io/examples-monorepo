"""Masked source-normalized robust fitting terms."""

import torch
from jaxtyping import Float
from torch import Tensor


def attached_eye(head: Float[Tensor, "n 3"], rotation: Float[Tensor, "n 3 3"], offset: Float[Tensor, "3"]) -> Float[Tensor, "n 3"]:
    """World eye position from a constant attachment in the Quest head frame."""
    return head + torch.einsum("nij,j->ni", rotation, offset)


def body_lengths(body: Float[Tensor, "n 12 3"]) -> Float[Tensor, "n 11"]:
    """Widths, midpoint torso and paired limb lengths in COCO 5–16 order."""
    vectors = [body[:, 0] - body[:, 1], body[:, 6] - body[:, 7], body[:, :2].mean(1) - body[:, 6:8].mean(1)]
    vectors.extend(body[:, a] - body[:, b] for a, b in ((0, 2), (1, 3), (2, 4), (3, 5), (6, 8), (7, 9), (8, 10), (9, 11)))
    return torch.stack([torch.linalg.vector_norm(v, dim=-1) for v in vectors], dim=-1)


def robust_points(
    predicted: Float[Tensor, "n j 3"], target: Float[Tensor, "n j 3"], weights: Float[Tensor, "n j"], scale: float
) -> Float[Tensor, ""]:
    """Huber loss on distances in units of scale, with transition at two."""
    valid = torch.isfinite(target).all(dim=-1) & (weights > 0)
    delta = torch.where(valid[..., None], predicted - torch.nan_to_num(target), 0.0)
    distance = torch.linalg.vector_norm(delta, dim=-1) / scale
    cost = torch.where(distance <= 2.0, 0.5 * distance.square(), 2.0 * (distance - 1.0))
    effective = torch.where(valid, weights, 0.0)
    return (cost * effective).sum() / effective.sum().clamp_min(1.0)


def rotation_temporal_penalty(rotations: Float[Tensor, "n k 3 3"], dt: Float[Tensor, "interval"]) -> Float[Tensor, ""]:
    """Penalize rapid changes in world rotation velocity, retaining smooth turns."""
    velocity = (rotations[1:] - rotations[:-1]) / dt[:, None, None, None].clamp_min(1e-6)
    change = (velocity[1:] - velocity[:-1]) / 0.5
    valid = (dt[1:] > 0) & (dt[1:] <= 0.1) & (dt[:-1] > 0) & (dt[:-1] <= 0.1)
    return (change.square().mean(dim=(1, 2, 3)) * valid).sum() / valid.sum().clamp_min(1)


def body_acceleration_penalty(points: Float[Tensor, "n j 3"], dt: Float[Tensor, "interval"]) -> Float[Tensor, ""]:
    """Body-only world acceleration, normalized to a 4 mm second difference at 24 Hz.

    Actual intervals handle irregular sampling. Gaps above 100 ms do not couple.
    """
    velocity = (points[1:] - points[:-1]) / dt[:, None, None].clamp_min(1e-6)
    acceleration = 2.0 * (velocity[1:] - velocity[:-1]) / (dt[1:] + dt[:-1])[:, None, None].clamp_min(1e-6)
    valid = (dt[1:] > 0) & (dt[1:] <= 0.1) & (dt[:-1] > 0) & (dt[:-1] <= 0.1)
    normalized = acceleration / (24.0**2 * 0.004)
    return (normalized.square().sum(-1).mean(-1) * valid).sum() / valid.sum().clamp_min(1)
