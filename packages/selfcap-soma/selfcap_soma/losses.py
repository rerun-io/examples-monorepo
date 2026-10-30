"""Masked source-normalized robust fitting terms."""

import torch
from jaxtyping import Float
from torch import Tensor


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
