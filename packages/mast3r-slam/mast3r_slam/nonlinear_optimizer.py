import math

import torch
from jaxtyping import Float


def check_convergence(
    iter: int,
    rel_error_threshold: float,
    delta_norm_threshold: float,
    old_cost: float,
    new_cost: float,
    delta: Float[torch.Tensor, "..."],
    verbose: bool = False,
) -> bool:
    """Check if the optimizer has converged based on relative error and delta norm."""
    cost_diff: float = old_cost - new_cost
    rel_dec: float = math.fabs(cost_diff / old_cost)
    delta_norm: Float[torch.Tensor, ""] = torch.linalg.norm(delta)

    converged: bool = rel_dec < rel_error_threshold or delta_norm < delta_norm_threshold
    if verbose:
        print(f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}")

    return converged


def huber(r: Float[torch.Tensor, "..."], k: float = 1.345) -> Float[torch.Tensor, "..."]:
    """Compute Huber weighting function: 1 where |r| < k, else k/|r|."""
    unit: Float[torch.Tensor, "1"] = torch.ones((1), dtype=r.dtype, device=r.device)
    r_abs: Float[torch.Tensor, "..."] = torch.abs(r)
    mask: torch.Tensor = r_abs < k
    w: Float[torch.Tensor, "..."] = torch.where(mask, unit, k / r_abs)
    return w
