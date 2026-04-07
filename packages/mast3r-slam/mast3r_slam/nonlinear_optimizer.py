import math

import rerun as rr
import torch
from jaxtyping import Bool, Float
from torch import Tensor


def check_convergence(
    iter: int,
    rel_error_threshold: float,
    delta_norm_threshold: float,
    old_cost: float,
    new_cost: float,
    delta: Float[Tensor, "..."],
    verbose: bool = False,
) -> bool:
    """Check if the optimizer has converged based on relative error and delta norm."""
    cost_diff: float = old_cost - new_cost
    rel_dec: float = math.inf
    if math.isfinite(old_cost) and old_cost > 0.0:
        rel_dec = math.fabs(cost_diff / old_cost)
    delta_norm: Float[Tensor, ""] = torch.linalg.norm(delta)

    converged: bool = rel_dec < rel_error_threshold or bool(delta_norm < delta_norm_threshold)
    if verbose:
        rr.log("/world/logs", rr.TextLog(f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}", level="DEBUG"))

    return converged


def huber(r: Float[Tensor, "..."], k: float = 1.345) -> Float[Tensor, "..."]:
    """Compute Huber weighting function: 1 where |r| < k, else k/|r|."""
    unit: Float[Tensor, "1"] = torch.ones((1), dtype=r.dtype, device=r.device)
    r_abs: Float[Tensor, "..."] = torch.abs(r)
    mask: Bool[Tensor, "..."] = r_abs < k
    w: Float[Tensor, "..."] = torch.where(mask, unit, k / r_abs)
    return w
