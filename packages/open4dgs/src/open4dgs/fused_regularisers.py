"""TRiGS's per-step regularisers (Bezier acceleration, rigid coherence, opacity, sigma floor, scale) as one function
(``Open4dgsTrainer``'s ``fused_regularisers``, recipe patch 0013's ``_core_regularisers`` seam): compiled, about 240
launches per step become a few fused kernels, and the dynamic rows are a mask instead of a ``nonzero`` (a host sync).

The terms are TRiGS's (``static_exclude_from_motion_losses`` / ``static_exclude_from_temporal_reg`` on): a mean over
the dynamic rows is the masked sum over every row divided by the dynamic-row count, 0 when there is none.
"""

import os

import torch
from torch import Tensor
from trigs.core.losses import rigid_coherence_loss


def core_regularisers(
    c: tuple[Tensor, ...],
    sh0: Tensor,
    opacities: Tensor,
    log_durations: Tensor,
    scales: Tensor,
    is_static: Tensor,
    gamma: Tensor,
    rigid_idx: Tensor,
    nn_indices: Tensor,
    rigid_lambda_c: float,
    rigid_detach_color: bool,
    sigma_floor_target: float,
    scale_rows: str,
    n_total_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(motion, rigid, opacity reg, sigma floor, scale) terms. ``c`` = zeta (c0_t, c0_r, c1_t, c1_r, c2_t, c2_r, base_t,
    base_r); ``scale_rows`` is ``all``, ``dynamic`` or ``off`` (no scale term)."""
    c0_t, c0_r, c1_t, c1_r, c2_t, c2_r, base_t, base_r = c
    dyn: Tensor = (is_static.reshape(-1) <= 0.5).to(c0_t.dtype)
    count: Tensor = dyn.sum().clamp_min(1.0)
    b_t: Tensor = c0_t - 2.0 * c1_t + c2_t
    b_r: Tensor = c0_r - 2.0 * c1_r + c2_r
    motion: Tensor = (dyn * ((b_t**2).sum(dim=-1) + (b_r**2).sum(dim=-1))).sum() / count
    zeta_base: Tensor = torch.cat([base_t, base_r], dim=-1)
    rigid: Tensor = rigid_coherence_loss(
        zeta_base.index_select(0, rigid_idx),
        sh0.index_select(0, rigid_idx),
        nn_indices,
        lambda_c=rigid_lambda_c,
        detach_color_weight=rigid_detach_color,
    )
    base_opa: Tensor = torch.sigmoid(opacities).reshape(-1)
    reg: Tensor = (dyn * base_opa * gamma.reshape(-1).detach()).sum() / count
    sigma_t: Tensor = torch.exp(log_durations).reshape(-1)
    sigma_floor: Tensor = (dyn * torch.clamp(sigma_floor_target - sigma_t, min=0.0).pow(2)).sum() / count
    if scale_rows == "off":
        scale: Tensor = motion.new_zeros(())
    elif scale_rows == "all":
        scale = scales.exp().sum() / float(3 * max(n_total_rows, 1))
    else:
        scale = (dyn[:, None] * scales.exp()).sum() / float(3 * max(n_total_rows, 1))
    return motion, rigid, reg, sigma_floor, scale


_COMPILED: list = []


def compiled_core_regularisers(device: torch.device):  # noqa: ANN201  # core_regularisers or its compiled graph
    """``core_regularisers`` compiled on CUDA, eager elsewhere and while beartype checks the process (``PIXI_DEV_MODE``)."""
    if device.type != "cuda" or os.environ.get("PIXI_DEV_MODE") == "1":
        return core_regularisers
    if not _COMPILED:
        _COMPILED.append(torch.compile(core_regularisers, dynamic=False))
    return _COMPILED[0]
