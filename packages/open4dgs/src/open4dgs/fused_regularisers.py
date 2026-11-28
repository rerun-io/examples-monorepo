"""TRiGS's per-step regularisers (Bezier acceleration, rigid coherence, opacity, sigma floor, scale) as one function
(``Open4dgsTrainer``'s ``fused_regularisers``, recipe patch 0013's ``_core_regularisers`` seam): compiled, about 240
launches per step become a few fused kernels, and the dynamic rows are a mask instead of a ``nonzero`` (a host sync).

The terms are TRiGS's (``static_exclude_from_motion_losses`` / ``static_exclude_from_temporal_reg`` on): a mean over
the dynamic rows is the masked sum over every row divided by the dynamic-row count, 0 when there is none.
"""

import functools
import os
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor
from trigs.core.losses import rigid_coherence_loss

ZETA_KEYS: tuple[str, ...] = ("zeta_c0_t", "zeta_c0_r", "zeta_c1_t", "zeta_c1_r", "zeta_c2_t", "zeta_c2_r", "zeta_base_t", "zeta_base_r")
"""The motion parameters ``core_regularisers`` reads as ``c``, in its order."""

ScaleRows = Literal["all", "dynamic", "off"]
"""Rows of the scale term: every row (``scale_reg_include_static``), the dynamic rows, or no scale term."""

RegulariserTerms = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


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
    scale_rows: ScaleRows,
) -> RegulariserTerms:
    """(motion, rigid, opacity reg, sigma floor, scale) terms. ``c`` = the ``ZETA_KEYS`` parameters; the scale term is
    normalised by the whole model's rows (the seam excludes chunked training)."""
    dyn: Tensor = ~(is_static.reshape(-1) > 0.5)  # TRiGS's _dynamic_rows (a NaN flag is a dynamic row)
    dyn_col: Tensor = dyn[:, None]

    # The loop gathers the dynamic rows before the nonlinear terms; here excluded rows read 0 instead (where, not a
    # multiply), so an excluded inf / NaN reaches neither a term nor a gradient.
    def dynamic_only(x: Tensor) -> Tensor:
        return torch.where(dyn_col if x.dim() == 2 else dyn, x, 0.0)

    c0_t, c0_r, c1_t, c1_r, c2_t, c2_r = (dynamic_only(x) for x in c[:6])
    base_t, base_r = c[6], c[7]
    weight: Tensor = dyn.to(c0_t.dtype)
    count: Tensor = weight.sum().clamp_min(1.0)
    b_t: Tensor = c0_t - 2.0 * c1_t + c2_t
    b_r: Tensor = c0_r - 2.0 * c1_r + c2_r
    motion: Tensor = (weight * ((b_t**2).sum(dim=-1) + (b_r**2).sum(dim=-1))).sum() / count
    zeta_base: Tensor = torch.cat([base_t, base_r], dim=-1)
    rigid: Tensor = rigid_coherence_loss(
        zeta_base.index_select(0, rigid_idx),
        sh0.index_select(0, rigid_idx),
        nn_indices,
        lambda_c=rigid_lambda_c,
        detach_color_weight=rigid_detach_color,
    )
    base_opa: Tensor = torch.sigmoid(dynamic_only(opacities.reshape(-1)))
    reg: Tensor = (weight * base_opa * dynamic_only(gamma.reshape(-1).detach())).sum() / count
    sigma_t: Tensor = torch.exp(dynamic_only(log_durations.reshape(-1)))
    sigma_floor: Tensor = (weight * torch.clamp(sigma_floor_target - sigma_t, min=0.0).pow(2)).sum() / count
    if scale_rows == "off":
        return motion, rigid, reg, sigma_floor, motion.new_zeros(())
    denominator: float = float(3 * max(int(scales.shape[0]), 1))
    if scale_rows == "all":
        return motion, rigid, reg, sigma_floor, scales.exp().sum() / denominator
    return motion, rigid, reg, sigma_floor, (weight[:, None] * dynamic_only(scales).exp()).sum() / denominator


@functools.cache
def _compiled_core_regularisers() -> Callable[..., RegulariserTerms]:
    return torch.compile(core_regularisers, dynamic=False)


def compiled_core_regularisers(device: torch.device) -> Callable[..., RegulariserTerms]:
    """``core_regularisers`` compiled on CUDA, eager elsewhere and while beartype checks the process (``PIXI_DEV_MODE``)."""
    if device.type != "cuda" or os.environ.get("PIXI_DEV_MODE") == "1":
        return core_regularisers
    return _compiled_core_regularisers()
