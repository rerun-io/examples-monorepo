"""TRiGS's per-step row gather, Bezier SE(3) deform and rasterizer-input prep as one function (``Open4dgsTrainer``'s
``fused_deform``): compiled, its forward and backward are a few fused kernels instead of ~750 small launches per step.

``gather_deform`` is ``TrigsTrainer._render_inputs`` after the active-row query, for the global-shutter path: the same
maths as ``_slice_rowwise_params`` + ``_deform`` + the opacity / scale / colour prep, on the active rows only.
"""

import functools
import os
from collections.abc import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Float32, Int64
from torch import Tensor
from trigs.math.quaternion import quaternion_multiply, quaternion_to_rotmat
from trigs.math.se3 import bezier_quadratic, exp_se3_quat, gauge_fix_omega, tau_from_window

DEFORM_KEYS: tuple[str, ...] = (
    "times",
    "log_durations",
    "zeta_c0_t",
    "zeta_c1_t",
    "zeta_c2_t",
    "zeta_c0_r",
    "zeta_c1_r",
    "zeta_c2_r",
    "zeta_base_t",
    "zeta_base_r",
    "anchors",
    "means",
    "quats",
    "opacities",
    "scales",
    "sh0",
)
"""Row-wise parameters ``gather_deform`` reads, in its argument order (``shN`` and ``is_static`` are optional)."""

DeformOutputs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
"""(means_t, quats_t, gamma, scales, opacities, colors) of the active rows."""


def gather_deform(
    rows: tuple[Tensor, ...],
    shN: Tensor | None,
    is_static: Tensor | None,
    idx: Int64[Tensor, " n"],
    t: Float32[Tensor, ""],
    with_shN: bool,
    detach_window: bool,
    gauge_eps: float,
    gauge_denom_clamp: float,
    smooth_cull_tau: float,
) -> DeformOutputs:
    """The active rows ``idx`` of ``rows`` (``DEFORM_KEYS`` order) deformed to time ``t``.

    Returns (means_t, quats_t, gamma, scales, opacities, colors) as ``_render_inputs`` builds them; ``smooth_cull_tau`` 0
    = no smooth-cull renormalisation.
    """
    (mu_t, log_sigma_t, c0_t, c1_t, c2_t, c0_r, c1_r, c2_r, base_t, base_r, anchors, means, quats, opacities, scales, sh0) = (
        v.index_select(0, idx) for v in rows
    )
    t_scalar: Tensor = t.to(dtype=mu_t.dtype)
    if detach_window:
        tau: Tensor = tau_from_window(t_scalar, mu_t.detach(), log_sigma_t.detach())
    else:
        tau = tau_from_window(t_scalar, mu_t, log_sigma_t)
    nu: Tensor = base_t + bezier_quadratic(tau, c0_t, c1_t, c2_t)
    omega: Tensor = base_r + bezier_quadratic(tau, c0_r, c1_r, c2_r)
    static_mask: Tensor | None = None
    if is_static is not None:
        static_mask = is_static.index_select(0, idx) > 0.5
        static3: Tensor = static_mask.expand_as(nu)
        nu = torch.where(static3, torch.zeros_like(nu), nu)
        omega = torch.where(static3, torch.zeros_like(omega), omega)
    a_perp, nu_par = gauge_fix_omega(omega, nu, anchors, eps=gauge_eps, denom_clamp=gauge_denom_clamp)
    dt: Tensor = t_scalar - mu_t
    u: Tensor = torch.cat([nu_par * dt, omega * dt], dim=-1).clamp(-1e4, 1e4)
    q_rel, p_par = exp_se3_quat(u)
    R: Tensor = quaternion_to_rotmat(q_rel)
    means_t: Tensor = (R @ (means - a_perp).unsqueeze(-1)).squeeze(-1) + a_perp + p_par
    quats_n: Tensor = F.normalize(quats, dim=-1)
    quats_t: Tensor = quaternion_multiply(q_rel, quats_n)
    sigma_t: Tensor = torch.exp(log_sigma_t).clamp(min=1e-6)
    gamma: Tensor = torch.exp(-0.5 * (dt / sigma_t) ** 2)
    if smooth_cull_tau > 0.0:
        gamma = ((gamma - smooth_cull_tau) / (1.0 - smooth_cull_tau)).clamp(min=0.0)
    if static_mask is not None:
        means_t = torch.where(static_mask.expand_as(means_t), means, means_t)
        quats_t = torch.where(static_mask.expand_as(quats_t), quats_n, quats_t)
        gamma = torch.where(static_mask, torch.ones_like(gamma), gamma)
    means_t = torch.nan_to_num(means_t, nan=0.0, posinf=0.0, neginf=0.0)
    quats_t = torch.nan_to_num(quats_t, nan=0.0, posinf=0.0, neginf=0.0)
    identity: Tensor = torch.zeros_like(quats_t)
    identity[..., 0] = 1.0
    quats_t = torch.where((quats_t * quats_t).sum(dim=-1, keepdim=True) < 1e-16, identity, quats_t)
    base_opa: Tensor = torch.sigmoid(opacities)
    if base_opa.dim() == 1:
        base_opa = base_opa.unsqueeze(-1)
    effective_opa: Tensor = torch.clamp(base_opa * gamma, min=1e-10).squeeze(-1)
    colors: Tensor = torch.cat([sh0.unsqueeze(1), shN.index_select(0, idx)], dim=1) if with_shN and shN is not None else sh0.unsqueeze(1)
    return means_t, quats_t, gamma, torch.exp(scales), effective_opa, colors


@functools.cache
def _compiled_gather_deform() -> Callable[..., DeformOutputs]:
    return torch.compile(gather_deform, dynamic=False)


def compiled_gather_deform(device: torch.device) -> Callable[..., DeformOutputs]:
    """``gather_deform`` compiled on CUDA (the active-row count is dynamic: one graph per ``with_shN``), eager elsewhere
    and while beartype checks the process (``PIXI_DEV_MODE``: it cannot check dynamo's fake tensors)."""
    if device.type != "cuda" or os.environ.get("PIXI_DEV_MODE") == "1":
        return gather_deform
    return _compiled_gather_deform()
