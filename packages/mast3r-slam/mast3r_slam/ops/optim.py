import math

import torch
from jaxtyping import Bool, Float, Float32, Float64, Int64
from torch import Tensor

from mast3r_slam.types import LocalOptConfig

POSE_DIM: int = 7
"""Sim3 tangent dimension."""


def check_convergence(
    rel_error_threshold: float,
    delta_norm_threshold: float,
    old_cost: float,
    new_cost: float,
    delta: Float[Tensor, "..."],
) -> bool:
    """Check if the optimizer has converged based on relative error and delta norm."""
    cost_diff: float = old_cost - new_cost
    rel_dec: float = math.inf
    if math.isfinite(old_cost) and old_cost > 0.0:
        rel_dec = math.fabs(cost_diff / old_cost)
    delta_norm: Float[Tensor, ""] = torch.linalg.norm(delta)

    return rel_dec < rel_error_threshold or bool(delta_norm < delta_norm_threshold)


def huber(r: Float[Tensor, "..."], k: float = 1.345) -> Float[Tensor, "..."]:
    """Compute Huber weighting function: 1 where |r| < k, else k/|r|."""
    unit: Float[Tensor, "1"] = torch.ones((1), dtype=r.dtype, device=r.device)
    r_abs: Float[Tensor, "..."] = torch.abs(r)
    mask: Bool[Tensor, "..."] = r_abs < k
    w: Float[Tensor, "..."] = torch.where(mask, unit, k / r_abs)
    return w


def solve_pose_blocks(
    hs: Float[Tensor, "4 e 7 7"], gs: Float[Tensor, "2 e 7"], ii_opt: Int64[Tensor, " e"], jj_opt: Int64[Tensor, " e"], n_vars: int
) -> Float64[Tensor, "n_vars 7"]:
    """Assemble the dense normal equations of a pose graph on the CPU (float64) and solve them by Cholesky.

    ``hs`` = per-edge Hessian blocks (ii, ij, ji, jj), ``gs`` = per-edge gradients (ii, jj); ``ii_opt`` / ``jj_opt`` index
    the free poses (negative = pinned). A failed factorisation gives a zero step. Every buffer is a torch tensor freed when
    the call returns (the Mojo loop this replaces kept its host buffers alive: host memory grew with keyframes^3).
    """
    hs_cpu: Float64[Tensor, "4 e 7 7"] = hs.cpu().to(dtype=torch.float64)
    gs_cpu: Float64[Tensor, "2 e 7"] = gs.cpu().to(dtype=torch.float64)
    h_blocks: Float64[Tensor, "n_vars n_vars 7 7"] = torch.zeros([n_vars, n_vars, POSE_DIM, POSE_DIM], dtype=torch.float64)
    b_dense: Float64[Tensor, "n_vars 7"] = torch.zeros([n_vars, POSE_DIM], dtype=torch.float64)
    ii_mask: Bool[Tensor, " e"] = ii_opt >= 0
    jj_mask: Bool[Tensor, " e"] = jj_opt >= 0
    ij_mask: Bool[Tensor, " e"] = ii_mask & jj_mask
    if bool(ii_mask.any()):
        h_blocks.index_put_((ii_opt[ii_mask], ii_opt[ii_mask]), hs_cpu[0][ii_mask], accumulate=True)
        b_dense.index_add_(0, ii_opt[ii_mask], gs_cpu[0][ii_mask])
    if bool(jj_mask.any()):
        h_blocks.index_put_((jj_opt[jj_mask], jj_opt[jj_mask]), hs_cpu[3][jj_mask], accumulate=True)
        b_dense.index_add_(0, jj_opt[jj_mask], gs_cpu[1][jj_mask])
    if bool(ij_mask.any()):
        h_blocks.index_put_((ii_opt[ij_mask], jj_opt[ij_mask]), hs_cpu[1][ij_mask], accumulate=True)
        h_blocks.index_put_((jj_opt[ij_mask], ii_opt[ij_mask]), hs_cpu[2][ij_mask], accumulate=True)
    h_dense: Float64[Tensor, "m m"] = h_blocks.permute(0, 2, 1, 3).reshape(n_vars * POSE_DIM, n_vars * POSE_DIM)
    damped: Float64[Tensor, "m m"] = h_dense + 1e-9 * torch.eye(n_vars * POSE_DIM, dtype=torch.float64)
    factor, info = torch.linalg.cholesky_ex(damped, upper=False)
    if int(info.max()) != 0:
        return torch.zeros([n_vars, POSE_DIM], dtype=torch.float64)
    return torch.cholesky_solve(-b_dense.reshape(n_vars * POSE_DIM, 1), factor, upper=False).view(n_vars, POSE_DIM)


def gauss_newton_rays(
    poses: Float32[Tensor, "n 8"],
    Xs: Float32[Tensor, "n hw 3"],
    Cs: Float32[Tensor, "n hw 1"],
    ii: Int64[Tensor, " e"],
    jj: Int64[Tensor, " e"],
    idx_ii2jj: Int64[Tensor, "e hw"],
    valid_match: Bool[Tensor, "e hw 1"],
    Q_ii2jj: Float32[Tensor, "e hw 1"],
    cfg: LocalOptConfig,
) -> None:
    """Ray-distance Gauss-Newton over Sim3 keyframe poses, in place on ``poses`` (the first ``cfg.pin`` stay fixed).

    ``ii``/``jj`` index rows of ``poses`` (two-way edges). The linearisation (Mojo step kernel) and the retraction run on
    the GPU; each iteration's dense system is solved on the CPU by ``solve_pose_blocks``. Stops after ``cfg.max_iters`` or
    when the step norm falls below ``cfg.delta_norm``.
    """
    import mast3r_slam_mojo_backends as mojo  # pyrefly: ignore  # the bare .so at the package root (mast3r_slam puts it on sys.path)

    unique: Int64[Tensor, " u"] = torch.unique(torch.cat([ii, jj]), sorted=True)
    ii_opt: Int64[Tensor, " e"] = (torch.searchsorted(unique, ii) - cfg.pin).cpu()
    jj_opt: Int64[Tensor, " e"] = (torch.searchsorted(unique, jj) - cfg.pin).cpu()
    n_vars: int = poses.shape[0] - cfg.pin
    step_args = (poses, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q_ii2jj, cfg.sigma_ray, cfg.sigma_dist, cfg.C_conf, cfg.Q_conf)
    for _ in range(cfg.max_iters):
        hs, gs = mojo.gauss_newton_rays_step(step_args)
        dx: Float32[Tensor, "n_vars 7"] = solve_pose_blocks(hs, gs, ii_opt, jj_opt, n_vars).to(device=poses.device, dtype=poses.dtype)
        mojo.pose_retr(poses, dx, cfg.pin)
        if float(torch.linalg.norm(dx)) < cfg.delta_norm:
            break
    torch.cuda.synchronize(device=poses.device)
