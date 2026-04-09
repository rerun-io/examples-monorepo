from __future__ import annotations

import mast3r_slam_mojo_backends as _mojo_backends  # pyrefly: ignore
import torch
from torch import Tensor

from mast3r_slam import _backends as _cuda_backends  # pyrefly: ignore

POSE_DIM: int = 7


def get_unique_kf_idx(ii: Tensor, jj: Tensor) -> Tensor:
    return torch.unique(torch.cat((ii, jj)), sorted=True)


def create_inds(unique_kf_idx: Tensor, pin: int, ii: Tensor, jj: Tensor) -> tuple[Tensor, Tensor]:
    ii_ind: Tensor = torch.searchsorted(unique_kf_idx, ii) - pin
    jj_ind: Tensor = torch.searchsorted(unique_kf_idx, jj) - pin
    return ii_ind, jj_ind


def assemble_dense_system(
    hs: Tensor,
    gs: Tensor,
    ii_opt: Tensor,
    jj_opt: Tensor,
    n_vars: int,
) -> tuple[Tensor, Tensor]:
    device: torch.device = hs.device
    flat_h: Tensor = torch.zeros((n_vars * n_vars, POSE_DIM * POSE_DIM), device=device, dtype=hs.dtype)
    block_terms: Tensor = torch.cat((hs[0], hs[1], hs[2], hs[3]), dim=0).reshape(-1, POSE_DIM * POSE_DIM)
    rows: Tensor = torch.cat((ii_opt, ii_opt, jj_opt, jj_opt), dim=0)
    cols: Tensor = torch.cat((ii_opt, jj_opt, ii_opt, jj_opt), dim=0)
    valid_h: Tensor = (rows >= 0) & (cols >= 0)
    if bool(valid_h.any()):
        block_indices: Tensor = rows[valid_h] * n_vars + cols[valid_h]
        flat_h.index_add_(0, block_indices, block_terms[valid_h])
    h_dense: Tensor = flat_h.view(n_vars, n_vars, POSE_DIM, POSE_DIM).permute(0, 2, 1, 3).reshape(n_vars * POSE_DIM, n_vars * POSE_DIM)

    b_dense: Tensor = torch.zeros((n_vars, POSE_DIM), device=device, dtype=gs.dtype)
    rhs_terms: Tensor = torch.cat((gs[0], gs[1]), dim=0)
    rhs_rows: Tensor = torch.cat((ii_opt, jj_opt), dim=0)
    valid_b: Tensor = rhs_rows >= 0
    if bool(valid_b.any()):
        b_dense.index_add_(0, rhs_rows[valid_b], rhs_terms[valid_b])

    return h_dense, b_dense


def solve_dense_system(h_dense: Tensor, b_dense: Tensor, n_vars: int) -> Tensor:
    if n_vars <= 0:
        return torch.zeros((0, POSE_DIM), device=h_dense.device, dtype=h_dense.dtype)

    rhs: Tensor = -b_dense.reshape(n_vars * POSE_DIM, 1)
    eye: Tensor = torch.eye(n_vars * POSE_DIM, device=h_dense.device, dtype=h_dense.dtype)
    chol_h: Tensor = h_dense + 1e-9 * eye
    l_factor, info = torch.linalg.cholesky_ex(chol_h, upper=False)
    if int(info.max().item()) != 0:
        return torch.zeros((n_vars, POSE_DIM), device=h_dense.device, dtype=h_dense.dtype)
    dx_vec: Tensor = torch.cholesky_solve(rhs, l_factor, upper=False)
    return dx_vec.view(n_vars, POSE_DIM)


def _gauss_newton_impl(
    step_fn,
    pose_retr_fn,
    pose_data: Tensor,
    ii: Tensor,
    jj: Tensor,
    max_iter: int,
    delta_thresh: float,
    *step_args,
) -> tuple[Tensor]:
    num_poses: int = int(pose_data.shape[0])
    num_fix: int = 1
    unique_kf_idx: Tensor = get_unique_kf_idx(ii, jj)
    ii_opt: Tensor
    jj_opt: Tensor
    ii_opt, jj_opt = create_inds(unique_kf_idx, num_fix, ii, jj)
    n_vars: int = num_poses - num_fix

    dx: Tensor = torch.zeros((max(n_vars, 0), POSE_DIM), device=pose_data.device, dtype=pose_data.dtype)
    for _ in range(max_iter):
        hs: Tensor
        gs: Tensor
        hs, gs = step_fn(pose_data, *step_args)
        h_dense: Tensor
        b_dense: Tensor
        h_dense, b_dense = assemble_dense_system(hs, gs, ii_opt, jj_opt, n_vars)
        dx = solve_dense_system(h_dense, b_dense, n_vars)
        pose_retr_fn(pose_data, dx, num_fix)
        delta_norm: Tensor = torch.linalg.norm(dx)
        if float(delta_norm.item()) < delta_thresh:
            break

    return (dx,)


def gauss_newton_points(
    Twc: Tensor,
    Xs: Tensor,
    Cs: Tensor,
    ii: Tensor,
    jj: Tensor,
    idx_ii2jj: Tensor,
    valid_match: Tensor,
    Q: Tensor,
    sigma_point: float,
    C_thresh: float,
    Q_thresh: float,
    max_iter: int,
    delta_thresh: float,
) -> tuple[Tensor]:
    if not hasattr(_mojo_backends, "gauss_newton_points_step") or not hasattr(_mojo_backends, "pose_retr"):
        return _cuda_backends.gauss_newton_points(
            Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
            sigma_point, C_thresh, Q_thresh, max_iter, delta_thresh,
        )
    return _gauss_newton_impl(
        lambda pose_data, *args: _mojo_backends.gauss_newton_points_step(pose_data, *args),
        _mojo_backends.pose_retr,
        Twc,
        ii,
        jj,
        max_iter,
        delta_thresh,
        Xs,
        Cs,
        ii,
        jj,
        idx_ii2jj,
        valid_match,
        Q,
        sigma_point,
        C_thresh,
        Q_thresh,
    )


def gauss_newton_rays(
    Twc: Tensor,
    Xs: Tensor,
    Cs: Tensor,
    ii: Tensor,
    jj: Tensor,
    idx_ii2jj: Tensor,
    valid_match: Tensor,
    Q: Tensor,
    sigma_ray: float,
    sigma_dist: float,
    C_thresh: float,
    Q_thresh: float,
    max_iter: int,
    delta_thresh: float,
) -> tuple[Tensor]:
    if not hasattr(_mojo_backends, "gauss_newton_rays_step") or not hasattr(_mojo_backends, "pose_retr"):
        return _cuda_backends.gauss_newton_rays(
            Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
            sigma_ray, sigma_dist, C_thresh, Q_thresh, max_iter, delta_thresh,
        )
    return _gauss_newton_impl(
        lambda pose_data, *args: _mojo_backends.gauss_newton_rays_step(pose_data, *args),
        _mojo_backends.pose_retr,
        Twc,
        ii,
        jj,
        max_iter,
        delta_thresh,
        Xs,
        Cs,
        ii,
        jj,
        idx_ii2jj,
        valid_match,
        Q,
        sigma_ray,
        sigma_dist,
        C_thresh,
        Q_thresh,
    )


def gauss_newton_calib(
    Twc: Tensor,
    Xs: Tensor,
    Cs: Tensor,
    K: Tensor,
    ii: Tensor,
    jj: Tensor,
    idx_ii2jj: Tensor,
    valid_match: Tensor,
    Q: Tensor,
    height: int,
    width: int,
    pixel_border: int,
    z_eps: float,
    sigma_pixel: float,
    sigma_depth: float,
    C_thresh: float,
    Q_thresh: float,
    max_iter: int,
    delta_thresh: float,
) -> tuple[Tensor]:
    if not hasattr(_mojo_backends, "gauss_newton_calib_step") or not hasattr(_mojo_backends, "pose_retr"):
        return _cuda_backends.gauss_newton_calib(
            Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q,
            height, width, pixel_border, z_eps, sigma_pixel, sigma_depth,
            C_thresh, Q_thresh, max_iter, delta_thresh,
        )
    return _gauss_newton_impl(
        lambda pose_data, *args: _mojo_backends.gauss_newton_calib_step(pose_data, *args),
        _mojo_backends.pose_retr,
        Twc,
        ii,
        jj,
        max_iter,
        delta_thresh,
        Xs,
        Cs,
        K,
        ii,
        jj,
        idx_ii2jj,
        valid_match,
        Q,
        height,
        width,
        pixel_border,
        z_eps,
        sigma_pixel,
        sigma_depth,
        C_thresh,
        Q_thresh,
    )
