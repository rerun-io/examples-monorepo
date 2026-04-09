from __future__ import annotations

import ctypes
import sys
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import lietorch
import torch

POSE_DIM = 7
RAYS_BLOCKS_PER_EDGE_MAX = 8


@lru_cache(maxsize=1)
def _ensure_modular_runtime_loaded() -> None:
    spec = find_spec("modular")
    if spec is None:
        return
    if spec.submodule_search_locations:
        modular_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    elif spec.origin is not None:
        modular_root = Path(spec.origin).resolve().parent
    else:
        return
    runtime_lib = modular_root / "lib" / "libMGPRT.so"
    expected_runtime_lib = Path(sys.prefix) / "lib" / "libMGPRT.so"
    if runtime_lib.exists() and not expected_runtime_lib.exists():
        expected_runtime_lib.parent.mkdir(parents=True, exist_ok=True)
        expected_runtime_lib.symlink_to(runtime_lib)
    if runtime_lib.exists():
        ctypes.CDLL(str(runtime_lib), mode=ctypes.RTLD_GLOBAL)


@lru_cache(maxsize=1)
def _get_custom_ops() -> Any:
    from max.experimental.torch import CustomOpLibrary

    _ensure_modular_runtime_loaded()
    kernels = Path(__file__).resolve().parent / "backend" / "max_gn_rays" / "kernels"
    return CustomOpLibrary(kernels)


@lru_cache(maxsize=None)
def _float_scalar_tensor(value: float, device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    return torch.tensor([value], device=device, dtype=torch.float32)


def available() -> bool:
    try:
        _get_custom_ops()
    except Exception:
        return False
    return True


def _scalar_tensor(value: float, device: torch.device) -> torch.Tensor:
    return _float_scalar_tensor(value, device.type, device.index)


def _get_unique_kf_idx(ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
    return torch.unique(torch.cat((ii, jj)), sorted=True)


def _create_inds(
    unique_kf_idx: torch.Tensor,
    pin: int,
    ii: torch.Tensor,
    jj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ii_ind = torch.searchsorted(unique_kf_idx, ii) - pin
    jj_ind = torch.searchsorted(unique_kf_idx, jj) - pin
    return ii_ind, jj_ind


def _solve_step_system(
    hs: torch.Tensor,
    gs: torch.Tensor,
    ii_opt_cpu: torch.Tensor,
    jj_opt_cpu: torch.Tensor,
    n_vars: int,
    out_device: torch.device,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if n_vars <= 0:
        return torch.zeros((0, POSE_DIM), device=out_device, dtype=out_dtype)

    hs_cpu = hs.cpu().to(dtype=torch.float64)
    gs_cpu = gs.cpu().to(dtype=torch.float64)
    h_blocks = torch.zeros((n_vars, n_vars, POSE_DIM, POSE_DIM), dtype=torch.float64)
    b_dense = torch.zeros((n_vars, POSE_DIM), dtype=torch.float64)

    ii_mask = ii_opt_cpu >= 0
    jj_mask = jj_opt_cpu >= 0
    ij_mask = ii_mask & jj_mask

    if bool(ii_mask.any().item()):
        ii_idx = ii_opt_cpu[ii_mask]
        h_blocks.index_put_((ii_idx, ii_idx), hs_cpu[0][ii_mask], accumulate=True)
        b_dense.index_add_(0, ii_idx, gs_cpu[0][ii_mask])

    if bool(jj_mask.any().item()):
        jj_idx = jj_opt_cpu[jj_mask]
        h_blocks.index_put_((jj_idx, jj_idx), hs_cpu[3][jj_mask], accumulate=True)
        b_dense.index_add_(0, jj_idx, gs_cpu[1][jj_mask])

    if bool(ij_mask.any().item()):
        ii_cross = ii_opt_cpu[ij_mask]
        jj_cross = jj_opt_cpu[ij_mask]
        h_blocks.index_put_((ii_cross, jj_cross), hs_cpu[1][ij_mask], accumulate=True)
        h_blocks.index_put_((jj_cross, ii_cross), hs_cpu[2][ij_mask], accumulate=True)

    h_dense = h_blocks.permute(0, 2, 1, 3).reshape(n_vars * POSE_DIM, n_vars * POSE_DIM)
    rhs = -b_dense.reshape(n_vars * POSE_DIM, 1)
    eye = torch.eye(n_vars * POSE_DIM, dtype=torch.float64)
    chol_h = h_dense + 1e-9 * eye
    l_factor, info = torch.linalg.cholesky_ex(chol_h, upper=False)
    if int(info.max().item()) != 0:
        return torch.zeros((n_vars, POSE_DIM), device=out_device, dtype=out_dtype)
    dx_vec = torch.cholesky_solve(rhs, l_factor, upper=False)
    return dx_vec.view(n_vars, POSE_DIM).to(device=out_device, dtype=out_dtype)


def _gauss_newton_rays_step_partial(
    twc: torch.Tensor,
    xs: torch.Tensor,
    cs: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    idx_ii2jj: torch.Tensor,
    valid_match: torch.Tensor,
    q_tensor: torch.Tensor,
    sigma_ray: float,
    sigma_dist: float,
    c_thresh: float,
    q_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ops = _get_custom_ops()
    num_edges = int(ii.shape[0])
    num_points = int(xs.shape[1])
    blocks_per_edge = max(1, min(RAYS_BLOCKS_PER_EDGE_MAX, (num_points + 16383) // 16384))
    num_partials = num_edges * blocks_per_edge
    hs_partial = torch.zeros((4, num_partials, POSE_DIM, POSE_DIM), device=twc.device, dtype=torch.float32)
    gs_partial = torch.zeros((2, num_partials, POSE_DIM), device=twc.device, dtype=torch.float32)
    ops.gauss_newton_rays_step_partial(
        hs_partial,
        gs_partial,
        twc,
        xs,
        cs,
        ii.to(dtype=torch.int32),
        jj.to(dtype=torch.int32),
        idx_ii2jj.to(dtype=torch.int32),
        valid_match.to(dtype=torch.uint8),
        q_tensor,
        _scalar_tensor(sigma_ray, twc.device),
        _scalar_tensor(sigma_dist, twc.device),
        _scalar_tensor(c_thresh, twc.device),
        _scalar_tensor(q_thresh, twc.device),
    )
    torch.cuda.synchronize(device=twc.device)
    if blocks_per_edge == 1:
        return hs_partial, gs_partial
    hs = hs_partial.view(4, num_edges, blocks_per_edge, POSE_DIM, POSE_DIM).sum(dim=2)
    gs = gs_partial.view(2, num_edges, blocks_per_edge, POSE_DIM).sum(dim=2)
    return hs, gs


def _pose_retr_inplace(twc: torch.Tensor, dx: torch.Tensor, num_fix: int) -> None:
    if num_fix >= int(twc.shape[0]):
        return
    tail = lietorch.Sim3(twc[num_fix:])
    update = lietorch.Sim3.exp(dx)
    twc[num_fix:] = tail.retr(update).data.to(device=twc.device, dtype=twc.dtype)


def gauss_newton_rays(
    Twc: torch.Tensor,
    Xs: torch.Tensor,
    Cs: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    idx_ii2jj: torch.Tensor,
    valid_match: torch.Tensor,
    Q: torch.Tensor,
    sigma_ray: float,
    sigma_dist: float,
    c_thresh: float,
    q_thresh: float,
    max_iter: int,
    delta_thresh: float,
) -> tuple[torch.Tensor]:
    """True isolated MAX custom-op GN rays path.

    This wrapper intentionally does not fall back to the shared-lib Mojo path.
    It exists so the MAX implementation can be developed and validated as a
    separate vertical slice while the active repo backend remains idiomatic
    Mojo.
    """
    twc = Twc.contiguous().float()
    xs = Xs.contiguous().float()
    cs = Cs.contiguous().float()
    ii = ii.contiguous()
    jj = jj.contiguous()
    idx_ii2jj = idx_ii2jj.contiguous()
    valid_u8 = valid_match.contiguous().to(dtype=torch.uint8)
    q_tensor = Q.contiguous().float()

    num_fix = 1
    unique_kf_idx = _get_unique_kf_idx(ii, jj)
    ii_opt, jj_opt = _create_inds(unique_kf_idx, num_fix, ii, jj)
    ii_opt_cpu = ii_opt.cpu()
    jj_opt_cpu = jj_opt.cpu()
    n_vars = int(twc.shape[0]) - num_fix
    dx = torch.zeros((max(n_vars, 0), POSE_DIM), device=twc.device, dtype=twc.dtype)

    for _ in range(max_iter):
        hs, gs = _gauss_newton_rays_step_partial(
            twc,
            xs,
            cs,
            ii,
            jj,
            idx_ii2jj,
            valid_u8,
            q_tensor,
            sigma_ray,
            sigma_dist,
            c_thresh,
            q_thresh,
        )
        dx = _solve_step_system(hs, gs, ii_opt_cpu, jj_opt_cpu, n_vars, twc.device, twc.dtype)
        _pose_retr_inplace(twc, dx, num_fix)
        if float(torch.linalg.norm(dx).item()) < delta_thresh:
            break

    return (dx,)
