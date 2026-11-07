"""Loop closure's pose-graph step: preconditioned conjugate gradient on the Mojo kernels (pgo.mojo).

The normal matrix A = J^T J of the Sim(3) pose graph is never assembled: `pgo_apply` multiplies by it per node,
gathering over each node's incident edges, and `pgo_blocks` returns its 7x7 diagonal blocks and the gradient
J^T r. The preconditioner is the odometry chain (edges k -> k + 1, the block-tridiagonal part of A) factored
exactly (`pgo_chain_factor` / `pgo_chain_solve`): block-Jacobi alone hit 7n iterations without converging on a
real 1,188-pose KITTI loop closure, since a chain's condition number grows like n^2. Vectors and scalars stay on
the device; the residual norm is read every `CHECK_EVERY` iterations. Everything is float64.
"""

import torch
from jaxtyping import Float32, Float64, Int64
from torch import Tensor

from dpvo.mojo_backend._launch import launch

TOLERANCE: float = 1e-10
"""Stop when |b - A x| <= TOLERANCE |b|."""
CHECK_EVERY: int = 10


def solve_system(
    J_Ginv_i: Float32[Tensor, "n_res 7 7"],
    J_Ginv_j: Float32[Tensor, "n_res 7 7"],
    ii: Int64[Tensor, "n_res"],
    jj: Int64[Tensor, "n_res"],
    res: Float32[Tensor, "n_res 7"],
    ep: float,
    lm: float,
    freen: int,
) -> Float32[Tensor, "n_poses 7"]:
    """One Levenberg-Marquardt step: solve (A + lm diag(A) + ep) delta = -J^T r on `res.device`; poses >= freen stay fixed."""
    device: torch.device = res.device
    n: int = int(torch.maximum(ii.max(), jj.max()).item()) + 1
    num_res: int = ii.shape[0]
    f64: dict = {"dtype": torch.float64, "device": device}
    J_i: Float64[Tensor, "n_res 7 7"] = J_Ginv_i.to(torch.float64).contiguous()
    J_j: Float64[Tensor, "n_res 7 7"] = J_Ginv_j.to(torch.float64).contiguous()
    nodes: Int64[Tensor, "two_res"] = torch.cat([ii, jj])
    order: Int64[Tensor, "two_res"] = torch.argsort(nodes, stable=True)
    incident: Int64[Tensor, "two_res"] = torch.cat([2 * torch.arange(num_res, device=device), 2 * torch.arange(num_res, device=device) + 1])[order]
    offsets: Int64[Tensor, "n_plus_1"] = torch.zeros(n + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(torch.bincount(nodes, minlength=n), 0)

    blocks: Float64[Tensor, "n 7 7"] = torch.empty(n, 7, 7, **f64)
    grad: Float64[Tensor, "n 7"] = torch.empty(n, 7, **f64)
    launch("pgo_blocks", (), (J_i, J_j, res.to(torch.float64).contiguous(), offsets, incident, blocks, grad))
    damp: Float64[Tensor, "n 7"] = torch.diagonal(blocks, dim1=1, dim2=2) * lm + ep
    free: Float64[Tensor, "n 1"] = torch.ones(n, 1, **f64)
    diag: Float64[Tensor, "n 7 7"] = blocks + torch.diag_embed(damp)
    sub: Float64[Tensor, "n_minus_1 7 7"] = torch.zeros(max(n - 1, 0), 7, 7, **f64)
    forward: Tensor = jj == ii + 1  # block (jj, ii) = J_j^T J_i
    backward: Tensor = ii == jj + 1  # block (ii, jj) = J_i^T J_j
    sub.index_add_(0, ii[forward], J_j[forward].mT @ J_i[forward])
    sub.index_add_(0, jj[backward], J_i[backward].mT @ J_j[backward])
    if freen >= 0:
        free[freen:] = 0.0
        diag[freen:] = torch.eye(7, **f64)
        sub[max(freen - 1, 0) :] = 0.0
    l_diag: Float64[Tensor, "n 7 7"] = torch.empty_like(diag)
    l_sub: Float64[Tensor, "n_minus_1 7 7"] = torch.empty_like(sub)
    launch("pgo_chain_factor", (), (diag, sub, l_diag, l_sub))

    def precondition(r: Float64[Tensor, "n 7"]) -> Float64[Tensor, "n 7"]:
        z: Float64[Tensor, "n 7"] = torch.empty(n, 7, **f64)
        launch("pgo_chain_solve", (), (l_diag, l_sub, r, z))
        return z * free

    def apply(x: Float64[Tensor, "n 7"]) -> Float64[Tensor, "n 7"]:
        y: Float64[Tensor, "n 7"] = torch.empty(n, 7, **f64)
        launch("pgo_apply", (), (J_i, J_j, ii, jj, offsets, incident, damp, x, y))
        return y * free

    b: Float64[Tensor, "n 7"] = -grad * free
    x: Float64[Tensor, "n 7"] = torch.zeros(n, 7, **f64)
    r: Float64[Tensor, "n 7"] = b.clone()
    z: Float64[Tensor, "n 7"] = precondition(r)
    p: Float64[Tensor, "n 7"] = z.clone()
    rz: Tensor = (r * z).sum()
    limit: float = TOLERANCE * float(torch.linalg.vector_norm(b).item())
    for iteration in range(7 * n):
        Ap: Float64[Tensor, "n 7"] = apply(p)
        alpha: Tensor = rz / (p * Ap).sum()
        x += alpha * p
        r -= alpha * Ap
        if (iteration + 1) % CHECK_EVERY == 0 and float(torch.linalg.vector_norm(r).item()) <= limit:
            break
        z = precondition(r)
        rz_next: Tensor = (r * z).sum()
        p = z + (rz_next / rz) * p
        rz = rz_next
    return x.to(torch.float32)
