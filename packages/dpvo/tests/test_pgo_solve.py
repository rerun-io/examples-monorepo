"""Loop closure's pose-graph step (`dpvo.fastba.solve_system`, Mojo PCG) against a dense float64 solve.

The dense reference builds the full residual Jacobian J (7 rows per edge, 7 columns per pose) and solves
(J^T J + lm diag(J^T J) + ep) delta = -J^T r, as the former Eigen extension did (SimplicialCholesky).
"""

import pytest
import torch
from torch import Tensor

from dpvo.fastba import solve_system


def _problem(n: int, loops: int, seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """A chain of `n` poses (edge i -> i+1) plus `loops` long edges, with well-conditioned 7x7 blocks."""
    gen = torch.Generator().manual_seed(seed)
    ii = torch.cat([torch.arange(n - 1), torch.randint(0, n // 2, (loops,), generator=gen)])
    jj = torch.cat([torch.arange(1, n), torch.randint(n // 2, n, (loops,), generator=gen)])
    r = ii.shape[0]
    J_i = torch.randn(r, 7, 7, generator=gen) * 0.3 + torch.eye(7)
    J_j = torch.randn(r, 7, 7, generator=gen) * 0.3 - torch.eye(7)
    res = torch.randn(r, 7, generator=gen) * 0.01
    return J_i, J_j, ii, jj, res


def _dense_reference(J_i: Tensor, J_j: Tensor, ii: Tensor, jj: Tensor, res: Tensor, ep: float, lm: float, free: int) -> Tensor:
    n = int(max(ii.max(), jj.max())) + 1
    r = ii.shape[0]
    J = torch.zeros(7 * r, 7 * n, dtype=torch.float64)
    for e in range(r):
        J[7 * e : 7 * e + 7, 7 * ii[e] : 7 * ii[e] + 7] = J_i[e].double()
        J[7 * e : 7 * e + 7, 7 * jj[e] : 7 * jj[e] + 7] = J_j[e].double()
    A = J.T @ J
    A.diagonal().mul_(1 + lm).add_(ep)
    b = -(J.T @ res.double().reshape(-1))
    m = 7 * (n if free < 0 else free)
    delta = torch.zeros(7 * n, dtype=torch.float64)
    delta[:m] = torch.linalg.solve(A[:m, :m], b[:m])
    return delta.float().reshape(n, 7)


DEVICES: list = ["cpu", pytest.param("cuda", marks=[pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")])]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("free", [-1, 30])
def test_solve_system_matches_the_dense_normal_equations(free: int, device: str) -> None:
    J_i, J_j, ii, jj, res = _problem(n=60, loops=6, seed=3)
    ours: Tensor = solve_system(J_i.to(device), J_j.to(device), ii.to(device), jj.to(device), res.to(device), 1e-4, 1e-3, free).cpu()
    ref: Tensor = _dense_reference(J_i, J_j, ii, jj, res, 1e-4, 1e-3, free)
    assert ours.shape == (60, 7) and ours.dtype == torch.float32
    assert ((ours - ref).abs().max() / ref.abs().max()).item() < 1e-5
    if free >= 0:
        assert torch.equal(ours[free:], torch.zeros_like(ours[free:]))
