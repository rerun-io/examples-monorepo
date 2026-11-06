"""RxSO3 and Sim3 self-consistency and torch.autograd.gradcheck (float64, torch's default tolerances).

No upstream lietorch needed: CPU runs in the unit lane, CUDA is integration.

Sim3's left Jacobian and its inverse are sim3.h's truncated series (sim3.mojo
docstring), exact only as the tangent norm goes to 0, so the Sim3 composites
that go through `expm_backward`, `logm_backward` or `Jinv` use tangents and
base elements of norm ~0.1-0.5, where the truncation (|Xi|^5 / 720) stays
far below gradcheck's tolerance. Upstream lietorch's own run_tests.py relaxes
Sim3 to tol 1e-3 for the same reason.
"""

import math
from collections.abc import Callable

import pytest
import torch
from conftest import DEVICES, require_mojo_extension
from torch import Tensor

require_mojo_extension()

from groups_common import homogeneous  # noqa: E402

from lietorch_mojo import SE3, RxSO3, Sim3  # noqa: E402

GROUPS: dict[str, type] = {"RxSO3": RxSO3, "Sim3": Sim3}


# ── forward identities (run_tests.py: test_exp_log, test_inv, test_adj, test_act) ──


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", tuple(GROUPS))
def test_exp_log_is_identity(group: str, device: str) -> None:
    G = GROUPS[group]
    a: Tensor = 0.2 * torch.randn(2, 3, 4, 5, G.manifold_dim, device=device, dtype=torch.float64)
    assert torch.allclose(G.exp(a).log(), a, atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", tuple(GROUPS))
def test_inv(group: str, device: str) -> None:
    G = GROUPS[group]
    X = G.exp(0.1 * torch.randn(2, 3, 4, 5, G.manifold_dim, device=device, dtype=torch.float64))
    a: Tensor = (X * X.inv()).log()
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", tuple(GROUPS))
def test_adj(group: str, device: str) -> None:
    G = GROUPS[group]
    X = G.exp(torch.randn(2, 3, 4, 5, G.manifold_dim, device=device, dtype=torch.float64))
    a: Tensor = torch.randn(2, 3, 4, 5, G.manifold_dim, device=device, dtype=torch.float64)
    c: Tensor = ((X * G.exp(a)) * (G.exp(X.adj(a)) * X).inv()).log()
    assert torch.allclose(c, torch.zeros_like(c), atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", tuple(GROUPS))
def test_act_matches_matrix(group: str, device: str) -> None:
    G = GROUPS[group]
    X = G.exp(torch.randn(16, G.manifold_dim, device=device, dtype=torch.float64))
    p: Tensor = torch.randn(16, 3, device=device, dtype=torch.float64)
    p2: Tensor = torch.matmul(X.matrix(), homogeneous(p)[..., None])[..., 0]
    assert torch.allclose(X.act(p), p2[..., :3], atol=1e-8)
    assert torch.allclose(X.act(homogeneous(p)), p2, atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", tuple(GROUPS))
def test_adjT_is_transpose_of_adj(group: str, device: str) -> None:
    G = GROUPS[group]
    X = G.exp(torch.randn(16, G.manifold_dim, device=device, dtype=torch.float64))
    a: Tensor = torch.randn(16, G.manifold_dim, device=device, dtype=torch.float64)
    b: Tensor = torch.randn(16, G.manifold_dim, device=device, dtype=torch.float64)
    assert torch.allclose((X.adj(a) * b).sum(-1), (a * X.adjT(b)).sum(-1), atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_sim3_with_unit_scale_is_se3(device: str) -> None:
    """Sim3 elements with s = 1 act, multiply and invert like the SE3 elements they extend."""
    se3 = SE3.exp(torch.randn(32, 6, device=device, dtype=torch.float64))
    sim3 = Sim3(se3)
    p: Tensor = torch.randn(32, 3, device=device, dtype=torch.float64)
    assert torch.allclose(sim3.act(p), se3.act(p), atol=1e-12)
    assert torch.allclose(sim3.inv().data[..., :7], se3.inv().data, atol=1e-12)
    assert torch.allclose((sim3 * sim3).data[..., :7], (se3 * se3).data, atol=1e-12)
    assert torch.allclose(sim3.log()[..., :6], se3.log(), atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", tuple(GROUPS))
def test_scale_round_trip(group: str, device: str) -> None:
    """The log-scale tangent coordinate maps to the data scale exp(sigma) and back, over a wide range."""
    G = GROUPS[group]
    a: Tensor = torch.zeros(9, G.manifold_dim, dtype=torch.float64)
    a[:, -1] = torch.tensor([-30.0, -5.0, -1e-3, -1e-9, 0.0, 1e-9, 1e-3, 5.0, 30.0], dtype=torch.float64)
    X = G.exp(a.to(device))
    assert torch.allclose(X.data[..., -1].cpu(), torch.exp(a[:, -1]), rtol=4e-16, atol=0)
    assert torch.allclose(X.log().cpu(), a, rtol=1e-15, atol=1e-15)


# ── gradcheck ────────────────────────────────────────────────────────────────

Composite = Callable[..., Tensor]


def _composites(group: str, device: str) -> dict[str, tuple[Composite, list[Tensor]]]:
    G = GROUPS[group]
    D: int = G.manifold_dim
    gen: torch.Generator = torch.Generator().manual_seed(0)

    def randn(*shape: int, scale: float = 1.0) -> Tensor:
        return (scale * torch.randn(*shape, generator=gen, dtype=torch.float64)).to(device)

    X = G.exp(randn(2, D, scale=0.2))
    Y = G.exp(randn(2, D, scale=0.2))
    X_big = G.exp(randn(2, D, scale=2.0))  # used only through act / matrix / vec, which are exact

    def tangent(n: int = 2, scale: float = 0.1) -> Tensor:
        return randn(n, D, scale=scale).requires_grad_(True)

    def zero(n: int = 2) -> Tensor:
        return torch.zeros(n, D, dtype=torch.float64, device=device, requires_grad=True)

    def fromvec(v: Tensor) -> Tensor:
        # run_tests.py test_fromvec_grad: normalise the quaternion, exponentiate the scale.
        if group == "RxSO3":
            q, s = v.split([4, 1], dim=-1)
            data = torch.cat([q / q.norm(dim=-1, keepdim=True), s.exp()], dim=-1)
        else:
            t, q, s = v.split([3, 4, 1], dim=-1)
            data = torch.cat([t, q / q.norm(dim=-1, keepdim=True), s.exp()], dim=-1)
        return G.InitFromVec(data).vec()

    axis: Tensor = torch.tensor([0.3, -0.4, 0.866], dtype=torch.float64)
    axis = axis / axis.norm()
    near_pi_tangent: Tensor = torch.zeros(1, D, dtype=torch.float64)
    near_pi_tangent[0, D - 4 : D - 1] = (math.pi - 1e-2) * axis
    near_pi_tangent[0, -1] = 0.3
    near_pi = G.exp(near_pi_tangent.to(device))

    return {
        "exp_log_at_zero": (lambda a: G.exp(a).log(), [zero()]),
        "exp_log": (lambda a: G.exp(a).log(), [tangent()]),
        "inv_log": (lambda a: (G.exp(a) * X).inv().log(), [tangent()]),
        "inv_log_at_zero": (lambda a: (G.exp(a) * X).inv().log(), [zero()]),
        "mul_log": (lambda a, b: (G.exp(a) * X * G.exp(b) * Y).log(), [tangent(), tangent()]),
        "retr_log": (lambda a: X.retr(a).log(), [tangent()]),
        "adj": (lambda a, b: (G.exp(a) * X).adj(b), [tangent(), randn(2, D).requires_grad_(True)]),
        "adjT": (lambda a, b: (G.exp(a) * X).adjT(b), [tangent(), randn(2, D).requires_grad_(True)]),
        "act3_big": (lambda a, p: (X_big * G.exp(a)).act(p), [zero(), randn(2, 3).requires_grad_(True)]),
        "act3": (lambda a, p: (X * G.exp(a)).act(p), [tangent(), randn(2, 3).requires_grad_(True)]),
        "act4": (lambda a, p: (X * G.exp(a)).act(p), [tangent(), randn(2, 4).requires_grad_(True)]),
        "matrix": (lambda a: (G.exp(a) * X_big).matrix(), [zero()]),
        "translation": (lambda a: (G.exp(a) * X_big).translation(), [zero()]),
        "vec": (lambda a: (G.exp(a) * X_big).vec(), [zero()]),
        "fromvec": (fromvec, [randn(1, 2, G.embedded_dim).requires_grad_(True)]),
        "near_pi_inv_act": (lambda a, p: (G.exp(a) * near_pi).inv().act(p), [zero(1), randn(1, 3).requires_grad_(True)]),
        "broadcast_mul_act": (lambda a, p: (G.exp(a)[:, None] * X[None]).act(p), [tangent(3), randn(3, 2, 3).requires_grad_(True)]),
    }


GRADCHECK_PARAMS: list = [pytest.param(g, n, id=f"{g}-{n}") for g in GROUPS for n in _composites(g, "cpu")]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("group", "name"), GRADCHECK_PARAMS)
def test_gradcheck_float64(group: str, name: str, device: str) -> None:
    fn, inputs = _composites(group, device)[name]
    assert torch.autograd.gradcheck(fn, tuple(inputs))


@pytest.mark.parametrize("device", DEVICES)
def test_rxso3_exp_log_jacobian_is_identity(device: str) -> None:
    """run_tests.py test_exp_log_grad at tol 1e-8: RxSO3's Jacobians are exact."""
    for a0 in (torch.zeros(1, 4, dtype=torch.float64), 0.2 * torch.randn(1, 4, dtype=torch.float64)):
        J: Tensor = torch.autograd.functional.jacobian(lambda a: RxSO3.exp(a).log(), a0.to(device))
        assert torch.allclose(J.reshape(4, 4), torch.eye(4, dtype=torch.float64, device=device), atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_sim3_exp_log_jacobian_is_identity_to_series_accuracy(device: str) -> None:
    """run_tests.py test_exp_log_grad: exact at 0; at |a| ~ 0.2 within lietorch's Sim3 tol (1e-3)."""
    J0: Tensor = torch.autograd.functional.jacobian(lambda a: Sim3.exp(a).log(), torch.zeros(1, 7, dtype=torch.float64, device=device))
    assert torch.allclose(J0.reshape(7, 7), torch.eye(7, dtype=torch.float64, device=device), atol=1e-12)
    a0: Tensor = 0.2 * torch.randn(1, 7, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    J: Tensor = torch.autograd.functional.jacobian(lambda a: Sim3.exp(a).log(), a0.to(device))
    assert torch.allclose(J.reshape(7, 7), torch.eye(7, dtype=torch.float64, device=device), atol=1e-3)
