"""SE3 and SO3 self-consistency and torch.autograd.gradcheck, after lietorch's run_tests.py.

These need no upstream lietorch: the CPU cases run in the unit lane, the CUDA
cases are integration tests. gradcheck runs in float64 with torch's default
tolerances (eps=1e-6, atol=1e-5, rtol=1e-3) on composite functions whose
inputs are tangent vectors or points and whose outputs are Euclidean, so the
analytical tangent-space backward is checked against finite differences.
"""

import math
import zlib
from collections.abc import Callable

import pytest
import torch
from conftest import DEVICES, require_mojo_extension
from torch import Tensor

require_mojo_extension()

from lietorch_mojo import SE3, SO3, backends  # noqa: E402
from lietorch_mojo.groups import LieGroup  # noqa: E402

D: int = SE3.manifold_dim
# Groups with a Mojo backend; tests over `GROUPS` run for each.
GROUPS: tuple[type[LieGroup], ...] = (SE3, SO3)
GROUP_IDS: tuple[str, ...] = tuple(G.group_name for G in GROUPS)


def _matv(A: Tensor, b: Tensor) -> Tensor:
    return torch.matmul(A, b[..., None])[..., 0]


def _homogeneous(p: Tensor) -> Tensor:
    return torch.cat([p, torch.ones_like(p[..., :1])], dim=-1)


def _tangent(G: type[LieGroup], translation: Tensor, phi: Tensor) -> Tensor:
    """A tangent of G from its translation part (dropped for SO3) and rotation part."""
    return torch.cat([translation, phi], dim=-1) if G is SE3 else phi


# ── forward identities (run_tests.py: test_exp_log, test_inv, test_adj, test_act) ──


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_exp_log_is_identity(G: type[LieGroup], device: str) -> None:
    a: Tensor = 0.2 * torch.randn(2, 3, 4, 5, 6, 7, G.manifold_dim, device=device, dtype=torch.float64)
    b: Tensor = G.exp(a).log()
    assert torch.allclose(a, b, atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_inv(G: type[LieGroup], device: str) -> None:
    X = G.exp(0.1 * torch.randn(2, 3, 4, 5, G.manifold_dim, device=device, dtype=torch.float64))
    a: Tensor = (X * X.inv()).log()
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_adj(G: type[LieGroup], device: str) -> None:
    X = G.exp(torch.randn(2, 3, 4, 5, G.manifold_dim, device=device, dtype=torch.float64))
    a: Tensor = torch.randn(2, 3, 4, 5, G.manifold_dim, device=device, dtype=torch.float64)
    b: Tensor = X.adj(a)
    Y1 = X * G.exp(a)
    Y2 = G.exp(b) * X
    c: Tensor = (Y1 * Y2.inv()).log()
    assert torch.allclose(c, torch.zeros_like(c), atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_act_matches_matrix(G: type[LieGroup], device: str) -> None:
    X = G.exp(torch.randn(16, G.manifold_dim, device=device, dtype=torch.float64))
    p: Tensor = torch.randn(16, 3, device=device, dtype=torch.float64)
    p1: Tensor = X.act(p)
    p2: Tensor = _matv(X.matrix(), _homogeneous(p))
    assert torch.allclose(p1, p2[..., :3], atol=1e-8)
    assert torch.allclose(X.act(_homogeneous(p)), p2, atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_adjT_is_transpose_of_adj(G: type[LieGroup], device: str) -> None:
    X = G.exp(torch.randn(16, G.manifold_dim, device=device, dtype=torch.float64))
    a: Tensor = torch.randn(16, G.manifold_dim, device=device, dtype=torch.float64)
    b: Tensor = torch.randn(16, G.manifold_dim, device=device, dtype=torch.float64)
    lhs: Tensor = (X.adj(a) * b).sum(-1)
    rhs: Tensor = (a * X.adjT(b)).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_edge_cases_round_trip(G: type[LieGroup], device: str) -> None:
    """Identity, tiny angles, angles near pi and large translations survive exp/log and inv.

    pi - 1e-6 is left out on purpose: its quaternion has |w| < EPS, where so3.h snaps the
    angle to pi, so lietorch's own exp/log round trip is off by 1.7e-2 there (the parity
    tests cover that branch against lietorch).
    """
    axis: Tensor = torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64)
    axis = axis / axis.norm()
    angles: Tensor = torch.tensor([0.0, 1e-12, 1e-8, 5e-7, 2e-6, 1e-3, math.pi - 1e-4, math.pi - 1e-3], dtype=torch.float64)
    tau: Tensor = torch.tensor([[1e4, -2e4, 3e4]], dtype=torch.float64).repeat(angles.shape[0], 1)
    a: Tensor = _tangent(G, tau, axis * angles[:, None]).to(device)
    X = G.exp(a)
    assert torch.allclose(X.log(), a, rtol=1e-9, atol=1e-6)
    eye: Tensor = (X * X.inv()).log()
    assert torch.allclose(eye, torch.zeros_like(eye), atol=1e-8)


# ── gradcheck (run_tests.py backward tests, with torch.autograd.gradcheck) ──

Composite = Callable[..., Tensor]


def _composites(device: str, G: type[LieGroup] = SE3) -> dict[str, tuple[Composite, list[Tensor]]]:
    gen: torch.Generator = torch.Generator().manual_seed(0)
    K: int = G.manifold_dim

    def randn(*shape: int, scale: float = 1.0) -> Tensor:
        return (scale * torch.randn(*shape, generator=gen, dtype=torch.float64)).to(device)

    X = G.exp(randn(2, K, scale=0.5))
    Y = G.exp(randn(2, K, scale=0.5))
    X_big = G.exp(randn(2, K, scale=5.0))
    axis: Tensor = torch.tensor([0.3, -0.4, 0.866], dtype=torch.float64)
    axis = axis / axis.norm()
    t: Tensor = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    near_pi = G.exp(_tangent(G, t, (math.pi - 1e-2) * axis)[None].to(device))
    tiny = G.exp(_tangent(G, t, 1e-7 * axis)[None].to(device))

    def tangent(n: int = 2, scale: float = 0.3) -> Tensor:
        return randn(n, K, scale=scale).requires_grad_(True)

    def zero(n: int = 2) -> Tensor:
        return torch.zeros(n, K, dtype=torch.float64, device=device, requires_grad=True)

    def fromvec(v: Tensor) -> Tensor:
        rest, q = v.split([v.shape[-1] - 4, 4], dim=-1)
        return G.InitFromVec(torch.cat([rest, q / q.norm(dim=-1, keepdim=True)], dim=-1)).vec()

    composites: dict[str, tuple[Composite, list[Tensor]]] = {
        "exp_log_at_zero": (lambda a: G.exp(a).log(), [zero()]),
        "exp_log": (lambda a: G.exp(a).log(), [tangent()]),
        "inv_log": (lambda a: (G.exp(a) * X).inv().log(), [tangent()]),
        "inv_log_at_zero": (lambda a: (G.exp(a) * X).inv().log(), [zero()]),
        "mul_log": (lambda a, b: (G.exp(a) * X * G.exp(b) * Y).log(), [tangent(), tangent()]),
        "retr_log": (lambda a: X.retr(a).log(), [tangent()]),
        "adj": (lambda a, b: (G.exp(a) * X).adj(b), [tangent(), randn(2, K).requires_grad_(True)]),
        "adjT": (lambda a, b: (G.exp(a) * X).adjT(b), [tangent(), randn(2, K).requires_grad_(True)]),
        "act3": (lambda a, p: (X_big * G.exp(a)).act(p), [zero(), randn(2, 3).requires_grad_(True)]),
        "act3_random": (lambda a, p: (X * G.exp(a)).act(p), [tangent(), randn(2, 3).requires_grad_(True)]),
        "act4": (lambda a, p: (X * G.exp(a)).act(p), [tangent(), randn(2, 4).requires_grad_(True)]),
        "matrix": (lambda a: (G.exp(a) * X).matrix(), [tangent()]),
        "translation": (lambda a: (G.exp(a) * X_big).translation(), [zero()]),
        "vec": (lambda a: (G.exp(a) * X_big).vec(), [zero()]),
        "fromvec": (fromvec, [randn(1, 2, G.embedded_dim).requires_grad_(True)]),
    }
    if G is SE3:
        composites["scale_log"] = (
            lambda a, s: SE3.exp(a).scale(s).log(),
            [tangent(1), (torch.rand(1, generator=gen, dtype=torch.float64) + 0.5).to(device).requires_grad_(True)],
        )
    composites |= {
        "near_pi_log": (lambda a: (G.exp(a) * near_pi).log(), [zero(1)]),
        "near_pi_inv_act": (lambda a, p: (G.exp(a) * near_pi).inv().act(p), [zero(1), randn(1, 3).requires_grad_(True)]),
        "tiny_log": (lambda a: (G.exp(a) * tiny).log(), [zero(1)]),
        "tiny_exp_tangent": (lambda a: G.exp(a).log(), [(1e-7 * torch.ones(1, K, dtype=torch.float64, device=device)).requires_grad_(True)]),
        "broadcast_mul_act": (
            lambda a, p: (G.exp(a)[:, None] * X[None]).act(p),
            [tangent(3), randn(3, 2, 3).requires_grad_(True)],
        ),
    }
    return composites


GRADCHECK_CASES: list[tuple[type[LieGroup], str]] = [(G, name) for G in GROUPS for name in _composites("cpu", G)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group_name", GRADCHECK_CASES, ids=[f"{G.group_name}-{n}" for G, n in GRADCHECK_CASES])
def test_gradcheck_float64(group_name: tuple[type[LieGroup], str], device: str) -> None:
    G, name = group_name
    fn, inputs = _composites(device, G)[name]
    assert torch.autograd.gradcheck(fn, tuple(inputs))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_exp_log_jacobian_is_identity(G: type[LieGroup], device: str) -> None:
    """run_tests.py test_exp_log_grad: d log(exp(a)) / da = I at a = 0 and at random a."""
    K: int = G.manifold_dim
    for a0 in (torch.zeros(1, K, dtype=torch.float64), 0.2 * torch.randn(1, K, dtype=torch.float64)):
        J: Tensor = torch.autograd.functional.jacobian(lambda a: G.exp(a).log(), a0.to(device))
        assert torch.allclose(J.reshape(K, K), torch.eye(K, dtype=torch.float64, device=device), atol=1e-8)


# ── closed-form zone just above EPS (regression for the verify-gradients findings) ──

CLOSED_ZONE_ANGLES: tuple[float, ...] = (1.01e-6, 2e-6, 1e-5, 1e-4, 1e-3)


def _closed_zone_tangents(device: str) -> Tensor:
    gen: torch.Generator = torch.Generator().manual_seed(4)
    axis: Tensor = torch.nn.functional.normalize(torch.randn(len(CLOSED_ZONE_ANGLES), 3, generator=gen, dtype=torch.float64), dim=-1)
    angles: Tensor = torch.tensor(CLOSED_ZONE_ANGLES, dtype=torch.float64)
    tau: Tensor = torch.randn(len(CLOSED_ZONE_ANGLES), 3, generator=gen, dtype=torch.float64)
    return torch.cat([tau, axis * angles[:, None]], dim=-1).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_closed_zone_exp_log_jacobian_is_identity(G: type[LieGroup], device: str) -> None:
    """The analytic gradient is right where se3.h's closed forms cancel: d log(exp(a)) / da = I to 1e-9."""
    K: int = G.manifold_dim
    for a0 in _closed_zone_tangents(device)[:, D - K :]:
        J: Tensor = torch.autograd.functional.jacobian(lambda a: G.exp(a).log(), a0[None])
        assert torch.allclose(J.reshape(K, K), torch.eye(K, dtype=torch.float64, device=device), atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_closed_zone_gradcheck_default_step(G: type[LieGroup], device: str) -> None:
    """gradcheck with torch's defaults (eps=1e-6, atol=1e-5, rtol=1e-3) just above EPS.

    With so3.h's (1 - cos) / theta^2, Exp's translation carried ~5e-11 of error at these angles
    (lietorch CPU, lietorch CUDA and this port alike), which a 1e-6 central difference turned into
    ~2.5e-5 of Jacobian noise; round 1 used lietorch's own step (eps=1e-4) here. The float64
    left_jacobian rewrite (so3.mojo) removes that error, so the default step must pass.
    """
    a: Tensor = _closed_zone_tangents(device)[:, D - G.manifold_dim :].contiguous().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: G.exp(t).log(), (a,))
    assert torch.autograd.gradcheck(lambda t: G.exp(t).act(torch.ones_like(t[..., :3])), (a,))


@pytest.mark.parametrize("device", DEVICES)
def test_closed_zone_float64_expm_backward_matches_series_oracle(device: str) -> None:
    """float64 expm_backward is within 1e-9 normwise of a series oracle (lietorch: ~1.3e-9 there)."""
    from series_oracle import se3_left_jacobian

    a: Tensor = _closed_zone_tangents(device)
    grad: Tensor = torch.randn(a.shape[0], 7, generator=torch.Generator().manual_seed(5), dtype=torch.float64).to(device)
    ours: Tensor = backends.expm_backward(3, grad, a)[0].cpu()
    truth: Tensor = (grad[:, None, :6].cpu() @ se3_left_jacobian(a.cpu()))[:, 0]
    err: Tensor = (ours - truth).norm(dim=-1) / truth.norm(dim=-1).clamp_min(1.0)
    assert err.max().item() <= 1e-9, err


@pytest.mark.parametrize("device", DEVICES)
def test_closed_zone_float64_so3_expm_backward_matches_series_oracle(device: str) -> None:
    """SO3 float64 expm_backward is within 1e-9 normwise of the series oracle's rotation block."""
    from series_oracle import se3_left_jacobian

    a: Tensor = _closed_zone_tangents(device)
    grad: Tensor = torch.randn(a.shape[0], 4, generator=torch.Generator().manual_seed(5), dtype=torch.float64).to(device)
    ours: Tensor = backends.expm_backward(1, grad, a[:, 3:].contiguous())[0].cpu()
    truth: Tensor = (grad[:, None, :3].cpu() @ se3_left_jacobian(a.cpu())[:, 3:, 3:])[:, 0]
    err: Tensor = (ours - truth).norm(dim=-1) / truth.norm(dim=-1).clamp_min(1.0)
    assert err.max().item() <= 1e-9, err


# ── EPS boundary with torch's default gradcheck step (regression for verify-gradients r2) ──

# Base rotation angles where a 1e-6 central difference lands on |phi| ~ EPS = 1e-6 (0 and 1e-9) or
# just above it (2e-6); the translation part is N(0, 1). Before the float64 left_jacobian rewrite,
# 14 composites failed here for this port and for upstream lietorch alike.
EPS_BOUNDARY_ANGLES: tuple[float, ...] = (0.0, 1e-9, 2e-6)


def _eps_boundary_composites(G: type[LieGroup]) -> dict[str, tuple[int, Composite]]:
    """name: (number of tangent inputs, fn(*tangents, p3, p4))."""
    return {
        "exp_log": (1, lambda a, p3, p4: G.exp(a).log()),
        "mul_log": (2, lambda a, b, p3, p4: (G.exp(a) * G.exp(b)).log()),
        "inv_log": (1, lambda a, p3, p4: G.exp(a).inv().log()),
        "rel_log": (2, lambda a, b, p3, p4: (G.exp(a).inv() * G.exp(b)).log()),
        "act3": (1, lambda a, p3, p4: G.exp(a).act(p3)),
        "act4": (1, lambda a, p3, p4: G.exp(a).act(p4)),
        "mul_op_pts": (1, lambda a, p3, p4: G.exp(a) * p3),
        "retr_log": (2, lambda a, b, p3, p4: G.exp(a).retr(b).log()),
        "matrix": (1, lambda a, p3, p4: G.exp(a).matrix()),
        "translation": (1, lambda a, p3, p4: G.exp(a).translation()),
        "vec": (1, lambda a, p3, p4: G.exp(a).vec()),
        "adjT_of_log": (2, lambda a, b, p3, p4: G.exp(a).adjT(G.exp(b).log())),
        "chain": (3, lambda a, b, v, p3, p4: (G.exp(a) * G.exp(b).inv()).retr(v).act(p3)),
        "inv_mul_act4": (2, lambda a, b, p3, p4: (G.exp(a).inv() * G.exp(b)).act(p4)),
    }


EPS_BOUNDARY_COMPOSITES: dict[str, tuple[int, Composite]] = _eps_boundary_composites(SE3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("angle", EPS_BOUNDARY_ANGLES)
@pytest.mark.parametrize("name", tuple(EPS_BOUNDARY_COMPOSITES))
@pytest.mark.parametrize("G", GROUPS, ids=GROUP_IDS)
def test_gradcheck_default_step_at_eps_boundary(G: type[LieGroup], name: str, angle: float, device: str) -> None:
    n_tangents, fn = _eps_boundary_composites(G)[name]
    seed_key: str = f"{name}{angle}" if G is SE3 else f"{G.group_name}{name}{angle}"
    gen: torch.Generator = torch.Generator().manual_seed(zlib.crc32(seed_key.encode()))
    tangents: list[Tensor] = []
    for _ in range(n_tangents):
        axis: Tensor = torch.nn.functional.normalize(torch.randn(2, 3, generator=gen, dtype=torch.float64), dim=-1)
        tangents.append(_tangent(G, torch.randn(2, 3, generator=gen, dtype=torch.float64), angle * axis))
    p3: Tensor = 2 * torch.randn(2, 3, generator=gen, dtype=torch.float64)
    p4: Tensor = torch.cat([p3, torch.tensor([[0.0], [1.0]], dtype=torch.float64)], dim=-1)
    inputs: tuple[Tensor, ...] = tuple(t.to(device).requires_grad_(True) for t in [*tangents, p3, p4])
    assert torch.autograd.gradcheck(fn, inputs)


@pytest.mark.parametrize("device", DEVICES)
def test_float64_exp_translation_is_accurate_just_above_eps(device: str) -> None:
    """Exp's translation J(phi) tau to a few ulps of |tau| where so3.h's (1 - cos) / theta^2 cancels."""
    from series_oracle import se3_left_jacobian

    angles: Tensor = torch.tensor([1e-6, 1.000001e-6, 1.5e-6, 2e-6, 5e-6, 1e-5, 1e-4, 1e-3, 0.5, 0.99, 1.01], dtype=torch.float64)
    gen: torch.Generator = torch.Generator().manual_seed(11)
    axis: Tensor = torch.nn.functional.normalize(torch.randn(angles.shape[0], 3, generator=gen, dtype=torch.float64), dim=-1)
    tau: Tensor = 100 * torch.randn(angles.shape[0], 3, generator=gen, dtype=torch.float64)
    a: Tensor = torch.cat([tau, axis * angles[:, None]], dim=-1)
    ours: Tensor = SE3.exp(a.to(device)).translation()[..., :3].cpu()
    truth: Tensor = (se3_left_jacobian(a)[:, :3, :3] @ tau[..., None])[..., 0]
    err: Tensor = (ours - truth).abs().amax(-1) / tau.abs().amax(-1)
    assert err.max().item() <= 8 * torch.finfo(torch.float64).eps, err
