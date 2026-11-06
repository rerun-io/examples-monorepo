"""The generic Python layer on all four groups: identities, broadcasting, cat/stack, LieGroupParameter, gradcheck.

No upstream lietorch needed: CPU cases run in the unit lane, CUDA cases are
integration tests. Groups whose Mojo kernels are not on this branch are
xfail(strict=False, raises=NotImplementedError); once the kernels land the
same tests run for real. gradcheck runs in float64 with torch's default
tolerances on composites whose inputs and outputs are Euclidean.
"""

import pytest
import torch
from conftest import DEVICES, require_mojo_extension
from torch import Tensor

require_mojo_extension()

from groups_common import COMPOSITES, composite_inputs, group_params, random_group_data  # noqa: E402

import lietorch_mojo  # noqa: E402
from lietorch_mojo.broadcasting import broadcast_inputs  # noqa: E402
from lietorch_mojo.groups import LieGroup  # noqa: E402

F64: torch.dtype = torch.float64
GROUPS: list = group_params()
# gradcheck and Jinv-by-finite-differences on Sim3 need exact Jacobians (see groups_common.SIM3_APPROXIMATE_JACOBIANS).
GROUPS_EXACT_JACOBIANS: list = group_params(exact_jacobians=True)


def _group(name: str) -> type[LieGroup]:
    return getattr(lietorch_mojo, name)


def _homogeneous(p: Tensor) -> Tensor:
    return torch.cat([p, torch.ones_like(p[..., :1])], dim=-1)


def _small_tangent(G: type[LieGroup], *shape: int, device: str, scale: float = 0.5, seed: int = 0) -> Tensor:
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    return (scale * torch.randn(*shape, G.manifold_dim, generator=gen, dtype=F64)).to(device)


# ── forward identities (upstream run_tests.py, every group) ──────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS)
def test_exp_log_inv_adj_identities(name: str, device: str) -> None:
    G = _group(name)
    a: Tensor = _small_tangent(G, 2, 3, 4, device=device, scale=0.2)
    assert torch.allclose(G.exp(a).log(), a, atol=1e-8)
    X = G.exp(_small_tangent(G, 2, 3, 4, device=device, seed=1))
    assert torch.allclose((X * X.inv()).log(), torch.zeros_like(a), atol=1e-8)
    b: Tensor = _small_tangent(G, 2, 3, 4, device=device, seed=2)
    c: Tensor = ((X * G.exp(b)) * (G.exp(X.adj(b)) * X).inv()).log()
    assert torch.allclose(c, torch.zeros_like(c), atol=1e-8)
    # adjT is the transpose of adj: <adjT(a), b> == <a, adj(b)>
    assert torch.allclose((X.adjT(a) * b).sum(-1), (a * X.adj(b)).sum(-1), atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS)
def test_act_matrix_translation_agree(name: str, device: str) -> None:
    G = _group(name)
    X = G.exp(_small_tangent(G, 5, device=device, scale=1.0))
    p: Tensor = torch.randn(5, 3, dtype=F64, device=device)
    M: Tensor = X.matrix()
    assert M.shape == (5, 4, 4)
    p4: Tensor = torch.cat([p, torch.randn(5, 1, dtype=F64, device=device)], dim=-1)
    assert torch.allclose(X.act(p), (M @ _homogeneous(p)[..., None])[..., :3, 0], atol=1e-10)
    assert torch.allclose(X * p, X.act(p))
    assert torch.allclose(X.act(p4), (M @ p4[..., None])[..., 0], atol=1e-10)
    assert torch.allclose(X.translation(), M[..., :, 3], atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS_EXACT_JACOBIANS)
def test_jinv_is_the_derivative_of_log_under_left_perturbation(name: str, device: str) -> None:
    """log(Exp(e a) X) = log(X) + e Jinv(X) a + O(e^2), checked by central differences."""
    G = _group(name)
    X = G.exp(_small_tangent(G, 6, device=device, scale=0.8))
    a: Tensor = _small_tangent(G, 6, device=device, scale=1.0, seed=3)
    eps: float = 1e-6
    fd: Tensor = ((G.exp(eps * a) * X).log() - (G.exp(-eps * a) * X).log()) / (2 * eps)
    assert torch.allclose(X.Jinv(a), fd, atol=1e-6)


@pytest.mark.parametrize("name", GROUPS)
def test_constructors_and_conversions(name: str) -> None:
    G = _group(name)
    E = G.Identity(2, 3, dtype=F64)
    assert E.shape == (2, 3) and E.tangent_shape == (2, 3, G.manifold_dim)
    assert torch.equal(E.data[1, 2], G.id_elem.double())
    assert torch.equal(E.log(), torch.zeros(2, 3, G.manifold_dim, dtype=F64))
    assert torch.equal(G.IdentityLike(E).data, E.data)
    X = G.Random(4, sigma=0.3, dtype=F64)
    assert X.shape == (4,) and X.dtype == F64 and X.data.shape[-1] == G.embedded_dim
    assert torch.allclose(X.quaternion().norm(dim=-1), torch.ones(4, dtype=F64))
    assert torch.equal(G.InitFromVec(X.vec()).data, X.data)
    assert repr(X) == f"{name}: size=torch.Size([4]), device=cpu, dtype=torch.float64"


@pytest.mark.parametrize("name", GROUPS)
def test_cat_stack_index_view_unbind_setitem(name: str) -> None:
    G = _group(name)
    gen: torch.Generator = torch.Generator().manual_seed(5)
    X = G(random_group_data(name, (2, 3), gen, F64, "cpu"))
    Y = G(random_group_data(name, (2, 3), gen, F64, "cpu"))
    for dim in (0, 1):
        C = lietorch_mojo.cat([X, Y], dim=dim)
        assert type(C) is G and torch.equal(C.data, torch.cat([X.data, Y.data], dim=dim))
        S = lietorch_mojo.stack([X, Y], dim=dim)
        assert type(S) is G and torch.equal(S.data, torch.stack([X.data, Y.data], dim=dim))
    assert torch.equal(X[1, ::2].data, X.data[1, ::2])
    assert X.view((6,)).shape == (6,)
    assert [U.shape for U in X.unbind(1)] == [torch.Size([2])] * 3
    Z = G(X.data.clone())
    Z[0] = Y[0]
    assert torch.equal(Z.data[0], Y.data[0]) and torch.equal(Z.data[1], X.data[1])
    # cat/stack keep autograd: gradients reach every input.
    a: Tensor = torch.zeros(2, G.manifold_dim, dtype=F64, requires_grad=True)
    b: Tensor = torch.zeros(3, G.manifold_dim, dtype=F64, requires_grad=True)
    lietorch_mojo.cat([G.exp(a), G.exp(b)], dim=0).log().sum().backward()
    assert torch.allclose(a.grad, torch.ones_like(a)) and torch.allclose(b.grad, torch.ones_like(b))


# ── broadcasting ─────────────────────────────────────────────────────────────

BROADCAST_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [((4, 1), (1, 5)), ((1, 3, 1), (2, 1, 5)), ((1,), (6,)), ((3, 2), (3, 2))]


def test_broadcast_inputs_follows_lietorch_rules() -> None:
    x: Tensor = torch.randn(4, 1, 7)
    y: Tensor = torch.randn(1, 5, 6)
    (x1, y1), shape = broadcast_inputs(x, y)
    assert shape == (4, 5) and x1.shape == (20, 7) and y1.shape == (20, 6)
    assert torch.equal(x1.view(4, 5, 7), x.expand(4, 5, 7)) and torch.equal(y1.view(4, 5, 6), y.expand(4, 5, 6))
    (x2,), shape2 = broadcast_inputs(torch.randn(2, 3, 7), None)
    assert shape2 == (2, 3) and x2.shape == (6, 7)
    # lietorch does not prepend missing dimensions: both operands need the same rank.
    with pytest.raises(AssertionError):
        broadcast_inputs(torch.randn(4, 7), torch.randn(2, 4, 6))
    with pytest.raises(AssertionError):
        broadcast_inputs(torch.randn(3, 7), torch.randn(4, 6))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("shapes", BROADCAST_SHAPES, ids=[f"{a}x{b}" for a, b in BROADCAST_SHAPES])
def test_broadcast_forward_and_backward_match_explicit_expand(name: str, shapes: tuple[tuple[int, ...], tuple[int, ...]], device: str) -> None:
    """A broadcast binary op equals the op on expanded operands; gradients of broadcast inputs sum over the expanded dims."""
    G = _group(name)
    xs, ys = shapes
    out_shape: tuple[int, ...] = tuple(max(n, m) for n, m in zip(xs, ys, strict=True))
    gen: torch.Generator = torch.Generator().manual_seed(6)
    X_data: Tensor = random_group_data(name, xs, gen, F64, device)
    Y_data: Tensor = random_group_data(name, ys, gen, F64, device)
    a: Tensor = torch.randn(*ys, G.manifold_dim, generator=gen, dtype=F64).to(device)
    p: Tensor = torch.randn(*ys, 3, generator=gen, dtype=F64).to(device)

    def run(expand: bool) -> list[Tensor]:
        x: Tensor = X_data.clone().requires_grad_(True)
        y: Tensor = Y_data.clone().requires_grad_(True)
        aa: Tensor = a.clone().requires_grad_(True)
        pp: Tensor = p.clone().requires_grad_(True)
        X, Y = G(x), G(y)
        if expand:  # contiguous copies: like upstream, unary ops `view` their input, which an expanded tensor refuses
            X, Y = G(x.expand(*out_shape, -1).contiguous()), G(y.expand(*out_shape, -1).contiguous())
            A, P = aa.expand(*out_shape, -1).contiguous(), pp.expand(*out_shape, -1).contiguous()
        else:
            A, P = aa, pp
        outs: list[Tensor] = [(X * Y).log(), X.adj(A), X.adjT(A), X.act(P), X.retr(A).log()]
        sum(o.square().sum() for o in outs).backward()
        return outs + [x.grad, y.grad, aa.grad, pp.grad]

    for i, (ours, ref) in enumerate(zip(run(expand=False), run(expand=True), strict=True)):
        assert ours.shape == ref.shape, (i, ours.shape, ref.shape)
        assert torch.allclose(ours, ref, atol=1e-9), f"output {i}"


# ── LieGroupParameter ────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS)
def test_lie_group_parameter_methods(name: str, device: str) -> None:
    G = _group(name)
    gen: torch.Generator = torch.Generator().manual_seed(7)
    X = G(random_group_data(name, (4,), gen, F64, device))
    Y = G(random_group_data(name, (4,), gen, F64, device))
    P = lietorch_mojo.LieGroupParameter(X)
    assert isinstance(P, torch.Tensor) and P.requires_grad and P.is_leaf
    assert P.shape == (4, G.manifold_dim) and P.dtype == F64 and P.device.type == device
    assert torch.equal(P.retr().data, X.retr(torch.zeros(4, G.manifold_dim, dtype=F64, device=device)).data)
    assert torch.allclose(P.log(), X.log(), atol=1e-12)
    assert torch.allclose(P.inv().data, X.inv().data, atol=1e-12)
    a: Tensor = torch.randn(4, G.manifold_dim, generator=gen, dtype=F64).to(device)
    assert torch.allclose(P.adj(a), X.adj(a), atol=1e-12)
    assert torch.allclose((P * Y).data, (X * Y).data, atol=1e-12)
    assert torch.allclose((P * lietorch_mojo.LieGroupParameter(Y)).data, (X * Y).data, atol=1e-12)
    assert torch.allclose(P[1:3].data, X[1:3].data, atol=1e-12)
    # add_ is a left retraction of the group; the tangent tensor itself stays zero.
    P.add_(a, alpha=-0.5)
    assert torch.allclose(P.group.data, (G.exp(-0.5 * a) * X).data, atol=1e-12)
    assert torch.equal(P.detach(), torch.zeros_like(P))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS)
def test_lie_group_parameter_gradient_is_the_tangent_gradient(name: str, device: str) -> None:
    """P.grad = d f(Exp(d) X) / d d at d = 0, the same gradient as differentiating through `retr`."""
    G = _group(name)
    gen: torch.Generator = torch.Generator().manual_seed(8)
    X = G(random_group_data(name, (5,), gen, F64, device))
    Z = G(random_group_data(name, (5,), gen, F64, device))
    pts: Tensor = torch.randn(5, 3, generator=gen, dtype=F64).to(device)

    def loss(Xr: LieGroup) -> Tensor:
        return (Xr * Z).log().square().sum() + Xr.act(pts).square().sum()

    P = lietorch_mojo.LieGroupParameter(X)
    loss(P.retr()).backward()
    d: Tensor = torch.zeros(5, G.manifold_dim, dtype=F64, device=device, requires_grad=True)
    loss(X.retr(d)).backward()
    assert torch.allclose(P.grad, d.grad, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS)
def test_lie_group_parameter_sgd_momentum_converges(name: str, device: str) -> None:
    """The upstream examples/pgo pattern: SGD with momentum on a LieGroupParameter (torch takes the add_ path, not foreach)."""
    G = _group(name)
    gen: torch.Generator = torch.Generator().manual_seed(9)
    target = G(random_group_data(name, (8,), gen, F64, device))
    start = G.exp(0.5 * torch.randn(8, G.manifold_dim, generator=gen, dtype=F64).to(device)) * target
    ii: Tensor = torch.arange(8, device=device)
    P = lietorch_mojo.LieGroupParameter(start)
    optimizer = torch.optim.SGD([P], lr=0.2, momentum=0.5)
    losses: list[float] = []
    for _ in range(150):
        optimizer.zero_grad()
        loss: Tensor = (P[ii] * target.inv()).log().square().sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    final: Tensor = (P.group * target.inv()).log()
    assert losses[0] > 0.1 and final.abs().max().item() < 1e-6
    assert torch.equal(P.detach(), torch.zeros_like(P))


# ── gradcheck (float64, torch's default tolerances) on every group ──────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS_EXACT_JACOBIANS)
@pytest.mark.parametrize("composite", list(COMPOSITES))
def test_gradcheck(composite: str, name: str, device: str) -> None:
    G = _group(name)
    gen: torch.Generator = torch.Generator().manual_seed(10)
    X = G(random_group_data(name, (3,), gen, F64, device))
    Y = G(random_group_data(name, (3,), gen, F64, device))
    fn, widths = COMPOSITES[composite]
    ins: list[Tensor] = [t.requires_grad_(True) for t in composite_inputs(G, widths, 3, gen, F64, device)]
    assert torch.autograd.gradcheck(lambda *t: fn(lietorch_mojo, G, X, Y, list(t)), ins)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS_EXACT_JACOBIANS)
def test_gradcheck_through_broadcasting(name: str, device: str) -> None:
    G = _group(name)
    gen: torch.Generator = torch.Generator().manual_seed(11)
    X = G(random_group_data(name, (3, 1), gen, F64, device))
    a: Tensor = (0.5 * torch.randn(3, 1, G.manifold_dim, generator=gen, dtype=F64)).to(device).requires_grad_(True)
    b: Tensor = (0.5 * torch.randn(1, 4, G.manifold_dim, generator=gen, dtype=F64)).to(device).requires_grad_(True)
    p: Tensor = torch.randn(1, 4, 3, generator=gen, dtype=F64).to(device).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: (G.exp(a) * X * G.exp(b)).log(), [a, b])
    assert torch.autograd.gradcheck(lambda a, p: (G.exp(a) * X).act(p), [a, p])
    assert torch.autograd.gradcheck(lambda a, b: (G.exp(a) * X).adj(b), [a, b])
