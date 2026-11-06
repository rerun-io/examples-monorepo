"""The generic Python layer on all four groups against upstream lietorch: forward, broadcasting, gradients, LieGroupParameter.

Groups whose Mojo kernels are not on this branch are xfail(strict=False,
raises=NotImplementedError). Group conversions need no kernels and always run.

Tolerances (worst |ours - lietorch| / max(|lietorch|, 1), elementwise):
1e-5 for float32, 1e-10 for float64.
"""

import pytest
import torch

lietorch = pytest.importorskip("lietorch", reason="needs the compiled lietorch (ai-demos build) for parity")
from conftest import DEVICES, DTYPES, require_mojo_extension  # noqa: E402

require_mojo_extension()
from groups_common import COMPOSITES, composite_inputs, group_params, random_group_data  # noqa: E402
from torch import Tensor  # noqa: E402

import lietorch_mojo  # noqa: E402

pytestmark = pytest.mark.integration

TOLERANCE: dict[torch.dtype, float] = {torch.float32: 1e-5, torch.float64: 1e-10}
GROUPS: list = group_params()
MODULES: tuple = (lietorch, lietorch_mojo)


def max_error(ours: Tensor, ref: Tensor) -> float:
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    assert ours.dtype == ref.dtype
    if ref.numel() == 0:
        return 0.0
    return ((ours.double() - ref.double()).abs() / ref.double().abs().clamp_min(1.0)).max().item()


BROADCAST_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [((5,), (5,)), ((4, 1), (1, 6)), ((1, 3, 1), (2, 1, 5)), ((1,), (7,))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("shapes", BROADCAST_SHAPES, ids=[f"{a}x{b}" for a, b in BROADCAST_SHAPES])
def test_api_forward_matches(shapes: tuple[tuple[int, ...], tuple[int, ...]], name: str, dtype: torch.dtype, device: str) -> None:
    gen: torch.Generator = torch.Generator().manual_seed(0)
    xs, ys = shapes
    K: int = getattr(lietorch, name).manifold_dim
    x_data: Tensor = random_group_data(name, xs, gen, dtype, device)
    y_data: Tensor = random_group_data(name, ys, gen, dtype, device)
    a: Tensor = torch.randn(*ys, K, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    p3: Tensor = (10 * torch.randn(*ys, 3, generator=gen, dtype=torch.float64)).to(device=device, dtype=dtype)
    p4: Tensor = torch.randn(*ys, 4, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)

    results: dict[str, list[Tensor]] = {}
    for mod in MODULES:
        G = getattr(mod, name)
        X, Y = G(x_data), G(y_data)
        results[mod.__name__] = [
            G.exp(a).data,
            X.log(),
            X.inv().data,
            (X * Y).data,
            X.mul(Y).data,
            X.retr(a).data,
            X.adj(a),
            X.adjT(a),
            X.Jinv(a),
            X.act(p3),
            X.act(p4),
            X * p3,
            X.matrix(),
            X.translation(),
            X.vec(),
            G.InitFromVec(x_data).data,
            G.Identity(xs, device=device, dtype=dtype).data,
            G.IdentityLike(X).data,
            mod.cat([X, X], dim=0).data,
            mod.stack([X, X], dim=-1).data,
            X[0].data,
            torch.stack([g.data for g in X.unbind(0)]),
        ]
    for i, (ours, ref) in enumerate(zip(results["lietorch_mojo"], results["lietorch"], strict=True)):
        err: float = max_error(ours, ref)
        assert err <= TOLERANCE[dtype], f"output {i}: {err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("composite", list(COMPOSITES))
def test_autograd_gradients_match(composite: str, name: str, dtype: torch.dtype, device: str) -> None:
    gen: torch.Generator = torch.Generator().manual_seed(2)
    batch: int = 32
    x_data: Tensor = random_group_data(name, (batch,), gen, dtype, device)
    y_data: Tensor = random_group_data(name, (batch,), gen, dtype, device)
    fn, widths = COMPOSITES[composite]
    base: list[Tensor] = composite_inputs(getattr(lietorch, name), widths, batch, gen, dtype, device)
    grads: dict[str, list[Tensor]] = {}
    for mod in MODULES:
        G = getattr(mod, name)
        ins: list[Tensor] = [b.clone().requires_grad_(True) for b in base]
        out: Tensor = fn(mod, G, G(x_data), G(y_data), ins)
        weight: Tensor = torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64).to(device=device, dtype=dtype)
        (out * weight).sum().backward()
        grads[mod.__name__] = [i.grad for i in ins]
    for i, (ours, ref) in enumerate(zip(grads["lietorch_mojo"], grads["lietorch"], strict=True)):
        err: float = max_error(ours, ref)
        assert err <= TOLERANCE[dtype], f"{composite} input {i}: {err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS)
def test_broadcast_gradients_match(name: str, device: str) -> None:
    """Gradients of operands that broadcast ((4, 1) against (1, 6)) are reduced the same way as upstream."""
    gen: torch.Generator = torch.Generator().manual_seed(4)
    K: int = getattr(lietorch, name).manifold_dim
    x_data: Tensor = random_group_data(name, (4, 1), gen, torch.float64, device)
    y_data: Tensor = random_group_data(name, (1, 6), gen, torch.float64, device)
    a0: Tensor = torch.randn(1, 6, K, generator=gen, dtype=torch.float64).to(device)
    p0: Tensor = torch.randn(1, 6, 3, generator=gen, dtype=torch.float64).to(device)
    grads: list[list[Tensor]] = []
    for mod in MODULES:
        G = getattr(mod, name)
        x, y, a, p = (t.clone().requires_grad_(True) for t in (x_data, y_data, a0, p0))
        X, Y = G(x), G(y)
        loss: Tensor = (X * Y).log().square().sum() + X.adj(a).square().sum() + X.act(p).square().sum() + X.retr(a).log().sum()
        loss.backward()
        grads.append([x.grad, y.grad, a.grad, p.grad])
    for i, (ref, ours) in enumerate(zip(*grads, strict=True)):
        assert max_error(ours, ref) <= TOLERANCE[torch.float64], f"input {i}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUPS)
def test_lie_group_parameter_sgd_trajectory_matches(name: str, device: str) -> None:
    """The upstream examples/pgo optimisation (SGD + momentum on a LieGroupParameter) follows the same path."""
    gen: torch.Generator = torch.Generator().manual_seed(5)
    K: int = getattr(lietorch, name).manifold_dim
    target_data: Tensor = random_group_data(name, (16,), gen, torch.float64, device)
    start_data: Tensor = random_group_data(name, (16,), gen, torch.float64, device)
    ii: Tensor = torch.randint(0, 16, (40,), generator=gen).to(device)
    jj: Tensor = torch.randint(0, 16, (40,), generator=gen).to(device)
    finals: list[Tensor] = []
    grads: list[Tensor] = []
    for mod in MODULES:
        G = getattr(mod, name)
        target = G(target_data)
        Eij = target[ii].inv() * target[jj]
        P = mod.LieGroupParameter(G(start_data.clone()))
        assert P.shape == (16, K)
        optimizer = torch.optim.SGD([P], lr=0.05, momentum=0.5)
        for step in range(25):
            optimizer.zero_grad()
            dE = (P[ii].inv() * P[jj]) * Eij.inv()
            loss: Tensor = dE.log().norm(dim=-1).sum() + (P[0] * target[0].inv()).log().square().sum()
            loss.backward()
            if step == 0:
                grads.append(P.grad.clone())
            optimizer.step()
        finals.append(P.group.data)
    assert max_error(grads[1], grads[0]) <= TOLERANCE[torch.float64]
    assert max_error(finals[1], finals[0]) <= 1e-8


def test_group_conversions_match() -> None:
    """Constructors that take another group (no kernels involved).

    Not compared: upstream's Sim3(SO3) and quaternion() raise (lietorch_mojo fixes both; test_api.py covers them).
    """
    gen: torch.Generator = torch.Generator().manual_seed(6)
    se3: Tensor = random_group_data("SE3", (3,), gen, torch.float64, "cpu")
    sim3: Tensor = random_group_data("Sim3", (3,), gen, torch.float64, "cpu")
    so3: Tensor = random_group_data("SO3", (3,), gen, torch.float64, "cpu")
    for build in (
        lambda m: m.SO3(m.SE3(se3)).data,
        lambda m: m.SE3(m.SO3(so3)).data,
        lambda m: m.RxSO3(m.Sim3(sim3)).data,
        lambda m: m.Sim3(m.SE3(se3)).data,
        lambda m: m.Sim3(m.Sim3(sim3)).data,
        lambda m: m.SE3(se3).scale(torch.full((3,), 2.0, dtype=torch.float64)).data,
    ):
        assert torch.equal(build(lietorch_mojo), build(lietorch))
