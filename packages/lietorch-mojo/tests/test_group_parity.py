"""Group API: lietorch_mojo vs upstream lietorch, SE3 and SO3.

- API surface: every public name of upstream lietorch has a counterpart with
  the same parameters (names, kinds, defaults) and the same class constants.
- Forward: methods through the public API, including broadcasting shapes.
- Backward: autograd gradients of composite functions, lietorch vs Mojo, on
  random inputs (the hard rule's "lietorch-vs-Mojo gradient comparison").

Tolerances (worst |ours - lietorch| / max(|lietorch|, 1), elementwise):
1e-5 for float32, 1e-10 for float64.
"""

import inspect
from collections.abc import Callable

import pytest
import torch

lietorch = pytest.importorskip("lietorch", reason="needs the compiled lietorch (ai-demos build) for parity")
from conftest import require_mojo_extension  # noqa: E402

require_mojo_extension()
import lietorch_backends  # noqa: E402
from conftest import DEVICES, DTYPES  # noqa: E402
from torch import Tensor  # noqa: E402

import lietorch_mojo  # noqa: E402
from lietorch_mojo import backends  # noqa: E402

pytestmark = pytest.mark.integration

TOLERANCE: dict[torch.dtype, float] = {torch.float32: 1e-5, torch.float64: 1e-10}
GROUP_NAMES: tuple[str, ...] = ("SO3", "RxSO3", "SE3", "Sim3")
# Groups with a Mojo backend: name -> (embedded dim N, manifold dim K).
BACKED_GROUPS: dict[str, tuple[int, int]] = {"SE3": (7, 6), "SO3": (4, 3)}


def max_error(ours: Tensor, ref: Tensor) -> float:
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    assert ours.dtype == ref.dtype
    if ref.numel() == 0:
        return 0.0
    return ((ours.double() - ref.double()).abs() / ref.double().abs().clamp_min(1.0)).max().item()


def _param_spec(fn: Callable[..., object]) -> list[tuple[str, inspect._ParameterKind, object]]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


# ── API surface ──────────────────────────────────────────────────────────────


def test_module_exports_match() -> None:
    for name in ("LieGroupParameter", "SO3", "RxSO3", "SE3", "Sim3", "cat", "stack"):
        assert hasattr(lietorch_mojo, name), name
    assert _param_spec(lietorch_mojo.cat) == _param_spec(lietorch.cat)
    assert _param_spec(lietorch_mojo.stack) == _param_spec(lietorch.stack)


@pytest.mark.parametrize("group_name", GROUP_NAMES)
def test_group_class_surface_matches(group_name: str) -> None:
    upstream: type = getattr(lietorch, group_name)
    ours: type = getattr(lietorch_mojo, group_name)
    for attr in ("group_name", "group_id", "manifold_dim", "embedded_dim"):
        assert getattr(ours, attr) == getattr(upstream, attr), attr
    assert torch.equal(ours.id_elem, upstream.id_elem)
    public: list[str] = [n for n in dir(upstream) if not n.startswith("_") or n in ("__mul__", "__getitem__", "__setitem__", "__init__")]
    for name in public:
        assert hasattr(ours, name), f"{group_name}.{name} missing"
        up_attr: object = inspect.getattr_static(upstream, name)
        our_attr: object = inspect.getattr_static(ours, name)
        assert type(up_attr) is type(our_attr), f"{group_name}.{name}: {type(up_attr)} vs {type(our_attr)}"
        if isinstance(up_attr, (staticmethod, classmethod)) or inspect.isfunction(up_attr):
            assert _param_spec(getattr(ours, name)) == _param_spec(getattr(upstream, name)), f"{group_name}.{name} signature"


def test_lie_group_parameter_surface_matches() -> None:
    for name in ("__new__", "__init__", "retr", "log", "inv", "adj", "__mul__", "add_", "__getitem__"):
        assert _param_spec(getattr(lietorch_mojo.LieGroupParameter, name)) == _param_spec(getattr(lietorch.LieGroupParameter, name)), name


def test_backends_module_has_every_lietorch_backend_function() -> None:
    upstream_names: list[str] = [n for n in dir(lietorch_backends) if not n.startswith("_")]
    for name in upstream_names:
        assert hasattr(backends, name), name
        # pybind11 signature line: "expm(arg0: int, arg1: torch.Tensor) -> torch.Tensor"
        doc: str = getattr(lietorch_backends, name).__doc__.splitlines()[0]
        n_args: int = doc.split("(", 1)[1].split(")")[0].count("arg")
        assert len(inspect.signature(getattr(backends, name)).parameters) == n_args, name


# ── forward through the public API ───────────────────────────────────────────


def _random_se3_data(shape: tuple[int, ...], dtype: torch.dtype, device: str, gen: torch.Generator) -> Tensor:
    t: Tensor = torch.randn(*shape, 3, generator=gen, dtype=torch.float64)
    q: Tensor = torch.randn(*shape, 4, generator=gen, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.cat([t, q], dim=-1).to(device=device, dtype=dtype)


def _random_group_data(group: str, shape: tuple[int, ...], dtype: torch.dtype, device: str, gen: torch.Generator) -> Tensor:
    """Random SE3 data, or its quaternion part for SO3 (same draws either way)."""
    data: Tensor = _random_se3_data(shape, dtype, device, gen)
    return data if group == "SE3" else data[..., 3:].contiguous()


BROADCAST_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((5,), (5,)),
    ((2, 3), (2, 3)),
    ((4, 1), (1, 6)),
    ((1, 3, 1), (2, 1, 5)),
    ((1,), (7,)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shapes", BROADCAST_SHAPES, ids=[f"{a}x{b}" for a, b in BROADCAST_SHAPES])
@pytest.mark.parametrize("group", tuple(BACKED_GROUPS))
def test_api_forward_matches(group: str, shapes: tuple[tuple[int, ...], tuple[int, ...]], dtype: torch.dtype, device: str) -> None:
    gen: torch.Generator = torch.Generator().manual_seed(0)
    xs, ys = shapes
    x_data: Tensor = _random_group_data(group, xs, dtype, device, gen)
    y_data: Tensor = _random_group_data(group, ys, dtype, device, gen)
    a: Tensor = torch.randn(*ys, BACKED_GROUPS[group][1], generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    p3: Tensor = (10 * torch.randn(*ys, 3, generator=gen, dtype=torch.float64)).to(device=device, dtype=dtype)
    p4: Tensor = torch.randn(*ys, 4, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    s: Tensor = torch.rand(*xs, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)

    results: dict[str, list[Tensor]] = {}
    for mod in (lietorch, lietorch_mojo):
        G = getattr(mod, group)
        X = G(x_data)
        Y = G(y_data)
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
            X.scale(s).data if group == "SE3" else X.data,  # scale() is SE3-only
            G.Identity(xs, device=device, dtype=dtype).data,
            G.IdentityLike(X).data,
            mod.cat([X, X], dim=0).data,
            mod.stack([X, X], dim=0).data,
            X[0].data,
            torch.stack([g.data for g in X.unbind(0)]),
            X.detach().data,
            X.view(X.shape).data,
        ]
    for i, (ours, ref) in enumerate(zip(results["lietorch_mojo"], results["lietorch"], strict=True)):
        err: float = max_error(ours, ref)
        assert err <= TOLERANCE[dtype], f"output {i}: {err:.3e}"


def test_setitem_and_shapes_match() -> None:
    gen: torch.Generator = torch.Generator().manual_seed(1)
    data: Tensor = _random_se3_data((4,), torch.float64, "cpu", gen)
    other: Tensor = _random_se3_data((2,), torch.float64, "cpu", gen)
    for mod in (lietorch, lietorch_mojo):
        X = mod.SE3(data.clone())
        X[1:3] = mod.SE3(other)
        assert torch.equal(X.data[1:3], other)
        assert X.shape == torch.Size([4])
        assert X.tangent_shape == torch.Size([4, 6])
        assert X.embedded_dim == 7 and X.manifold_dim == 6
    assert repr(lietorch_mojo.SE3(data)) == repr(lietorch.SE3(data))


# ── gradients: lietorch vs Mojo through autograd ─────────────────────────────

GradFn = Callable[[object, list[Tensor]], Tensor]


def _composites(mod: object, group: str, X_data: Tensor, Y_data: Tensor) -> dict[str, tuple[GradFn, list[int]]]:
    """name -> (fn(inputs), shapes' last dims). Inputs are tangents/points; outputs Euclidean."""
    G = getattr(mod, group)
    N, K = BACKED_GROUPS[group]

    def X() -> object:
        return G(X_data)

    def Y() -> object:
        return G(Y_data)

    composites: dict[str, tuple[GradFn, list[int]]] = {
        "exp_log": (lambda ins: G.exp(ins[0]).log(), [K]),
        "inv_log": (lambda ins: (G.exp(ins[0]) * X()).inv().log(), [K]),
        "mul_log": (lambda ins: (G.exp(ins[0]) * X() * G.exp(ins[1]) * Y()).log(), [K, K]),
        "retr_log": (lambda ins: X().retr(ins[0]).log(), [K]),
        "adj": (lambda ins: (G.exp(ins[0]) * X()).adj(ins[1]), [K, K]),
        "adjT": (lambda ins: (G.exp(ins[0]) * X()).adjT(ins[1]), [K, K]),
        "act3": (lambda ins: (X() * G.exp(ins[0])).act(ins[1]), [K, 3]),
        "act4": (lambda ins: (X() * G.exp(ins[0])).act(ins[1]), [K, 4]),
        "matrix": (lambda ins: (G.exp(ins[0]) * X()).matrix(), [K]),
        "translation": (lambda ins: (G.exp(ins[0]) * X()).translation(), [K]),
        "vec": (lambda ins: (G.exp(ins[0]) * X()).vec(), [K]),
        "fromvec": (lambda ins: G.InitFromVec(_normalise_quaternion(ins[0])).vec(), [N]),
    }
    if group == "SE3":
        composites["scale_log"] = (lambda ins: G.exp(ins[0]).scale(ins[1][..., 0]).log(), [K, 1])
    return composites


def _normalise_quaternion(a: Tensor) -> Tensor:
    """Normalise the quaternion slot of SE3 data (t, q) or SO3 data (q)."""
    rest, q = a.split([a.shape[-1] - 4, 4], dim=-1)
    return torch.cat([rest, q / q.norm(dim=-1, keepdim=True)], dim=-1)


COMPOSITE_CASES: list[tuple[str, str]] = [
    (group, name) for group, (N, _) in BACKED_GROUPS.items() for name in _composites(lietorch, group, torch.zeros(1, N), torch.zeros(1, N))
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_name", COMPOSITE_CASES, ids=[f"{g}-{n}" for g, n in COMPOSITE_CASES])
def test_autograd_gradients_match(group_name: tuple[str, str], dtype: torch.dtype, device: str) -> None:
    group, name = group_name
    gen: torch.Generator = torch.Generator().manual_seed(2)
    batch: int = 64
    X_data: Tensor = _random_group_data(group, (batch,), dtype, device, gen)
    Y_data: Tensor = _random_group_data(group, (batch,), dtype, device, gen)
    widths: list[int] = _composites(lietorch, group, X_data, Y_data)[name][1]
    base: list[Tensor] = [torch.randn(batch, w, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype) for w in widths]
    if name == "scale_log":
        base[1] = base[1].abs() + 0.5
    grads: dict[str, list[Tensor]] = {}
    for mod in (lietorch, lietorch_mojo):
        fn, _ = _composites(mod, group, X_data, Y_data)[name]
        ins: list[Tensor] = [b.clone().requires_grad_(True) for b in base]
        out: Tensor = fn(ins)
        weight: Tensor = torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64).to(device=device, dtype=dtype)
        (out * weight).sum().backward()
        grads[mod.__name__] = [i.grad for i in ins]
    for i, (ours, ref) in enumerate(zip(grads["lietorch_mojo"], grads["lietorch"], strict=True)):
        err: float = max_error(ours, ref)
        assert err <= TOLERANCE[dtype], f"{name} input {i}: {err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", tuple(BACKED_GROUPS))
def test_lie_group_parameter_gradients_match(group: str, device: str) -> None:
    gen: torch.Generator = torch.Generator().manual_seed(4)
    X_data: Tensor = _random_group_data(group, (8,), torch.float64, device, gen)
    pts: Tensor = torch.randn(8, 3, generator=gen, dtype=torch.float64).to(device)
    grads: list[Tensor] = []
    for mod in (lietorch, lietorch_mojo):
        P = mod.LieGroupParameter(getattr(mod, group)(X_data))
        loss: Tensor = (P.retr().act(pts) ** 2).sum() + P.log().sum()
        loss.backward()
        grads.append(P.grad)
    assert max_error(grads[1], grads[0]) <= TOLERANCE[torch.float64]
