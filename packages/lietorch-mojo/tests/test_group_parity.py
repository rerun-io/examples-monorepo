"""Group API: lietorch_mojo vs upstream lietorch, SE3.

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
def test_api_forward_matches(shapes: tuple[tuple[int, ...], tuple[int, ...]], dtype: torch.dtype, device: str) -> None:
    gen: torch.Generator = torch.Generator().manual_seed(0)
    xs, ys = shapes
    x_data: Tensor = _random_se3_data(xs, dtype, device, gen)
    y_data: Tensor = _random_se3_data(ys, dtype, device, gen)
    a: Tensor = torch.randn(*ys, 6, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    p3: Tensor = (10 * torch.randn(*ys, 3, generator=gen, dtype=torch.float64)).to(device=device, dtype=dtype)
    p4: Tensor = torch.randn(*ys, 4, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    s: Tensor = torch.rand(*xs, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)

    results: dict[str, list[Tensor]] = {}
    for mod in (lietorch, lietorch_mojo):
        X = mod.SE3(x_data)
        Y = mod.SE3(y_data)
        results[mod.__name__] = [
            mod.SE3.exp(a).data,
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
            mod.SE3.InitFromVec(x_data).data,
            X.scale(s).data,
            mod.SE3.Identity(xs, device=device, dtype=dtype).data,
            mod.SE3.IdentityLike(X).data,
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


def test_other_groups_raise_not_implemented() -> None:
    for name, dim in (("SO3", 3),):
        with pytest.raises(NotImplementedError):
            getattr(lietorch_mojo, name).exp(torch.zeros(1, dim, dtype=torch.float64))


# ── gradients: lietorch vs Mojo through autograd ─────────────────────────────

GradFn = Callable[[object, list[Tensor]], Tensor]


def _composites(mod: object, X_data: Tensor, Y_data: Tensor) -> dict[str, tuple[GradFn, list[int]]]:
    """name -> (fn(inputs), shapes' last dims). Inputs are tangents/points; outputs Euclidean."""
    G = mod.SE3

    def X() -> object:
        return G(X_data)

    def Y() -> object:
        return G(Y_data)

    return {
        "exp_log": (lambda ins: G.exp(ins[0]).log(), [6]),
        "inv_log": (lambda ins: (G.exp(ins[0]) * X()).inv().log(), [6]),
        "mul_log": (lambda ins: (G.exp(ins[0]) * X() * G.exp(ins[1]) * Y()).log(), [6, 6]),
        "retr_log": (lambda ins: X().retr(ins[0]).log(), [6]),
        "adj": (lambda ins: (G.exp(ins[0]) * X()).adj(ins[1]), [6, 6]),
        "adjT": (lambda ins: (G.exp(ins[0]) * X()).adjT(ins[1]), [6, 6]),
        "act3": (lambda ins: (X() * G.exp(ins[0])).act(ins[1]), [6, 3]),
        "act4": (lambda ins: (X() * G.exp(ins[0])).act(ins[1]), [6, 4]),
        "matrix": (lambda ins: (G.exp(ins[0]) * X()).matrix(), [6]),
        "translation": (lambda ins: (G.exp(ins[0]) * X()).translation(), [6]),
        "vec": (lambda ins: (G.exp(ins[0]) * X()).vec(), [6]),
        "fromvec": (lambda ins: G.InitFromVec(_normalise_se3(ins[0])).vec(), [7]),
        "scale_log": (lambda ins: G.exp(ins[0]).scale(ins[1][..., 0]).log(), [6, 1]),
    }


def _normalise_se3(a: Tensor) -> Tensor:
    t, q = a.split([3, 4], dim=-1)
    return torch.cat([t, q / q.norm(dim=-1, keepdim=True)], dim=-1)


COMPOSITE_NAMES: tuple[str, ...] = tuple(_composites(lietorch, torch.zeros(1, 7), torch.zeros(1, 7)).keys())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", COMPOSITE_NAMES)
def test_autograd_gradients_match(name: str, dtype: torch.dtype, device: str) -> None:
    gen: torch.Generator = torch.Generator().manual_seed(2)
    batch: int = 64
    X_data: Tensor = _random_se3_data((batch,), dtype, device, gen)
    Y_data: Tensor = _random_se3_data((batch,), dtype, device, gen)
    widths: list[int] = _composites(lietorch, X_data, Y_data)[name][1]
    base: list[Tensor] = [torch.randn(batch, w, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype) for w in widths]
    if name == "scale_log":
        base[1] = base[1].abs() + 0.5
    grads: dict[str, list[Tensor]] = {}
    for mod in (lietorch, lietorch_mojo):
        fn, _ = _composites(mod, X_data, Y_data)[name]
        ins: list[Tensor] = [b.clone().requires_grad_(True) for b in base]
        out: Tensor = fn(ins)
        weight: Tensor = torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64).to(device=device, dtype=dtype)
        (out * weight).sum().backward()
        grads[mod.__name__] = [i.grad for i in ins]
    for i, (ours, ref) in enumerate(zip(grads["lietorch_mojo"], grads["lietorch"], strict=True)):
        err: float = max_error(ours, ref)
        assert err <= TOLERANCE[dtype], f"{name} input {i}: {err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
def test_lie_group_parameter_gradients_match(device: str) -> None:
    gen: torch.Generator = torch.Generator().manual_seed(4)
    X_data: Tensor = _random_se3_data((8,), torch.float64, device, gen)
    pts: Tensor = torch.randn(8, 3, generator=gen, dtype=torch.float64).to(device)
    grads: list[Tensor] = []
    for mod in (lietorch, lietorch_mojo):
        P = mod.LieGroupParameter(mod.SE3(X_data))
        loss: Tensor = (P.retr().act(pts) ** 2).sum() + P.log().sum()
        loss.backward()
        grads.append(P.grad)
    assert max_error(grads[1], grads[0]) <= TOLERANCE[torch.float64]
