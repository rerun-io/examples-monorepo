"""RxSO3 and Sim3: lietorch_mojo vs upstream lietorch.

- Raw backend ops (forward and backward, fed the same upstream gradients) on
  the input sets of `scaled_cases.py`, float32 and float64, CPU and CUDA.
- The public API forward, including broadcasting.
- Autograd gradients of composite functions, lietorch vs Mojo.

Tolerances as for SE3 (`row_error`, worst element): 1e-5 for float32, 1e-10
for float64.

float32 references come from lietorch's CPU build on every device. lietorch's
CUDA float32 Sim3 is the outlier: against a float64 truth on the same
float32 inputs it is 2-450x less accurate than its own CPU build (tiny
angles: expm 1.7e-4 vs 8.0e-5, logm 9.3e-5 vs 2.1e-7, Jinv 5.0e-5 vs 2.1e-7),
while this port gives the CPU values on both devices. The raw-op test
also checks that Mojo is never farther from the float64 truth than lietorch on
the same device.
"""

import pytest
import torch

lietorch = pytest.importorskip("lietorch", reason="needs the compiled lietorch (ai-demos build) for parity")
from conftest import require_mojo_extension  # noqa: E402

require_mojo_extension()

from collections.abc import Callable  # noqa: E402

import lietorch_backends  # noqa: E402
from conftest import DEVICES, DTYPES  # noqa: E402
from scaled_cases import (  # noqa: E402
    BACKWARD_OPS,
    DIMS,
    FORWARD_OPS,
    GROUP_IDS,
    OpInputs,
    ScaledCase,
    call_op,
    make_inputs,
    row_error,
    scaled_cases,
)
from torch import Tensor  # noqa: E402

import lietorch_mojo  # noqa: E402
from lietorch_mojo import backends  # noqa: E402

pytestmark = pytest.mark.integration

TOLERANCE: dict[torch.dtype, float] = {torch.float32: 1e-5, torch.float64: 1e-10}
CASES: dict[str, list[ScaledCase]] = {name: scaled_cases(gid) for name, gid in GROUP_IDS.items()}
CASE_PARAMS: list = [pytest.param(g, c, id=f"{g}-{c.name}") for g, cases in CASES.items() for c in cases]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", FORWARD_OPS + BACKWARD_OPS)
@pytest.mark.parametrize(("group", "case"), CASE_PARAMS)
def test_op_matches_lietorch(group: str, case: ScaledCase, op: str, dtype: torch.dtype, device: str) -> None:
    gid: int = GROUP_IDS[group]
    inputs: OpInputs = make_inputs(gid, case, dtype, device)
    ours: list[Tensor] = call_op(backends, gid, op, inputs)
    assert all(bool(torch.isfinite(o).all()) for o in ours)
    ref_device: str = "cpu" if dtype == torch.float32 else device  # module docstring
    ref: list[Tensor] = call_op(lietorch_backends, gid, op, make_inputs(gid, case, dtype, ref_device))
    err: float = row_error([o.to(ref_device) for o in ours], ref)
    assert err <= TOLERANCE[dtype], f"{group} {op} {case.name} {dtype} {device}: row error {err:.3e}"
    if dtype == torch.float32:
        f64: OpInputs = OpInputs(
            X=inputs.X.double(),
            Y=inputs.Y.double(),
            a=inputs.a.double(),
            p3=inputs.p3.double(),
            p4=inputs.p4.double(),
            grads={k: v.double() for k, v in inputs.grads.items()},
        )
        truth: list[Tensor] = [t.float() for t in call_op(lietorch_backends, gid, op, f64)]
        ours_err: float = row_error(ours, truth)
        lietorch_err: float = row_error(call_op(lietorch_backends, gid, op, inputs), truth)
        assert ours_err <= lietorch_err + TOLERANCE[dtype], f"{group} {op} {case.name} {device}: Mojo {ours_err:.3e} vs lietorch {lietorch_err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group", tuple(GROUP_IDS))
def test_backward_tail_slots_are_zero(group: str, dtype: torch.dtype, device: str) -> None:
    """Gradients w.r.t. group elements fill the first K of N slots; slot N-1 is zero (as lietorch)."""
    gid: int = GROUP_IDS[group]
    N, _ = DIMS[gid]
    inputs: OpInputs = make_inputs(gid, CASES[group][0], dtype, device)
    for op in ("logm_backward", "inv_backward", "mul_backward", "adj_backward", "adjT_backward", "act_backward", "act4_backward"):
        group_grads: list[Tensor] = [o for o in call_op(backends, gid, op, inputs) if o.shape[-1] == N]
        assert group_grads, op
        for g in group_grads:
            assert torch.all(g[:, N - 1] == 0), op


# ── public API ───────────────────────────────────────────────────────────────


def _random_data(group: str, shape: tuple[int, ...], dtype: torch.dtype, device: str, gen: torch.Generator) -> Tensor:
    q: Tensor = torch.randn(*shape, 4, generator=gen, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    s: Tensor = torch.exp(0.5 * torch.randn(*shape, 1, generator=gen, dtype=torch.float64))
    parts: list[Tensor] = [q, s] if group == "RxSO3" else [torch.randn(*shape, 3, generator=gen, dtype=torch.float64), q, s]
    return torch.cat(parts, dim=-1).to(device=device, dtype=dtype)


BROADCAST_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [((5,), (5,)), ((4, 1), (1, 6)), ((1,), (7,))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shapes", BROADCAST_SHAPES, ids=[f"{a}x{b}" for a, b in BROADCAST_SHAPES])
@pytest.mark.parametrize("group", tuple(GROUP_IDS))
def test_api_forward_matches(group: str, shapes: tuple[tuple[int, ...], tuple[int, ...]], dtype: torch.dtype, device: str) -> None:
    gen: torch.Generator = torch.Generator().manual_seed(0)
    xs, ys = shapes
    K: int = DIMS[GROUP_IDS[group]][1]
    x_data: Tensor = _random_data(group, xs, dtype, device, gen)
    y_data: Tensor = _random_data(group, ys, dtype, device, gen)
    a: Tensor = torch.randn(*ys, K, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    p3: Tensor = (10 * torch.randn(*ys, 3, generator=gen, dtype=torch.float64)).to(device=device, dtype=dtype)
    p4: Tensor = torch.randn(*ys, 4, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
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
            X.retr(a).data,
            X.adj(a),
            X.adjT(a),
            X.Jinv(a),
            X.act(p3),
            X.act(p4),
            X.matrix(),
            X.translation(),
            X.vec(),
            G.InitFromVec(x_data).data,
            G.Identity(xs, device=device, dtype=dtype).data,
            mod.cat([X, X], dim=0).data,
            X[0].data,
        ]
    for i, (ours, ref) in enumerate(zip(results["lietorch_mojo"], results["lietorch"], strict=True)):
        assert ours.shape == ref.shape, f"output {i}"
        err: float = ((ours.double() - ref.double()).abs() / ref.double().abs().clamp_min(1.0)).max().item()
        assert err <= TOLERANCE[dtype], f"{group} output {i}: {err:.3e}"


def test_shapes_and_repr_match() -> None:
    gen: torch.Generator = torch.Generator().manual_seed(1)
    for group in GROUP_IDS:
        data: Tensor = _random_data(group, (4,), torch.float64, "cpu", gen)
        for mod in (lietorch, lietorch_mojo):
            X = getattr(mod, group)(data)
            assert X.shape == torch.Size([4])
            assert X.tangent_shape == torch.Size([4, DIMS[GROUP_IDS[group]][1]])
        assert repr(getattr(lietorch_mojo, group)(data)) == repr(getattr(lietorch, group)(data))
    se3: Tensor = _random_data("Sim3", (3,), torch.float64, "cpu", gen)[..., :7]
    assert torch.equal(lietorch_mojo.Sim3(lietorch_mojo.SE3(se3)).data, lietorch.Sim3(lietorch.SE3(se3)).data)
    sim3: Tensor = _random_data("Sim3", (3,), torch.float64, "cpu", gen)
    assert torch.equal(lietorch_mojo.RxSO3(lietorch_mojo.Sim3(sim3)).data, lietorch.RxSO3(lietorch.Sim3(sim3)).data)


# ── gradients: lietorch vs Mojo through autograd ─────────────────────────────

GradFn = Callable[[list[Tensor]], Tensor]


def _composites(mod: object, group: str, X_data: Tensor, Y_data: Tensor) -> dict[str, tuple[GradFn, list[int]]]:
    G = getattr(mod, group)
    K: int = G.manifold_dim

    def X() -> object:
        return G(X_data)

    def Y() -> object:
        return G(Y_data)

    return {
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
    }


COMPOSITE_NAMES: tuple[str, ...] = tuple(_composites(lietorch, "Sim3", torch.zeros(1, 8), torch.zeros(1, 8)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", COMPOSITE_NAMES)
@pytest.mark.parametrize("group", tuple(GROUP_IDS))
def test_autograd_gradients_match(group: str, name: str, dtype: torch.dtype, device: str) -> None:
    gen: torch.Generator = torch.Generator().manual_seed(2)
    batch: int = 64
    X_data: Tensor = _random_data(group, (batch,), dtype, device, gen)
    Y_data: Tensor = _random_data(group, (batch,), dtype, device, gen)
    widths: list[int] = _composites(lietorch, group, X_data, Y_data)[name][1]
    base: list[Tensor] = [torch.randn(batch, w, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype) for w in widths]
    grads: dict[str, list[Tensor]] = {}
    # float32 reference: lietorch's CPU build (module docstring).
    ref_device: str = "cpu" if dtype == torch.float32 else device
    for mod, dev in ((lietorch, ref_device), (lietorch_mojo, device)):
        fn, _ = _composites(mod, group, X_data.to(dev), Y_data.to(dev))[name]
        ins: list[Tensor] = [b.to(dev).clone().requires_grad_(True) for b in base]
        out: Tensor = fn(ins)
        weight: Tensor = torch.randn(out.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64).to(device=dev, dtype=dtype)
        (out * weight).sum().backward()
        grads[mod.__name__] = [i.grad.cpu() for i in ins]
    for i, (ours, ref) in enumerate(zip(grads["lietorch_mojo"], grads["lietorch"], strict=True)):
        err: float = ((ours.double() - ref.double()).abs() / ref.double().abs().clamp_min(1.0)).max().item()
        assert err <= TOLERANCE[dtype], f"{group} {name} input {i}: {err:.3e}"
