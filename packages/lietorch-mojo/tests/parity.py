"""The one parity harness for SO3, RxSO3, SE3 and Sim3: what is compared with upstream lietorch, and how.

`freeze_references.py` runs these functions once on upstream lietorch and writes their
inputs and outputs to the reference .rrd (`references.py`); the parity tests run the same
functions on lietorch_mojo and compare. Both modules expose the same backend functions and
group classes, so every function here takes the backend module or the package module (and
the group class, for its group id and dims) as an argument. Nothing here imports
lietorch_mojo or lietorch.

Error metrics: `row_error` (raw ops) is, per element, max |ours - ref| over its output row
divided by max(|ref row|, 1), worst element; `max_error` (public API) is the worst
|ours - ref| / max(|ref|, 1) over all values. Tolerances: 1e-5 for float32, 1e-10 for float64.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields
from types import ModuleType
from typing import TYPE_CHECKING

import torch
from cases import OpCase
from groups_common import COMPOSITES, Composite
from torch import Tensor

if TYPE_CHECKING:
    from lietorch_mojo.groups import LieGroup

TOLERANCE: dict[torch.dtype, float] = {torch.float32: 1e-5, torch.float64: 1e-10}
DTYPE_NAMES: dict[torch.dtype, str] = {torch.float32: "float32", torch.float64: "float64"}
REFERENCE_DEVICES: tuple[str, ...] = ("cpu", "cuda")

FORWARD_OPS: tuple[str, ...] = ("expm", "logm", "inv", "mul", "adj", "adjT", "act", "act4", "as_matrix", "projector", "Jinv")
BACKWARD_OPS: tuple[str, ...] = (
    "expm_backward",
    "logm_backward",
    "inv_backward",
    "mul_backward",
    "adj_backward",
    "adjT_backward",
    "act_backward",
    "act4_backward",
)
ALL_OPS: tuple[str, ...] = FORWARD_OPS + BACKWARD_OPS

# Raw ops of RxSO3 and Sim3 in float32 compare against lietorch's CPU build on every device.
# lietorch's CUDA float32 Sim3 is the outlier: against a float64 truth on the same float32
# inputs it is 2-450x less accurate than its own CPU build (tiny angles: expm 1.7e-4 vs
# 8.0e-5, logm 9.3e-5 vs 2.1e-7, Jinv 5.0e-5 vs 2.1e-7), while this port gives the CPU values
# on both devices. Those groups' float32 raw ops are also checked against that float64 truth:
# Mojo must never be farther from it than lietorch on the same device.
F32_CPU_REFERENCE_GROUPS: tuple[str, ...] = ("RxSO3", "Sim3")


def row_error(ours: list[Tensor], ref: list[Tensor]) -> float:
    """Worst element of max |ours - ref| over its row / max(|ref row|, 1)."""
    worst: float = 0.0
    for o, r in zip(ours, ref, strict=True):
        assert o.shape == r.shape and o.dtype == r.dtype and o.device == r.device
        od: Tensor = o.double().reshape(o.shape[0], -1)
        rd: Tensor = r.double().reshape(r.shape[0], -1)
        if od.numel() == 0:
            continue
        err: Tensor = (od - rd).abs().amax(dim=1) / rd.abs().amax(dim=1).clamp_min(1.0)
        worst = max(worst, err.max().item())
    return worst


def max_error(ours: Tensor, ref: Tensor) -> float:
    """Worst |ours - ref| / max(|ref|, 1) over all values."""
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    assert ours.dtype == ref.dtype
    if ref.numel() == 0:
        return 0.0
    return ((ours.double() - ref.double()).abs() / ref.double().abs().clamp_min(1.0)).max().item()


# ── raw backend ops ──────────────────────────────────────────────────────────


def grad_widths(G: "type[LieGroup]") -> dict[str, int]:
    """Width of the upstream gradient fed to each backward op: its output row (group outputs fill all N slots, as autograd does)."""
    N, K = G.embedded_dim, G.manifold_dim
    return {
        "expm_backward": N,
        "logm_backward": K,
        "inv_backward": N,
        "mul_backward": N,
        "adj_backward": K,
        "adjT_backward": K,
        "act_backward": 3,
        "act4_backward": 4,
    }


@dataclass(frozen=True, slots=True)
class OpInputs:
    """Tensors for one op call, on one device and dtype."""

    X: Tensor
    """Group elements (B, N)."""
    Y: Tensor
    """Second group elements for mul (B, N)."""
    a: Tensor
    """Tangent vectors (B, K)."""
    p3: Tensor
    """Points (B, 3)."""
    p4: Tensor
    """Homogeneous points (B, 4)."""
    grads: dict[str, Tensor]
    """Upstream gradient per backward op name."""

    def named(self) -> dict[str, Tensor]:
        """Every tensor by field name (`grad/<op>` for the upstream gradients)."""
        out: dict[str, Tensor] = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "grads"}
        return out | {f"grad/{op}": g for op, g in self.grads.items()}

    @staticmethod
    def from_named(tensors: dict[str, Tensor], device: str) -> "OpInputs":
        def get(name: str) -> Tensor:
            return tensors[name].to(device).contiguous()

        grads: dict[str, Tensor] = {k.removeprefix("grad/"): get(k) for k in tensors if k.startswith("grad/")}
        return OpInputs(X=get("X"), Y=get("Y"), a=get("a"), p3=get("p3"), p4=get("p4"), grads=grads)

    def map(self, fn: Callable[[Tensor], Tensor]) -> "OpInputs":
        return OpInputs(X=fn(self.X), Y=fn(self.Y), a=fn(self.a), p3=fn(self.p3), p4=fn(self.p4), grads={k: fn(v) for k, v in self.grads.items()})


def make_inputs(G: "type[LieGroup]", case: OpCase, dtype: torch.dtype, device: str, seed: int = 0) -> OpInputs:
    """The op inputs of `case` cast to `dtype`; Y = X rolled by one row; upstream gradients ~ N(0, 1) from `seed`."""
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    n: int = case.X.shape[0]
    grads: dict[str, Tensor] = {
        name: torch.randn(n, width, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype) for name, width in grad_widths(G).items()
    }
    return OpInputs(
        X=case.X.to(device=device, dtype=dtype).contiguous(),
        Y=case.X.roll(1, dims=0).to(device=device, dtype=dtype).contiguous(),
        a=case.a.to(device=device, dtype=dtype).contiguous(),
        p3=case.p3.to(device=device, dtype=dtype).contiguous(),
        p4=case.p4.to(device=device, dtype=dtype).contiguous(),
        grads=grads,
    )


def call_op(backend: ModuleType, G: "type[LieGroup]", op: str, inp: OpInputs) -> list[Tensor]:
    """Call `backend.<op>` on G's group id with the op's natural arguments; always return a list of outputs."""
    fn: Callable[..., Tensor | list[Tensor]] = getattr(backend, op)
    args: dict[str, tuple[Tensor, ...]] = {
        "expm": (inp.a,),
        "logm": (inp.X,),
        "inv": (inp.X,),
        "mul": (inp.X, inp.Y),
        "adj": (inp.X, inp.a),
        "adjT": (inp.X, inp.a),
        "act": (inp.X, inp.p3),
        "act4": (inp.X, inp.p4),
        "as_matrix": (inp.X,),
        "projector": (inp.X,),
        "Jinv": (inp.X, inp.a),
        "expm_backward": (inp.grads["expm_backward"], inp.a),
        "logm_backward": (inp.grads["logm_backward"], inp.X),
        "inv_backward": (inp.grads["inv_backward"], inp.X),
        "mul_backward": (inp.grads["mul_backward"], inp.X, inp.Y),
        "adj_backward": (inp.grads["adj_backward"], inp.X, inp.a),
        "adjT_backward": (inp.grads["adjT_backward"], inp.X, inp.a),
        "act_backward": (inp.grads["act_backward"], inp.X, inp.p3),
        "act4_backward": (inp.grads["act4_backward"], inp.X, inp.p4),
    }
    out: Tensor | list[Tensor] = fn(G.group_id, *args[op])
    return out if isinstance(out, list) else [out]


def _grad(t: Tensor) -> Tensor:
    assert t.grad is not None
    return t.grad


# ── public API forward ───────────────────────────────────────────────────────

BROADCAST_SHAPES: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((5,), (5,)),
    ((2, 3), (2, 3)),
    ((4, 1), (1, 6)),
    ((1, 3, 1), (2, 1, 5)),
    ((1,), (7,)),
]


def shapes_id(shapes: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    """`(4, 1), (1, 6)` -> `4_1x1_6` (entity-path and test-id safe)."""
    return "x".join("_".join(str(d) for d in s) for s in shapes)


API_FORWARD_INPUTS: tuple[str, ...] = ("x", "y", "a", "p3", "p4", "s")


def api_forward_outputs(mod: ModuleType, group_name: str, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Every public method on X = G(x), Y = G(y) (x broadcasts against y); `scale` is SE3-only."""
    G = getattr(mod, group_name)
    x: Tensor = inputs["x"]
    X, Y = G(x), G(inputs["y"])
    a, p3, p4 = inputs["a"], inputs["p3"], inputs["p4"]
    out: dict[str, Tensor] = {
        "exp": G.exp(a).data,
        "log": X.log(),
        "inv": X.inv().data,
        "mul_op": (X * Y).data,
        "mul": X.mul(Y).data,
        "retr": X.retr(a).data,
        "adj": X.adj(a),
        "adjT": X.adjT(a),
        "Jinv": X.Jinv(a),
        "act3": X.act(p3),
        "act4": X.act(p4),
        "mul_point": X * p3,
        "matrix": X.matrix(),
        "translation": X.translation(),
        "vec": X.vec(),
        "InitFromVec": G.InitFromVec(x).data,
        "Identity": G.Identity(x.shape[:-1], device=x.device, dtype=x.dtype).data,
        "IdentityLike": G.IdentityLike(X).data,
        "cat": mod.cat([X, X], dim=0).data,
        "stack0": mod.stack([X, X], dim=0).data,
        "stack_last": mod.stack([X, X], dim=-1).data,
        "getitem": X[0].data,
        "unbind": torch.stack([g.data for g in X.unbind(0)]),
        "detach": X.detach().data,
        "view": X.view(X.shape).data,
    }
    if group_name == "SE3":
        out["scale"] = X.scale(inputs["s"]).data
    return out


# ── autograd gradients of composites ─────────────────────────────────────────


def _normalise_quaternion(a: Tensor) -> Tensor:
    """Normalise the quaternion slot of SE3 data (t, q) or SO3 data (q)."""
    rest, q = a.split([a.shape[-1] - 4, 4], dim=-1)
    return torch.cat([rest, q / q.norm(dim=-1, keepdim=True)], dim=-1)


def parity_composites(group_name: str) -> dict[str, tuple[Composite, list[int | str]]]:
    """groups_common.COMPOSITES plus `fromvec` (SE3, SO3: InitFromVec of the quaternion-normalised vector) and `scale_log` (SE3)."""
    out: dict[str, tuple[Composite, list[int | str]]] = dict(COMPOSITES)
    if group_name in ("SE3", "SO3"):
        out["fromvec"] = (lambda mod, G, X, Y, ins: G.InitFromVec(_normalise_quaternion(ins[0])).vec(), ["N"])
    if group_name == "SE3":
        out["scale_log"] = (lambda mod, G, X, Y, ins: G.exp(ins[0]).scale(ins[1][..., 0]).log(), ["K", 1])
    return out


@dataclass(frozen=True, slots=True)
class CompositeSuite:
    """One way of drawing composite inputs; the freezer draws them, the tests read them back."""

    name: str
    batch: int
    """Rows per input."""
    tangent_scale: float
    """Composite inputs are tangent_scale * N(0, 1)."""
    f32_cpu_reference_groups: tuple[str, ...]
    """Groups whose float32 reference is lietorch's CPU build on every device (see F32_CPU_REFERENCE_GROUPS)."""


# "unit": the SE3/SO3 and RxSO3/Sim3 API tests' inputs (64 rows, unit-scale tangents, float32 Sim3/RxSO3
# against lietorch's CPU build). "half": the four-group test's inputs (32 rows, 0.5-scaled tangents,
# every reference on the same device). The two references differ for float32 RxSO3/Sim3 on CUDA; both are kept.
COMPOSITE_SUITES: tuple[CompositeSuite, ...] = (
    CompositeSuite(name="unit", batch=64, tangent_scale=1.0, f32_cpu_reference_groups=F32_CPU_REFERENCE_GROUPS),
    CompositeSuite(name="half", batch=32, tangent_scale=0.5, f32_cpu_reference_groups=()),
)


def reference_device(dtype: torch.dtype, device: str, group_name: str, f32_cpu_groups: tuple[str, ...]) -> str:
    """The lietorch build a comparison uses: CPU for float32 of `f32_cpu_groups`, else the device under test."""
    return "cpu" if dtype == torch.float32 and group_name in f32_cpu_groups else device


def composite_gradients(
    mod: ModuleType, group_name: str, composite: str, inputs: dict[str, Tensor], weight: Callable[[torch.Size], Tensor]
) -> list[Tensor]:
    """Gradients of sum(composite(ins) * weight) w.r.t. each input `in<k>`; X = G(x), Y = G(y)."""
    G = getattr(mod, group_name)
    fn, _ = parity_composites(group_name)[composite]
    ins: list[Tensor] = []
    while f"in{len(ins)}" in inputs:
        ins.append(inputs[f"in{len(ins)}"].clone().requires_grad_(True))
    out: Tensor = fn(mod, G, G(inputs["x"]), G(inputs["y"]), ins)
    (out * weight(out.shape)).sum().backward()
    return [_grad(i) for i in ins]


# ── broadcasting gradients, LieGroupParameter, conversions ───────────────────


def broadcast_gradients(mod: ModuleType, group_name: str, inputs: dict[str, Tensor]) -> list[Tensor]:
    """Gradients of x (4, 1, N), y (1, 6, N), a (1, 6, K), p (1, 6, 3) through ops that broadcast x against y."""
    G = getattr(mod, group_name)
    x, y, a, p = (inputs[k].clone().requires_grad_(True) for k in ("x", "y", "a", "p"))
    X, Y = G(x), G(y)
    loss: Tensor = (X * Y).log().square().sum() + X.adj(a).square().sum() + X.act(p).square().sum() + X.retr(a).log().sum()
    loss.backward()
    return [_grad(x), _grad(y), _grad(a), _grad(p)]


def lie_group_parameter_gradient(mod: ModuleType, group_name: str, inputs: dict[str, Tensor]) -> Tensor:
    """Gradient of a LieGroupParameter through retr().act and log."""
    P = mod.LieGroupParameter(getattr(mod, group_name)(inputs["x"]))
    loss: Tensor = (P.retr().act(inputs["pts"]) ** 2).sum() + P.log().sum()
    loss.backward()
    return P.grad


SGD_NODES: int = 16
SGD_STEPS: int = 25


def sgd_trajectory(mod: ModuleType, group_name: str, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """The upstream examples/pgo optimisation (SGD + momentum on a LieGroupParameter): (first-step gradient, final data)."""
    G = getattr(mod, group_name)
    ii, jj = inputs["ii"], inputs["jj"]
    target = G(inputs["target"])
    Eij = target[ii].inv() * target[jj]
    P = mod.LieGroupParameter(G(inputs["start"].clone()))
    assert P.shape == (SGD_NODES, G.manifold_dim)
    optimizer = torch.optim.SGD([P], lr=0.05, momentum=0.5)
    grad0: Tensor | None = None
    for step in range(SGD_STEPS):
        optimizer.zero_grad()
        dE = (P[ii].inv() * P[jj]) * Eij.inv()
        loss: Tensor = dE.log().norm(dim=-1).sum() + (P[0] * target[0].inv()).log().square().sum()
        loss.backward()
        if step == 0:
            grad0 = P.grad.clone()
        optimizer.step()
    assert grad0 is not None
    return grad0, P.group.data


# Constructors that take another group (no kernels involved), plus SE3.scale. Not compared: upstream's
# Sim3(SO3) and quaternion() raise (lietorch_mojo fixes both; test_api.py covers them).
CONVERSIONS: dict[str, Callable[[ModuleType, dict[str, Tensor]], Tensor]] = {
    "SO3_of_SE3": lambda m, d: m.SO3(m.SE3(d["se3"])).data,
    "SE3_of_SO3": lambda m, d: m.SE3(m.SO3(d["so3"])).data,
    "RxSO3_of_Sim3": lambda m, d: m.RxSO3(m.Sim3(d["sim3"])).data,
    "Sim3_of_SE3": lambda m, d: m.Sim3(m.SE3(d["se3"])).data,
    "Sim3_of_Sim3": lambda m, d: m.Sim3(m.Sim3(d["sim3"])).data,
    "SE3_scale": lambda m, d: m.SE3(d["se3"]).scale(torch.full((3,), 2.0, dtype=torch.float64)).data,
}
