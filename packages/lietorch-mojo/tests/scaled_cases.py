"""Input sets and op table for the RxSO3 and Sim3 parity tests.

The rotation and translation parts reuse the SE3 sets of `cases.py`
(random, large angles, identity, tiny angles, near pi, |w| < EPS, large
translations, unnormalised quaternions); each gets a scale (data) and a
log-scale (tangent). Two more sets exercise rxso3.h's `calcW` branches on
sigma: |sigma| < EPS, sigma just above EPS, and theta < EPS with sigma
away from zero.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import torch
from cases import Se3Case, se3_cases
from torch import Tensor

RXSO3_ID: int = 2
SIM3_ID: int = 4
GROUP_IDS: dict[str, int] = {"RxSO3": RXSO3_ID, "Sim3": SIM3_ID}
# group id -> (N, K)
DIMS: dict[int, tuple[int, int]] = {RXSO3_ID: (5, 4), SIM3_ID: (8, 7)}

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


@dataclass(frozen=True, slots=True)
class ScaledCase:
    """One named batch of RxSO3 or Sim3 inputs (float64, CPU)."""

    name: str
    X: Tensor
    """Group elements (B, N): RxSO3 qx qy qz qw s; Sim3 tx ty tz qx qy qz qw s."""
    a: Tensor
    """Tangent vectors (B, K): RxSO3 phi sigma; Sim3 tau phi sigma."""
    p3: Tensor
    p4: Tensor


def _assemble(group_id: int, c: Se3Case, scale: Tensor, sigma: Tensor, name: str) -> ScaledCase:
    t, q = c.X.split([3, 4], dim=-1)
    tau, phi = c.a.split([3, 3], dim=-1)
    if group_id == RXSO3_ID:
        X: Tensor = torch.cat([q, scale[:, None]], dim=-1)
        a: Tensor = torch.cat([phi, sigma[:, None]], dim=-1)
    else:
        X = torch.cat([t, q, scale[:, None]], dim=-1)
        a = torch.cat([tau, phi, sigma[:, None]], dim=-1)
    return ScaledCase(name=name, X=X, a=a, p3=c.p3, p4=c.p4)


def scaled_cases(group_id: int, n: int = 256, seed: int = 0) -> list[ScaledCase]:
    gen: torch.Generator = torch.Generator().manual_seed(seed + 100)
    out: list[ScaledCase] = []
    base: list[Se3Case] = se3_cases(n, seed)
    for c in base:
        sigma: Tensor = torch.randn(n, generator=gen, dtype=torch.float64)
        scale: Tensor = torch.exp(torch.randn(n, generator=gen, dtype=torch.float64))
        if c.name == "identity":
            sigma = torch.zeros(n, dtype=torch.float64)
            scale = torch.ones(n, dtype=torch.float64)
        out.append(_assemble(group_id, c, scale, sigma, c.name))
    # calcW's sigma branches: |sigma| < EPS, just above, moderate; random rotations.
    sig: Tensor = torch.tensor([0.0, 1e-9, -5e-7, 9e-7, 2e-6, -2e-6, 1e-3, -0.5, 2.0, -3.0], dtype=torch.float64)
    sigma = sig.repeat_interleave(math.ceil(n / sig.shape[0]))[:n]
    out.append(_assemble(group_id, base[0], torch.exp(sigma), sigma, "sigma_branches"))
    # theta < EPS with sigma away from zero (calcW's third branch).
    out.append(_assemble(group_id, base[3], torch.exp(-sigma.flip(0) - 0.7), -sigma.flip(0) - 0.7, "tiny_angle_scaled"))
    return out


@dataclass(frozen=True, slots=True)
class OpInputs:
    X: Tensor
    Y: Tensor
    a: Tensor
    p3: Tensor
    p4: Tensor
    grads: dict[str, Tensor]


def make_inputs(group_id: int, case: ScaledCase, dtype: torch.dtype, device: str, seed: int = 0) -> OpInputs:
    N, K = DIMS[group_id]
    widths: dict[str, int] = {
        "expm_backward": N,
        "logm_backward": K,
        "inv_backward": N,
        "mul_backward": N,
        "adj_backward": K,
        "adjT_backward": K,
        "act_backward": 3,
        "act4_backward": 4,
    }
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    n: int = case.X.shape[0]

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype).contiguous()

    return OpInputs(
        X=cast(case.X),
        Y=cast(case.X.roll(1, dims=0)),
        a=cast(case.a),
        p3=cast(case.p3),
        p4=cast(case.p4),
        grads={k: cast(torch.randn(n, w, generator=gen, dtype=torch.float64)) for k, w in widths.items()},
    )


def call_op(backend: ModuleType, group_id: int, op: str, inp: OpInputs) -> list[Tensor]:
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
    out: Tensor | list[Tensor] = fn(group_id, *args[op])
    return out if isinstance(out, list) else [out]


def row_error(ours: list[Tensor], ref: list[Tensor]) -> float:
    """Worst over elements of max |ours - ref| over the row / max(|ref row|, 1) (as test_backend_parity)."""
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
