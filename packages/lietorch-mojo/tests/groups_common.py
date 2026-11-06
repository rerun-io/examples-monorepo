"""Helpers shared by the four-group tests: group params (xfail where kernels are missing), random data, composites.

Import after `require_mojo_extension()`: building the params probes the extension.
"""

from collections.abc import Callable
from types import ModuleType

import pytest
import torch
from torch import Tensor

from lietorch_mojo import SE3, SO3, RxSO3, Sim3, backends
from lietorch_mojo.groups import LieGroup

GROUPS: tuple[type[LieGroup], ...] = (SO3, RxSO3, SE3, Sim3)

# Upstream lietorch's Sim3 Jacobians are series approximations: d log(Exp a) / da is off the identity by
# 3.9e-7 at |a| ~ 0.1 and by 1.4e-2 at |a| ~ 0.5 (float64), so gradcheck against finite differences fails
# there. A Mojo Sim3 that keeps lietorch's formulas for parity inherits that; one with exact Jacobians passes.
SIM3_APPROXIMATE_JACOBIANS = pytest.mark.xfail(
    strict=False, reason="Sim3: lietorch's Jacobians are series approximations (exp-log Jacobian off by 1.4e-2 at |a|=0.5)"
)


def has_kernels(group: type[LieGroup]) -> bool:
    """True when this branch's Mojo extension implements `group` (the backend raises NotImplementedError otherwise)."""
    try:
        backends.expm(group.group_id, torch.zeros(1, group.manifold_dim, dtype=torch.float64))
    except NotImplementedError:
        return False
    return True


def group_params(exact_jacobians: bool = False) -> list:
    """One pytest param per group name; groups without Mojo kernels on this branch are xfail(strict=False).

    `exact_jacobians`: the test compares gradients with finite differences, so Sim3 is xfail(strict=False)
    too (see SIM3_APPROXIMATE_JACOBIANS).
    """
    params: list = []
    for group in GROUPS:
        marks: list = []
        if not has_kernels(group):
            reason: str = f"{group.group_name}: Mojo kernels are not on this branch"
            marks.append(pytest.mark.xfail(raises=NotImplementedError, strict=False, reason=reason))
        elif exact_jacobians and group is Sim3:
            marks.append(SIM3_APPROXIMATE_JACOBIANS)
        params.append(pytest.param(group.group_name, id=group.group_name, marks=marks))
    return params


def random_group_data(group_name: str, shape: tuple[int, ...], gen: torch.Generator, dtype: torch.dtype, device: str) -> Tensor:
    """Random elements in lietorch's layout, built without the kernels: [t] q [s] with a unit quaternion and s > 0."""
    q: Tensor = torch.randn(*shape, 4, generator=gen, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    t: Tensor = torch.randn(*shape, 3, generator=gen, dtype=torch.float64)
    s: Tensor = torch.exp(0.3 * torch.randn(*shape, 1, generator=gen, dtype=torch.float64))
    parts: dict[str, list[Tensor]] = {"SO3": [q], "RxSO3": [q, s], "SE3": [t, q], "Sim3": [t, q, s]}
    return torch.cat(parts[group_name], dim=-1).to(device=device, dtype=dtype)


# Composite functions for gradient tests: inputs are tangents ("K" = manifold_dim wide) or points, outputs Euclidean.
# `mod` is lietorch or lietorch_mojo (for cat/stack); G, X, Y are that module's group class and elements.
Composite = Callable[[ModuleType, type, object, object, list[Tensor]], Tensor]
COMPOSITES: dict[str, tuple[Composite, list[int | str]]] = {
    "exp_log": (lambda mod, G, X, Y, ins: G.exp(ins[0]).log(), ["K"]),
    "inv_log": (lambda mod, G, X, Y, ins: (G.exp(ins[0]) * X).inv().log(), ["K"]),
    "mul_log": (lambda mod, G, X, Y, ins: (G.exp(ins[0]) * X * G.exp(ins[1]) * Y).log(), ["K", "K"]),
    "retr_log": (lambda mod, G, X, Y, ins: X.retr(ins[0]).log(), ["K"]),
    "adj": (lambda mod, G, X, Y, ins: (G.exp(ins[0]) * X).adj(ins[1]), ["K", "K"]),
    "adjT": (lambda mod, G, X, Y, ins: (G.exp(ins[0]) * X).adjT(ins[1]), ["K", "K"]),
    "act3": (lambda mod, G, X, Y, ins: (X * G.exp(ins[0])).act(ins[1]), ["K", 3]),
    "act4": (lambda mod, G, X, Y, ins: (X * G.exp(ins[0])).act(ins[1]), ["K", 4]),
    "matrix": (lambda mod, G, X, Y, ins: (G.exp(ins[0]) * X).matrix(), ["K"]),
    "translation": (lambda mod, G, X, Y, ins: (G.exp(ins[0]) * X).translation(), ["K"]),
    "vec": (lambda mod, G, X, Y, ins: (G.exp(ins[0]) * X).vec(), ["K"]),
    "cat_stack_log": (lambda mod, G, X, Y, ins: mod.stack([mod.cat([G.exp(ins[0]), X], 0), mod.cat([Y, G.exp(ins[1])], 0)], 0).log(), ["K", "K"]),
}


def composite_inputs(G: type[LieGroup], widths: list[int | str], batch: int, gen: torch.Generator, dtype: torch.dtype, device: str) -> list[Tensor]:
    """Inputs for a composite: 0.5-scaled Gaussian tangents ("K") and points, built in float64 then cast."""
    return [
        (0.5 * torch.randn(batch, G.manifold_dim if w == "K" else int(w), generator=gen, dtype=torch.float64)).to(device=device, dtype=dtype)
        for w in widths
    ]
