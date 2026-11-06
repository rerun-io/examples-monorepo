"""Helpers shared by the four-group tests and the reference freezer: group names, random data, composites.

Nothing here imports lietorch_mojo, so `freeze_references.py` can use the same
data builders and composites with upstream lietorch.
"""

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lietorch_mojo.groups import LieGroup

GROUP_NAMES: tuple[str, ...] = ("SO3", "RxSO3", "SE3", "Sim3")


def group_params(exact_jacobians: bool = False) -> list[str]:
    """The group names to parametrise over.

    `exact_jacobians`: the test compares gradients with finite differences at unit-scale tangents, so Sim3 is
    left out. Its Jacobians are sim3.h's truncated series (exp-log Jacobian off by 1.4e-2 at |a| = 0.5);
    test_rxso3_sim3_gradcheck.py gradchecks Sim3 on small tangents instead.
    """
    return [name for name in GROUP_NAMES if not (exact_jacobians and name == "Sim3")]


def homogeneous(p: Tensor) -> Tensor:
    """Points (..., 3) as homogeneous points (..., 4) with w = 1."""
    return torch.cat([p, torch.ones_like(p[..., :1])], dim=-1)


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


def composite_inputs(G: "type[LieGroup]", widths: list[int | str], batch: int, gen: torch.Generator, dtype: torch.dtype, device: str) -> list[Tensor]:
    """Inputs for a composite: 0.5-scaled Gaussian tangents ("K") and points, built in float64 then cast."""
    return [
        (0.5 * torch.randn(batch, G.manifold_dim if w == "K" else int(w), generator=gen, dtype=torch.float64)).to(device=device, dtype=dtype)
        for w in widths
    ]
