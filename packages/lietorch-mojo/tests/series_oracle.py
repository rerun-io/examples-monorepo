"""float64 oracle for the SE3 left Jacobian, from Taylor series where se3.h's closed forms cancel.

se3.h's calcQ coefficients lose most of their float64 digits at small rotation
angles (e.g. (t^2 + 2 cos t - 2) / (2 t^4)), so neither lietorch nor this port
in float64 is a reference there. This oracle sums 16 series terms for t < 1
and uses the closed forms above that.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor

_TERMS: int = 16
_C1: list[float] = [(-1) ** (k + 1) / math.factorial(2 * k + 1) for k in range(1, _TERMS + 1)]  # (t - sin t) / t^3
_C2: list[float] = [(-1) ** k / math.factorial(2 * k) for k in range(2, _TERMS + 2)]  # (t^2 + 2 cos t - 2) / (2 t^4)
_C3: list[float] = [(-1) ** k * (k - 1) / math.factorial(2 * k + 1) for k in range(2, _TERMS + 2)]  # (2t - 3 sin t + t cos t) / (2 t^5)
_CJ1: list[float] = [(-1) ** k / math.factorial(2 * k + 2) for k in range(_TERMS)]  # (1 - cos t) / t^2


def _hat(v: Tensor) -> Tensor:
    x, y, z = v.unbind(-1)
    o: Tensor = torch.zeros_like(x)
    return torch.stack([torch.stack([o, -z, y], -1), torch.stack([z, o, -x], -1), torch.stack([-y, x, o], -1)], -2)


def _coef(t: Tensor, series: list[float], closed: Callable[[Tensor], Tensor]) -> Tensor:
    t2: Tensor = t * t
    r: Tensor = torch.zeros_like(t)
    for c in reversed(series):
        r = r * t2 + c
    big: Tensor = torch.where(t < 1.0, torch.ones_like(t), t)
    return torch.where(t < 1.0, r, closed(big))[..., None, None]


def se3_left_jacobian(a: Tensor) -> Tensor:
    """(n, 6) float64 tangents [tau, phi] -> (n, 6, 6) left Jacobians."""
    tau, phi = a[..., :3], a[..., 3:]
    t: Tensor = phi.norm(dim=-1)
    T, P = _hat(tau), _hat(phi)
    eye: Tensor = torch.eye(3, dtype=a.dtype).expand_as(P)
    J: Tensor = eye + _coef(t, _CJ1, lambda s: (1 - torch.cos(s)) / s**2) * P + _coef(t, _C1, lambda s: (s - torch.sin(s)) / s**3) * (P @ P)
    Q: Tensor = (
        0.5 * T
        + _coef(t, _C1, lambda s: (s - torch.sin(s)) / s**3) * (P @ T + T @ P + P @ T @ P)
        + _coef(t, _C2, lambda s: (s**2 + 2 * torch.cos(s) - 2) / (2 * s**4)) * (P @ P @ T + T @ P @ P - 3 * P @ T @ P)
        + _coef(t, _C3, lambda s: (2 * s - 3 * torch.sin(s) + s * torch.cos(s)) / (2 * s**5)) * (P @ T @ P @ P + P @ P @ T @ P)
    )
    out: Tensor = torch.zeros(a.shape[:-1] + (6, 6), dtype=a.dtype)
    out[..., :3, :3] = J
    out[..., :3, 3:] = Q
    out[..., 3:, 3:] = J
    return out
