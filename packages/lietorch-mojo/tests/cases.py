"""Input sets for the SE3 parity tests: random data plus the edge cases of se3.h/so3.h.

Every set is built in float64 on the CPU and cast afterwards, so float32 and
float64 runs see the same values (up to rounding).
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Se3Case:
    """One named batch of SE3 inputs."""

    name: str
    """Short label used in test ids and reports."""
    X: Tensor
    """Group elements (B, 7): tx ty tz qx qy qz qw."""
    a: Tensor
    """Tangent vectors (B, 6): tau then phi."""
    p3: Tensor
    """Points (B, 3)."""
    p4: Tensor
    """Homogeneous points (B, 4)."""


def _axes(n: int, gen: torch.Generator) -> Tensor:
    v: Tensor = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    return v / v.norm(dim=-1, keepdim=True)


def _quat_from_angle(axes: Tensor, angles: Tensor) -> Tensor:
    half: Tensor = 0.5 * angles[:, None]
    return torch.cat([axes * torch.sin(half), torch.cos(half)], dim=-1)


def _points(n: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    p3: Tensor = 10.0 * torch.randn(n, 3, generator=gen, dtype=torch.float64)
    w: Tensor = torch.randn(n, 1, generator=gen, dtype=torch.float64)
    w[: max(1, n // 4)] = 1.0
    w[max(1, n // 4) : max(2, n // 2)] = 0.0
    p4: Tensor = torch.cat([10.0 * torch.randn(n, 3, generator=gen, dtype=torch.float64), w], dim=-1)
    return p3, p4


def _case(name: str, t: Tensor, q: Tensor, tau: Tensor, phi: Tensor, gen: torch.Generator) -> Se3Case:
    p3, p4 = _points(t.shape[0], gen)
    return Se3Case(name=name, X=torch.cat([t, q], dim=-1), a=torch.cat([tau, phi], dim=-1), p3=p3, p4=p4)


def se3_cases(n: int = 256, seed: int = 0) -> list[Se3Case]:
    """The input sets, each with `n` elements (float64, CPU)."""
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    cases: list[Se3Case] = []

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    # Random: unit quaternions from normalised Gaussians, unit-scale tangents.
    q: Tensor = randn(n, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    cases.append(_case("random", randn(n, 3), q, randn(n, 3), randn(n, 3), gen))

    # Large tangents (rotation angles up to ~15 rad, so several wraps past pi).
    q = randn(n, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    cases.append(_case("random_large_angle", 3.0 * randn(n, 3), q, 3.0 * randn(n, 3), 5.0 * randn(n, 3), gen))

    # Identity elements and zero tangents.
    ident: Tensor = torch.zeros(n, 4, dtype=torch.float64)
    ident[:, 3] = 1.0
    cases.append(
        _case(
            "identity",
            torch.zeros(n, 3, dtype=torch.float64),
            ident,
            torch.zeros(n, 3, dtype=torch.float64),
            torch.zeros(n, 3, dtype=torch.float64),
            gen,
        )
    )

    # Tiny rotation angles, below lietorch's EPS = 1e-6 (Taylor branches).
    angles: Tensor = torch.tensor([1e-12, 1e-10, 1e-8, 1e-7, 5e-7, 9e-7], dtype=torch.float64).repeat_interleave(math.ceil(n / 6))[:n]
    axes: Tensor = _axes(n, gen)
    cases.append(_case("tiny_angle", randn(n, 3), _quat_from_angle(axes, angles), randn(n, 3), axes * angles[:, None], gen))

    # Rotation angles near pi from both sides; w crosses zero.
    deltas: Tensor = torch.tensor([1e-2, 1e-4, 1e-6, 1e-8, 0.0, -1e-8, -1e-6, -1e-4], dtype=torch.float64).repeat_interleave(math.ceil(n / 8))[:n]
    angles = math.pi - deltas
    axes = _axes(n, gen)
    cases.append(_case("near_pi", randn(n, 3), _quat_from_angle(axes, angles), randn(n, 3), axes * angles[:, None], gen))

    # |w| < EPS branch of SO3::Log (quaternion w exactly tiny or zero, both signs).
    v: Tensor = _axes(n, gen)
    w: Tensor = torch.tensor([0.0, 1e-7, -1e-7, 5e-7, -9e-7], dtype=torch.float64).repeat_interleave(math.ceil(n / 5))[:n]
    q = torch.cat([v * torch.sqrt(1.0 - w * w)[:, None], w[:, None]], dim=-1)
    cases.append(_case("w_tiny", randn(n, 3), q, randn(n, 3), randn(n, 3), gen))

    # Large translations.
    q = randn(n, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    cases.append(_case("large_translation", 1e4 * randn(n, 3), q, 1e4 * randn(n, 3), randn(n, 3), gen))

    # Non-unit quaternions in the data (lietorch normalises on every read).
    q = 3.0 * randn(n, 4)
    cases.append(_case("unnormalised", randn(n, 3), q, randn(n, 3), randn(n, 3), gen))
    return cases


# Inputs where float32 is ill-conditioned: rotation angles in (EPS, 1e-2), where
# lietorch's float32 Jacobian coefficients (theta - sin theta) / theta^3 etc.
# cancel catastrophically. Tested by comparing both implementations to a float64
# reference instead of to each other.
def se3_ill_conditioned_f32_case(n: int = 256, seed: int = 1) -> Se3Case:
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    angles: Tensor = torch.logspace(-5.9, -2, n, dtype=torch.float64)
    axes: Tensor = _axes(n, gen)
    t: Tensor = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    tau: Tensor = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    return _case("small_angle_f32", t, _quat_from_angle(axes, angles), tau, axes * angles[:, None], gen)
