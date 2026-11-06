"""Input sets for the raw backend-op parity references: random data plus the edge cases of se3.h, so3.h and rxso3.h.

Every set is built in float64 on the CPU and cast afterwards, so float32 and
float64 runs see the same values (up to rounding). RxSO3 and Sim3 reuse the
SE3 sets for their rotation and translation parts; each gets a scale (data)
and a log-scale (tangent). Two more sets exercise rxso3.h's `calcW` branches
on sigma: |sigma| < EPS, sigma just above EPS, and theta < EPS with sigma
away from zero.

`freeze_references.py` builds these sets once from upstream lietorch; the
tests read the frozen rows back from the reference .rrd.
"""

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class OpCase:
    """One named batch of group inputs (float64, CPU), laid out as lietorch stores them.

    SE3: X (B, 7) = tx ty tz qx qy qz qw, a (B, 6) = tau phi. SO3: X (B, 4) = q, a (B, 3) = phi.
    RxSO3: X (B, 5) = q s, a (B, 4) = phi sigma. Sim3: X (B, 8) = t q s, a (B, 7) = tau phi sigma.
    """

    name: str
    """Short label used in test ids and reports."""
    X: Tensor
    """Group elements (B, N)."""
    a: Tensor
    """Tangent vectors (B, K)."""
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


def _case(name: str, t: Tensor, q: Tensor, tau: Tensor, phi: Tensor, gen: torch.Generator) -> OpCase:
    p3, p4 = _points(t.shape[0], gen)
    return OpCase(name=name, X=torch.cat([t, q], dim=-1), a=torch.cat([tau, phi], dim=-1), p3=p3, p4=p4)


def se3_cases(n: int = 256, seed: int = 0) -> list[OpCase]:
    """The input sets, each with `n` elements (float64, CPU)."""
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    cases: list[OpCase] = []

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
def se3_ill_conditioned_f32_case(n: int = 256, seed: int = 1) -> OpCase:
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    angles: Tensor = torch.logspace(-5.9, -2, n, dtype=torch.float64)
    axes: Tensor = _axes(n, gen)
    t: Tensor = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    tau: Tensor = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    return _case("small_angle_f32", t, _quat_from_angle(axes, angles), tau, axes * angles[:, None], gen)


def so3_case(case: OpCase) -> OpCase:
    """The rotation part of an SE3 case: X = its quaternions (B, 4), a = its phi (B, 3); points unchanged."""
    return replace(case, X=case.X[:, 3:].contiguous(), a=case.a[:, 3:].contiguous())


def so3_cases(n: int = 256, seed: int = 0) -> list[OpCase]:
    """`se3_cases` reduced to SO3 (large_translation is dropped: its rotation part repeats `random`)."""
    return [so3_case(c) for c in se3_cases(n, seed) if c.name != "large_translation"]


def _scaled(group_name: str, c: OpCase, scale: Tensor, sigma: Tensor, name: str) -> OpCase:
    """An RxSO3 or Sim3 case from an SE3 case plus a scale (data) and a log-scale (tangent)."""
    t, q = c.X.split([3, 4], dim=-1)
    tau, phi = c.a.split([3, 3], dim=-1)
    if group_name == "RxSO3":
        return OpCase(name=name, X=torch.cat([q, scale[:, None]], dim=-1), a=torch.cat([phi, sigma[:, None]], dim=-1), p3=c.p3, p4=c.p4)
    return OpCase(name=name, X=torch.cat([t, q, scale[:, None]], dim=-1), a=torch.cat([tau, phi, sigma[:, None]], dim=-1), p3=c.p3, p4=c.p4)


def scaled_cases(group_name: str, n: int = 256, seed: int = 0) -> list[OpCase]:
    """The SE3 sets with a scale and log-scale each (RxSO3 or Sim3), plus the two `calcW` sigma-branch sets."""
    gen: torch.Generator = torch.Generator().manual_seed(seed + 100)
    out: list[OpCase] = []
    base: list[OpCase] = se3_cases(n, seed)
    for c in base:
        sigma: Tensor = torch.randn(n, generator=gen, dtype=torch.float64)
        scale: Tensor = torch.exp(torch.randn(n, generator=gen, dtype=torch.float64))
        if c.name == "identity":
            sigma = torch.zeros(n, dtype=torch.float64)
            scale = torch.ones(n, dtype=torch.float64)
        out.append(_scaled(group_name, c, scale, sigma, c.name))
    # calcW's sigma branches: |sigma| < EPS, just above, moderate; random rotations.
    sig: Tensor = torch.tensor([0.0, 1e-9, -5e-7, 9e-7, 2e-6, -2e-6, 1e-3, -0.5, 2.0, -3.0], dtype=torch.float64)
    sigma = sig.repeat_interleave(math.ceil(n / sig.shape[0]))[:n]
    out.append(_scaled(group_name, base[0], torch.exp(sigma), sigma, "sigma_branches"))
    # theta < EPS with sigma away from zero (calcW's third branch).
    out.append(_scaled(group_name, base[3], torch.exp(-sigma.flip(0) - 0.7), -sigma.flip(0) - 0.7, "tiny_angle_scaled"))
    return out


def op_cases(group_name: str) -> list[OpCase]:
    """The raw-op input sets of one group: SE3 and SO3 from `se3_cases`/`so3_cases`, RxSO3 and Sim3 from `scaled_cases`."""
    if group_name == "SE3":
        return se3_cases()
    if group_name == "SO3":
        return so3_cases()
    return scaled_cases(group_name)


def ill_conditioned_case(group_name: str) -> OpCase:
    """The float32 small-angle set (SE3 and SO3 only; see `se3_ill_conditioned_f32_case`)."""
    case: OpCase = se3_ill_conditioned_f32_case()
    return case if group_name == "SE3" else so3_case(case)
