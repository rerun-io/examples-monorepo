"""Per-keyframe log-scale latent of the metric prior in the global BA (ViPE ``GBA_SCALE_LATENT``).

For keyframe ``t`` let ``m_t`` be the robust (Huber) location of
``log(d_BA) - log(d_prior)`` over its valid patches, measured against the raw
prior every time so nothing compounds, and ``w_t`` its data weight (valid
patches / M, 0 below ``min_patches``). The latent ``s`` minimises

    sum_t w_t (s_t - m_t)^2 + lambda_smooth sum_t (s_{t+1} - s_t)^2 + lambda_anchor sum_t s_t^2

where keyframes with too little parallax or ``|m_t| > clamp`` keep only the
smoothness term. Then the ``w``-weighted mean of ``s`` is removed (the raw prior
still pins the metric scale of the whole trajectory) and ``s`` is clamped to
``[-clamp, clamp]``. The global BA sees ``d_prior * exp(s_t)`` for every patch
of keyframe ``t``; the local BA keeps the raw prior.

Ported from ViPE ``vipe/metrictrail/dpvo/{scale_latent,burst_fixes}.py`` and
``DPVO._update_scale_latent``/``_parallax_rows`` (dpvo.py:863-910) without the
statistics. The numbers are bit-identical to ViPE's.
"""

import torch
from jaxtyping import Bool, Float32, Float64, Int64
from lietorch_mojo import SE3
from torch import Tensor

from dpvo import projective_ops as pops
from dpvo.dpvo import DPVO

SCALE_HUBER: float = 0.2
"""Huber threshold of the per-keyframe offset ``m_t`` (log units)."""
SCALE_LAMBDA_ANCHOR: float = 1e-3
"""Weight of the zero anchor; keeps the system positive definite when every ``w_t`` is 0."""
SCALE_IRLS_ITERATIONS: int = 5
"""Reweighting iterations of the Huber location."""
PARALLAX_CAP_PX: float = 1e3
"""Non-finite or larger per-patch flows count as this many pixels."""


def robust_log_scale_offsets(
    log_ratio: Float32[Tensor, "t m"],
    valid: Bool[Tensor, "t m"],
    *,
    huber: float,
    min_patches: int,
    patches_per_frame: int,
    iterations: int = SCALE_IRLS_ITERATIONS,
) -> tuple[Float32[Tensor, "t"], Int64[Tensor, "t"], Float32[Tensor, "t"]]:
    """Huber location of each row of ``log_ratio`` by iterative reweighting.

    Args:
        log_ratio: ``log(d_BA) - log(d_prior)`` per keyframe and patch.
        valid: Entries that may be used (non-finite ones are dropped as well).
        huber: Huber threshold, ``> 0``.
        min_patches: Rows with fewer valid entries get weight 0.
        patches_per_frame: ``M``; the weight is ``count / M``.
        iterations: Reweighting iterations after the median start.

    Returns:
        ``(m, count, w)``: location, valid count and data weight per row.
    """
    if not huber > 0.0:
        raise ValueError("huber must be positive")
    valid = valid & torch.isfinite(log_ratio)
    count: Int64[Tensor, "t"] = valid.sum(dim=1)
    values: Float32[Tensor, "t m"] = torch.where(valid, log_ratio, torch.zeros_like(log_ratio))
    nan_filled: Float32[Tensor, "t m"] = torch.where(valid, log_ratio, torch.full_like(log_ratio, float("nan")))
    m: Float32[Tensor, "t"] = torch.nan_to_num(torch.nanmedian(nan_filled, dim=1).values, nan=0.0)
    for _ in range(iterations):
        deviation: Float32[Tensor, "t m"] = (values - m[:, None]).abs()
        robust: Float32[Tensor, "t m"] = torch.where(deviation <= huber, torch.ones_like(deviation), huber / deviation.clamp_min(1e-6))
        robust = robust * valid.to(robust.dtype)
        denominator: Float32[Tensor, "t"] = robust.sum(dim=1)
        candidate: Float32[Tensor, "t"] = (robust * values).sum(dim=1) / denominator.clamp_min(1e-12)
        m = torch.where(denominator > 0, candidate, m)
    w: Float32[Tensor, "t"] = torch.where(count >= min_patches, count.to(log_ratio.dtype) / float(patches_per_frame), torch.zeros_like(m))
    return m, count, w


def solve_smooth_scale(
    m: Float32[Tensor, "t"], w: Float32[Tensor, "t"], *, lambda_smooth: float, lambda_anchor: float
) -> Float64[Tensor, "t"]:
    """Solve the smoothed, anchored 1-D least squares of the module docstring (dense float64).

    Args:
        m: Per-keyframe offsets.
        w: Per-keyframe data weights (0 disables a keyframe's data term).
        lambda_smooth: Weight of the first-difference penalty, ``>= 0``.
        lambda_anchor: Weight of the zero anchor, ``> 0``.

    Returns:
        The latent ``s`` per keyframe.
    """
    if not lambda_anchor > 0.0:
        raise ValueError("lambda_anchor must be positive")
    if lambda_smooth < 0.0:
        raise ValueError("lambda_smooth must be non-negative")
    count: int = int(m.numel())
    if count == 0:
        return m.new_zeros(0, dtype=torch.float64)
    m64: Float64[Tensor, "t"] = m.to(torch.float64)
    w64: Float64[Tensor, "t"] = w.to(torch.float64)
    A: Float64[Tensor, "t t"] = torch.diag(w64 + lambda_anchor)
    if count > 1:
        index: Int64[Tensor, "t1"] = torch.arange(count - 1, device=m.device)
        A[index, index] += lambda_smooth
        A[index + 1, index + 1] += lambda_smooth
        A[index, index + 1] -= lambda_smooth
        A[index + 1, index] -= lambda_smooth
    return torch.linalg.solve(A, w64 * m64)


def gba_scale_latent(
    m: Float32[Tensor, "t"],
    w: Float32[Tensor, "t"],
    parallax_px: Float32[Tensor, "t"],
    *,
    clamp: float,
    min_parallax_px: float,
    lambda_smooth: float,
    lambda_anchor: float = SCALE_LAMBDA_ANCHOR,
) -> Float32[Tensor, "t"]:
    """Zero-mean, parallax-gated, smoothed and clamped per-keyframe log-scale latent.

    Args:
        m: Robust per-keyframe ``log(d_BA) - log(d_prior)`` (:func:`robust_log_scale_offsets`).
        w: Data weights from :func:`robust_log_scale_offsets`.
        parallax_px: Mean translation-only flow of each keyframe into its predecessor.
        clamp: Bound on ``|s|`` and gate on ``|m|``, ``> 0``.
        min_parallax_px: Keyframes below it keep only the smoothness term.
        lambda_smooth: Smoothness weight.
        lambda_anchor: Anchor weight.

    Returns:
        ``s`` per keyframe.
    """
    if not clamp > 0.0:
        raise ValueError("clamp must be positive")
    m = torch.nan_to_num(m.to(torch.float32), nan=0.0, posinf=clamp * 10.0, neginf=-clamp * 10.0)
    w = torch.nan_to_num(w.to(torch.float32), nan=0.0)
    parallax: Float32[Tensor, "t"] = torch.nan_to_num(parallax_px.to(torch.float32), nan=0.0, posinf=1e3)
    keep: Bool[Tensor, "t"] = (w > 0) & ~(parallax < min_parallax_px) & ~(m.abs() > clamp)
    w_eff: Float32[Tensor, "t"] = torch.where(keep, w, torch.zeros_like(w))
    if m.numel() == 0:
        return m.new_zeros(0)
    s: Float64[Tensor, "t"] = solve_smooth_scale(m, w_eff, lambda_smooth=lambda_smooth, lambda_anchor=lambda_anchor)
    # Remove the weighted mean so that the raw prior keeps the trajectory's metric scale
    s = s - (w_eff.to(s.dtype) * s).sum() / w_eff.to(s.dtype).sum() if float(w_eff.sum().item()) > 0.0 else torch.zeros_like(s)
    return s.clamp(-clamp, clamp).to(torch.float32)


def parallax_rows(dpvo: DPVO) -> Float32[Tensor, "t1"]:
    """Mean translation-only flow (px) of each keyframe ``1..n-1``'s patch centres into the previous keyframe.

    Args:
        dpvo: Tracker with ``n >= 2`` keyframes.

    Returns:
        One value per keyframe ``1..n-1``.
    """
    n: int = dpvo.n
    M: int = dpvo.M
    frames: Int64[Tensor, "t1"] = torch.arange(1, n, device=dpvo.device)
    kk: Int64[Tensor, "e"] = (frames[:, None] * M + torch.arange(M, device=dpvo.device)[None, :]).reshape(-1)
    ii: Int64[Tensor, "e"] = dpvo.ix[kk]
    poses: SE3 = SE3(dpvo.poses)
    at_source: Float32[Tensor, "1 e ps ps 2"] = pops.transform(poses, dpvo.patches, dpvo.intrinsics, ii, ii, kk)
    at_previous: Float32[Tensor, "1 e ps ps 2"] = pops.transform(poses, dpvo.patches, dpvo.intrinsics, ii, ii - 1, kk, tonly=True)
    flow: Float32[Tensor, "1 e"] = (at_previous - at_source)[:, :, dpvo.P // 2, dpvo.P // 2, :].norm(dim=-1)
    flow = torch.nan_to_num(flow, nan=PARALLAX_CAP_PX, posinf=PARALLAX_CAP_PX).clamp(max=PARALLAX_CAP_PX)
    return flow.view(n - 1, M).mean(dim=1)


def keyframe_scale_latent(
    dpvo: DPVO,
    disps: Float32[Tensor, "n_slots M"],
    weights: Float32[Tensor, "n_slots M"],
    *,
    clamp: float,
    min_patches: int,
    min_parallax_px: float,
    lambda_smooth: float,
) -> Float32[Tensor, "n"]:
    """The latent ``s`` of the tracker's ``n`` keyframes from the current patch depths and the raw prior.

    Args:
        dpvo: Tracker.
        disps: Raw prior inverse depth per keyframe slot and patch.
        weights: Prior weight per keyframe slot and patch.
        clamp: See :func:`gba_scale_latent`.
        min_patches: See :func:`robust_log_scale_offsets`.
        min_parallax_px: See :func:`gba_scale_latent`.
        lambda_smooth: See :func:`gba_scale_latent`.

    Returns:
        ``s`` for keyframe slots ``0..n-1``.
    """
    n: int = dpvo.n
    centre: int = dpvo.P // 2
    depth: Float32[Tensor, "n M"] = dpvo.pg.patches_[:n, :, 2, centre, centre]
    prior: Float32[Tensor, "n M"] = disps[:n]
    valid: Bool[Tensor, "n M"] = (weights[:n] > 0) & (depth > 0) & (prior > 0) & torch.isfinite(depth) & torch.isfinite(prior)
    log_ratio: Float32[Tensor, "n M"] = torch.log(depth.clamp_min(1e-6)) - torch.log(prior.clamp_min(1e-6))
    offsets: tuple[Float32[Tensor, "n"], Int64[Tensor, "n"], Float32[Tensor, "n"]] = robust_log_scale_offsets(
        log_ratio, valid, huber=SCALE_HUBER, min_patches=min_patches, patches_per_frame=dpvo.M
    )
    parallax: Float32[Tensor, "n"]
    if n >= 2:
        rows: Float32[Tensor, "t1"] = parallax_rows(dpvo)
        parallax = torch.cat([rows[:1], rows])  # keyframe 0 shares keyframe 1's parallax
    else:
        parallax = torch.zeros(n, device=dpvo.device)
    return gba_scale_latent(
        offsets[0], offsets[2], parallax, clamp=clamp, min_parallax_px=min_parallax_px, lambda_smooth=lambda_smooth
    )
