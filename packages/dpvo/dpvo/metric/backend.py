"""Terminate-time backend of MetricDPVO: final re-measurement and culled-frame interpolation.

Ported from ViPE ``DPVO._remeasure_final`` (dpvo.py:485-640),
``vipe/metrictrail/dpvo/remeasure_burnin.py`` and ``burst_fixes.culled_interp``
without the statistics.

- :class:`FinalRemeasure` (``remeasure_final="two"``): after terminate's 12
  updates, re-measure every stored and active factor with the update operator
  under the converged poses and run the metric global BA after each refresh, in
  passes of 8 and 24 iterations.
- :class:`CulledInterp` (``culled_interp``): a culled frame keeps its cull-time
  pose relative to the keyframe before it, while BA later moves the keyframe
  after it, so the trajectory snaps there. The correction of that keyframe is
  spread along the culled run instead.
"""

import torch
from jaxtyping import Bool, Float, Float32, Int64
from lietorch_mojo import SE3
from torch import Tensor

from dpvo.dpvo import DPVO
from dpvo.hooks import DPVOObserver
from dpvo.metric.config import MetricDPVOConfig

REMEASURE_PASSES: tuple[int, ...] = (8, 24)
"""Iterations (refresh + global BA) of each final re-measurement pass."""


def warm_start_active(net: Float[Tensor, "1 n_edges DIM"], active_net: Float[Tensor, "1 n_act DIM"], n_inactive: int) -> None:
    """Copy the active edges' tracking-time GRU state into rows ``n_inactive:`` of ``net``, if the shapes line up."""
    n_active: int = int(net.shape[1]) - n_inactive
    if active_net.shape[1] == n_active and active_net.shape[2] == net.shape[2] and n_active > 0:
        net[:, n_inactive:] = active_net.to(net.dtype)


class FinalRemeasure(DPVOObserver):
    """Final re-measured global BA (:meth:`on_terminate`).

    Args:
        cfg: Tracker config (``remeasure_burnin``, ``remeasure_warm_active``, ``remeasure_carry_state``).
    """

    def __init__(self, cfg: MetricDPVOConfig) -> None:
        self.burnin: int = cfg.remeasure_burnin
        self.warm_active: bool = cfg.remeasure_warm_active
        self.carry_state: bool = cfg.remeasure_carry_state

    @torch.no_grad()
    def on_terminate(self, dpvo: DPVO) -> None:
        """Re-measure inactive then active factors and solve the global BA (``t0 = 1``) after every refresh.

        Each pass zeroes the GRU state (unless ``carry_state`` past the first),
        warm-starts the active rows in the first pass, and runs ``burnin``
        refreshes without a BA first. The graph keeps the final measurements.
        """
        pg = dpvo.pg
        n_inactive: int = int(pg.ii_inac.numel())
        n_edges: int = n_inactive + int(pg.ii.numel())
        if n_edges == 0:
            return
        ii: Int64[Tensor, "n_edges"] = torch.cat((pg.ii_inac, pg.ii))
        jj: Int64[Tensor, "n_edges"] = torch.cat((pg.jj_inac, pg.jj))
        kk: Int64[Tensor, "n_edges"] = torch.cat((pg.kk_inac, pg.kk))
        target: Float32[Tensor, "1 n_edges 2"] = torch.cat((pg.target_inac, pg.target), dim=1).float().contiguous()
        weight: Float32[Tensor, "1 n_edges 2"] = torch.cat((pg.weight_inac, pg.weight), dim=1).float().contiguous()
        net: Float[Tensor, "1 n_edges DIM"] = torch.zeros(1, n_edges, dpvo.DIM, device=dpvo.device, dtype=dpvo.feature_dtype)

        for pass_id, iterations in enumerate(REMEASURE_PASSES):
            if pass_id == 0 or not self.carry_state:
                net.zero_()
            if pass_id == 0 and self.warm_active:
                warm_start_active(net, pg.net, n_inactive)
            for _ in range(self.burnin):
                dpvo.refresh_factors(ii, jj, kk, target, weight, net)
            for _ in range(iterations):
                dpvo.refresh_factors(ii, jj, kk, target, weight, net)
                pg.target_inac, pg.weight_inac = target[:, :n_inactive], weight[:, :n_inactive]
                pg.target, pg.weight = target[:, n_inactive:], weight[:, n_inactive:]
                dpvo.run_global_ba(t0=1)

        pg.target_inac = target[:, :n_inactive].contiguous()
        pg.weight_inac = weight[:, :n_inactive].contiguous()
        pg.target = target[:, n_inactive:].contiguous()
        pg.weight = weight[:, n_inactive:].contiguous()


def culled_interp(
    poses: Float32[Tensor, "n_frames 7"], kept: Bool[Tensor, "n_frames"], next_rel: dict[int, tuple[int, Float32[Tensor, "7"]]]
) -> Float32[Tensor, "n_frames 7"]:
    """Spread each kept keyframe's BA correction over the run of culled frames before it.

    For consecutive kept frames ``a < b`` with culled frames between, the
    cull-time relative pose ``dPn`` recorded when ``b - 1`` was culled chains the
    run to ``T = dPn * P_{b-1}``; the discrepancy ``D = P_b * T^-1`` is applied as
    ``P_t <- exp(alpha_t log D) * P_t`` with ``alpha_t = (t - a) / (b - a)``.

    Args:
        poses: World-to-camera poses (``t q``) of every input frame, before inversion.
        kept: Frames whose pose comes from the keyframe buffer.
        next_rel: ``{t: (t_next, dPn)}`` recorded when ``t`` was culled, with
            ``P_{t_next} = dPn * P_t`` at cull time.

    Returns:
        The corrected poses; kept poses are unchanged. A run whose last culled
        frame does not point at ``b`` keeps its poses.
    """
    out: Float32[Tensor, "n_frames 7"] = poses.clone()
    kept_idx: list[int] = torch.nonzero(kept).flatten().tolist()
    for a, b in zip(kept_idx[:-1], kept_idx[1:], strict=True):
        if b - a < 2:
            continue
        rel: tuple[int, Float32[Tensor, "7"]] | None = next_rel.get(b - 1)
        if rel is None or rel[0] != b:
            continue
        chained = SE3(rel[1].to(poses).reshape(1, 7)) * SE3(poses[b - 1].reshape(1, 7))
        assert isinstance(chained, SE3)
        discrepancy = SE3(poses[b].reshape(1, 7)) * chained.inv()
        assert isinstance(discrepancy, SE3)
        xi: Float32[Tensor, "1 6"] = discrepancy.log()
        alpha: Float32[Tensor, "run"] = (torch.arange(a + 1, b, device=poses.device, dtype=poses.dtype) - float(a)) / float(b - a)
        corrected = SE3.exp(alpha[:, None] * xi) * SE3(poses[a + 1 : b])
        assert isinstance(corrected, SE3)
        out[a + 1 : b] = corrected.data
    return out


class CulledInterp(DPVOObserver):
    """Culled-frame interpolation: records each cull's relative pose to the next frame and corrects the trajectory."""

    def __init__(self) -> None:
        self.delta_next: dict[int, tuple[int, Float32[Tensor, "7"]]] = {}
        """``{t: (t_next, dPn)}``: culled timestamp, the next buffered frame and its pose relative to ``t``."""

    def on_cull(self, dpvo: DPVO, k: int) -> None:
        """Record ``P_{k+1} * P_k^-1`` at cull time (slot ``k + 1 <= n - 3`` always exists)."""
        relative = SE3(dpvo.pg.poses_[k + 1]) * SE3(dpvo.pg.poses_[k]).inv()
        assert isinstance(relative, SE3)
        self.delta_next[int(dpvo.pg.tstamps_[k])] = (int(dpvo.pg.tstamps_[k + 1]), relative.data.clone())

    def refine_trajectory(self, dpvo: DPVO, poses: SE3) -> SE3:
        """Apply :func:`culled_interp` on the CPU; the trajectory stays on the CPU."""
        kept: Bool[Tensor, "n_frames"] = torch.zeros(dpvo.counter, dtype=torch.bool)
        kept[[int(t) for t in dpvo.traj]] = True
        return SE3(culled_interp(poses.data.detach().cpu(), kept, self.delta_next))
