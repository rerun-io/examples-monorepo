"""Graph policy of MetricDPVO: the keyframe-hole guard and reverse proximity loop edges.

Ported from ViPE ``vipe/metrictrail/dpvo/graph_guards.py`` and
``DPVO._append_reverse_loop_edges`` (dpvo.py:372-392) without the edge tags and
the statistics, which no numeric path reads.

- **Hole guard** (``keyframe_max_cull``): keyframe() keeps a candidate that the
  flow test would cull once at least ``keyframe_max_cull`` candidates in a row
  were culled since the last kept keyframe and their summed rotation path
  reaches ``keyframe_max_cull_rot_deg``.
- **Reverse loop edges** (``loop_reverse_edges``): for every accepted proximity
  pair (old frame ``i``, new frame ``j``) also observe frame ``j``'s patches in
  frame ``i``, when frame ``i``'s feature map stays in the ring for the edge's
  whole active lifetime.
"""

import math

import torch
from jaxtyping import Bool, Float32, Float64, Int64
from lietorch_mojo import SE3
from torch import Tensor

from dpvo.dpvo import DPVO
from dpvo.metric.config import MetricDPVOConfig


def loop_pairs(ii_frames: Int64[Tensor, "p"], jj_frames: Int64[Tensor, "p"]) -> Int64[Tensor, "q 2"]:
    """Unique ``(i, j)`` keyframe pairs, sorted lexicographically.

    Args:
        ii_frames: Source frame of one edge per loop pair candidate (every ``M``-th edge of ``edges_loop``).
        jj_frames: Target frame of the same edges.
    """
    if ii_frames.numel() == 0:
        return torch.empty(0, 2, dtype=torch.long, device=ii_frames.device)
    return torch.unique(torch.stack((ii_frames.long(), jj_frames.long()), dim=1), dim=0)


def reverse_loop_edges(
    pairs: Int64[Tensor, "q 2"], n: int, mem: int, M: int, margin: int
) -> tuple[Int64[Tensor, "r"], Int64[Tensor, "r"]]:
    """Edges that observe the new frame ``j``'s patches in the old frame ``i`` of each pair.

    A pair is used only while the old frame's feature map stays resident for the
    new edges' active lifetime, ``(n - 1 - i) < mem - margin``.

    Args:
        pairs: Accepted ``(i old, j new)`` pairs (:func:`loop_pairs`).
        n: Keyframes in the tracker.
        mem: Length of the feature-map ring.
        M: Patches per frame.
        margin: Frames an edge stays active (``removal_window + 1``).

    Returns:
        ``(kk, jj)``: the patches ``j * M + 0..M-1`` and the frame ``i`` they are observed in.
    """
    if pairs.numel() == 0:
        empty: Int64[Tensor, "0"] = torch.empty(0, dtype=torch.long, device=pairs.device)
        return empty, empty
    i: Int64[Tensor, "q"] = pairs[:, 0].long()
    j: Int64[Tensor, "q"] = pairs[:, 1].long()
    resident: Bool[Tensor, "q"] = (n - 1 - i) < (mem - margin)
    i, j = i[resident], j[resident]
    patch: Int64[Tensor, "M"] = torch.arange(M, device=pairs.device, dtype=torch.long)
    kk: Int64[Tensor, "r"] = (j[:, None] * M + patch[None, :]).reshape(-1)
    jj: Int64[Tensor, "r"] = i[:, None].expand(-1, M).reshape(-1)
    return kk, jj


def quat_angle_deg(q: Float64[Tensor, "4"], q_ref: Float64[Tensor, "4"] | None = None) -> float:
    """Rotation angle in degrees of the unit quaternion ``q`` (``x y z w``), or of ``q * q_ref^-1``.

    Float64: in float32 ``acos`` near 1 loses about 1e-4 deg per small increment.
    """
    q = q / q.norm().clamp_min(1e-12)
    d: Float64[Tensor, ""] = q[3].abs() if q_ref is None else (q * (q_ref / q_ref.norm().clamp_min(1e-12))).sum().abs()
    return math.degrees(2.0 * math.acos(float(d.clamp(max=1.0).item())))


class HoleGuard:
    """Counts the culled run since the last kept keyframe and decides when to keep a candidate anyway.

    Args:
        max_cull: Culls in a row after which the guard may keep; ``<= 0`` disables the guard.
        rot_deg: Rotation path (deg) the run must also cover.
    """

    def __init__(self, max_cull: int, rot_deg: float) -> None:
        self.max_cull: int = max_cull
        self.rot_deg: float = rot_deg
        self.enabled: bool = max_cull > 0
        self.run: int = 0
        """Candidates culled since the last kept keyframe."""
        self.rot_path_deg: float = 0.0
        """Summed rotation between consecutive culled candidates."""
        self.last_q: Float64[Tensor, "4"] | None = None
        """Quaternion of the previous culled candidate's relative pose to the last kept keyframe."""

    def should_keep(self) -> bool:
        """Keep the next candidate regardless of the flow test."""
        return self.enabled and self.run >= self.max_cull and self.rot_path_deg >= self.rot_deg

    def on_cull(self, dP_data: Float32[Tensor, "7"]) -> None:
        """A candidate was culled; ``dP_data`` is its pose relative to the last kept keyframe (``t q``)."""
        if not self.enabled:
            return
        q: Float64[Tensor, "4"] = dP_data.detach().double().reshape(-1)[3:7].clone()
        self.rot_path_deg += quat_angle_deg(q) if self.last_q is None else quat_angle_deg(q, self.last_q)
        self.run += 1
        self.last_q = q

    def on_keep(self) -> None:
        """A candidate was kept: start a new run."""
        self.run = 0
        self.rot_path_deg = 0.0
        self.last_q = None


class MetricGraphPolicy:
    """:class:`~dpvo.hooks.GraphPolicy` with the hole guard and reverse loop edges; counts the loop edges.

    Args:
        cfg: Tracker config (``keyframe_max_cull``, ``keyframe_max_cull_rot_deg``,
            ``loop_reverse_edges``, ``removal_window``).
    """

    def __init__(self, cfg: MetricDPVOConfig) -> None:
        self.hole_guard: HoleGuard = HoleGuard(cfg.keyframe_max_cull, cfg.keyframe_max_cull_rot_deg)
        self.reverse_edges: bool = cfg.loop_reverse_edges
        self.reverse_margin: int = cfg.removal_window + 1
        self.loop_edges_added: int = 0
        """Proximity loop edges appended so far (forward only)."""

    def force_keep(self, dpvo: DPVO) -> bool:
        """The hole guard's veto."""
        return self.hole_guard.should_keep()

    def on_cull(self, dpvo: DPVO, k: int) -> None:
        """Feed the culled slot's pose relative to the last kept keyframe (just stored in ``pg.delta``) to the guard."""
        if self.hole_guard.enabled:
            dP: SE3 = dpvo.pg.delta[int(dpvo.pg.tstamps_[k])][1]
            self.hole_guard.on_cull(dP.data)

    def on_keep(self, dpvo: DPVO, k: int, forced: bool) -> None:
        """Reset the guard's run."""
        self.hole_guard.on_keep()

    def extra_loop_edges(
        self, dpvo: DPVO, lii: Int64[Tensor, "e"], ljj: Int64[Tensor, "e"]
    ) -> tuple[Int64[Tensor, "r"], Int64[Tensor, "r"]] | None:
        """Count the proximity edges; with ``loop_reverse_edges``, return their reverse edges."""
        self.loop_edges_added += int(lii.numel())
        if not self.reverse_edges:
            return None
        pairs: Int64[Tensor, "q 2"] = loop_pairs(dpvo.ix[lii[:: dpvo.M]], ljj[:: dpvo.M])
        return reverse_loop_edges(pairs, dpvo.n, dpvo.mem, dpvo.M, self.reverse_margin)
