"""The map's points by host keyframe, grouped into blocks the live map draws again only when they change.

Drawing the whole map at every cadence grew a recording with clip length x map size (1.0 GB of map points on a
24-minute walk), and reading it whole each second cost O(map). The back end hands out only what changed since a
revision (``Vio.map_delta``); :class:`MapPointBlocks` keeps each host's points, and :meth:`MapPointBlocks.redraw` gives
the blocks of ``keyframes_per_block`` hosts (in the order they joined) whose points changed. Arrays only: no Rerun and
no compiled extension, so the rules are tested on arrays.
"""

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float32, Int64
from numpy import ndarray

KEYFRAMES_PER_BLOCK: int = 32
"""Host keyframes per block: one entity per block keeps the viewer's per-entity cost low (~47 entities on a 24-minute
walk) while a local BA or a loop closure redraws only the blocks it moved."""
MAP_POINT_TOLERANCE_M: float = 0.001
"""A host's points count as moved when one of them moved farther than this (Euclidean) since they were last taken, or
its point set changed."""

HostPoints = tuple[Int64[ndarray, " q"], Float32[ndarray, "q 3"]]
"""One host's point ids and world positions, metres, in the back end's order (camera, then id)."""


@dataclass(slots=True)
class MapPointBlocks:
    """Each host keyframe's points as last taken, and the blocks to draw again.

    A host's stored points are those drawn, or those waiting in a marked block: a delta that moves them by no more than
    ``tolerance_m`` leaves them as they are, so slow drift still adds up against what was drawn.
    """

    keyframes_per_block: int = KEYFRAMES_PER_BLOCK
    """Host keyframes a block holds (:data:`KEYFRAMES_PER_BLOCK`)."""
    tolerance_m: float = MAP_POINT_TOLERANCE_M
    """Largest point move, metres, that leaves a host as drawn (:data:`MAP_POINT_TOLERANCE_M`)."""
    revision: int = 0
    """The map revision of the newest delta taken: the next read's ``since``."""
    joined: int = 0
    """Hosts that ever got a block; the next one's place. A host keeps its block for the whole run, so block entity
    paths stay stable as keyframes leave the map."""
    block_of_host: dict[int, int] = field(default_factory=dict)
    """The block of each host in the map that has had points, by keyframe timestamp."""
    hosts_of_block: list[list[int]] = field(default_factory=list)
    """The hosts of each block, in the order they joined it."""
    host_points: dict[int, HostPoints] = field(default_factory=dict)
    """Each host's points as drawn or waiting in a marked block; hosts with none are left out."""
    marked: set[int] = field(default_factory=set)
    """Blocks to draw again at the next :meth:`redraw`."""

    def apply(
        self,
        revision: int,
        keyframe_t_ns: Int64[ndarray, " k"],
        changed_hosts: Int64[ndarray, " c"],
        point_counts: Int64[ndarray, " c"],
        point_ids: Int64[ndarray, " p"],
        points_world: Float32[ndarray, "p 3"],
    ) -> None:
        """Take one map delta (``slam_rs._core.MapDelta``'s fields): drop the hosts that left the map, replace each
        changed host's points when they moved, and mark the blocks either touched. A delta no newer than the last one
        taken (Realtime hands the newest posted one out again until the back end posts another) changes nothing.

        Args:
            revision: The delta's map revision.
            keyframe_t_ns: The map's keyframes now.
            changed_hosts: The hosts whose points moved since the read's revision, in id order.
            point_counts: How many of the points below each changed host has.
            point_ids: The changed hosts' point ids, host by host.
            points_world: Their world positions, metres.
        """
        if revision <= self.revision:
            return
        self.revision = revision
        present: set[int] = set(keyframe_t_ns.tolist())
        for host in [host for host in self.block_of_host if host not in present]:
            block: int = self.block_of_host.pop(host)
            self.hosts_of_block[block].remove(host)
            if self.host_points.pop(host, None) is not None:
                self.marked.add(block)
        ends: Int64[ndarray, " c"] = np.cumsum(point_counts)
        for host, start, end in zip(changed_hosts.tolist(), (ends - point_counts).tolist(), ends.tolist(), strict=True):
            if start == end:
                if self.host_points.pop(host, None) is not None:
                    self.marked.add(self.block_of_host[host])
                continue
            ids: Int64[ndarray, " q"] = point_ids[start:end]
            positions: Float32[ndarray, "q 3"] = points_world[start:end]
            held: HostPoints | None = self.host_points.get(host)
            if held is not None and np.array_equal(held[0], ids):
                moved_sq: float = float(np.square(held[1] - positions).sum(axis=1).max())
                if moved_sq <= self.tolerance_m * self.tolerance_m:
                    continue
            if host not in self.block_of_host:
                block = self.joined // self.keyframes_per_block
                self.joined += 1
                if block == len(self.hosts_of_block):
                    self.hosts_of_block.append([])
                self.block_of_host[host] = block
                self.hosts_of_block[block].append(host)
            # Copies, not views: a view would keep the whole delta alive while this host stays unchanged.
            self.host_points[host] = (ids.copy(), positions.copy())
            self.marked.add(self.block_of_host[host])

    def redraw(self) -> list[tuple[int, Int64[ndarray, " q"], Float32[ndarray, "q 3"]]]:
        """The marked blocks, ascending, with their hosts' points in join order; a block left with none comes back
        empty. Clears the marks."""
        redraw: list[tuple[int, Int64[ndarray, " q"], Float32[ndarray, "q 3"]]] = []
        for block in sorted(self.marked):
            parts: list[HostPoints] = [self.host_points[host] for host in self.hosts_of_block[block] if host in self.host_points]
            if parts:
                redraw.append((block, np.concatenate([ids for ids, _ in parts]), np.concatenate([positions for _, positions in parts])))
            else:
                redraw.append((block, np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.float32)))
        self.marked.clear()
        return redraw
