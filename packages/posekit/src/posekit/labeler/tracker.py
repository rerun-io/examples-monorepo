"""Machine propagation of hand boxes along the 5 fps grid (MEgATrack section 4.2, the reviewer acts only on failures).

For every (cam, side) and grid time the machine box is, in this order:

1. ``reviewed``: the stored label at that exact (t, cam, side) (its box, or None for ``no_hand``/``rejected``).
2. ``chained``: the track of the same (cam, side) at the previous grid time had a box (any origin): the EgoHand
   proposal of this camera, of ANY side, with the highest IoU >= ``CHAIN_IOU`` to that box. The track keeps its
   side, so a hand EgoHand mis-sides for a few frames stays on its track.
   ``held``: no such proposal for ONE grid step holds the previous box; a second miss ends the chain.
3. ``detected``: no chain: the best EgoHand proposal of that side with score >= ``DETECT_SCORE``. This is how a
   box "propagates across views": by EgoHand's per-view side (no 3D fit runs on aarch64).
4. ``none``: no hand.

A ``no_hand`` label ends the chain at its time; a ``rejected`` label hides only that time (the chain passes through).
One proposal feeds at most one track per frame: reviewed boxes claim the proposals they overlap (IoU >= ``CLAIM_IOU``),
then chains take proposals greedily by IoU, then detections take what is left.

``propagate`` is a pure, deterministic function of (proposals, stored labels). ``TrackCache`` keeps the computed
prefix of one segment and recomputes from the earliest time a label write touched.
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from posekit.labeler.queue import ProposalProvider, iou
from posekit.labeler.schema import HAND_SIDES, Box, HandSide, Proposal, TrackBox, TrackFrame, ViewLabel
from posekit.labeler.store import LabelStore

CHAIN_IOU: float = 0.3
"""IoU a proposal needs with the previous box to continue a chain."""
DETECT_SCORE: float = 0.5
"""EgoHand score from which a proposal starts a track when there is no chain."""
CLAIM_IOU: float = 0.5
"""IoU with a reviewed box from which a proposal belongs to that box (no other track may take it)."""
MAX_HELD: int = 1
"""Grid steps a chain may hold its box without a matching proposal."""
STEPS_PER_LOCK: int = 50
"""Grid times ``TrackCache`` computes per hold of its lock."""

TrackKey: TypeAlias = tuple[int, HandSide]
"""(cam, side)."""


@dataclass(frozen=True, slots=True)
class Chain:
    """The state a track carries to the next grid time."""

    box: Box
    """Last box of the track (a held step keeps it)."""
    held: int
    """Consecutive held steps so far."""


@dataclass(frozen=True, slots=True)
class TrackStep:
    """The machine boxes of one grid time and the chains they leave for the next."""

    frame: TrackFrame
    """Boxes of every (cam, side)."""
    chains: dict[TrackKey, Chain]
    """Live chains after this time."""


def track_step(t_ns: int, cams: list[int], proposals: list[Proposal], labels: list[ViewLabel], chains: dict[TrackKey, Chain]) -> TrackStep:
    """Propagate the tracks one grid time; see the module docstring.

    Args:
        t_ns: Grid time, ns.
        cams: Camera indices, rig order.
        proposals: Every proposal at ``t_ns`` (only EgoHand ones are used).
        labels: Stored labels at ``t_ns``.
        chains: Live chains after the previous grid time.
    """
    stored: dict[TrackKey, ViewLabel] = {(label.cam, label.side): label for label in labels}
    boxes: list[TrackBox] = []
    next_chains: dict[TrackKey, Chain] = {}
    for cam in cams:
        candidates: list[Proposal] = [p for p in proposals if p.model == "egohand" and p.cam == cam]
        claimed: set[int] = set()
        decided: dict[HandSide, TrackBox] = {}
        for side in HAND_SIDES:
            label: ViewLabel | None = stored.get((cam, side))
            if label is None:
                continue
            decided[side] = TrackBox(cam, side, label.box, "reviewed", None)
            if label.box is not None:
                next_chains[(cam, side)] = Chain(label.box, 0)
                claimed.update(k for k, p in enumerate(candidates) if iou(p.box, label.box) >= CLAIM_IOU)
            elif label.status == "rejected" and (cam, side) in chains:
                next_chains[(cam, side)] = chains[(cam, side)]
        chained_sides: list[HandSide] = [side for side in HAND_SIDES if side not in decided and (cam, side) in chains]
        pairs: list[tuple[float, int, int]] = []
        for side_index, side in enumerate(HAND_SIDES):
            if side not in chained_sides:
                continue
            previous: Box = chains[(cam, side)].box
            for k, p in enumerate(candidates):
                overlap: float = iou(previous, p.box)
                if k not in claimed and overlap >= CHAIN_IOU:
                    pairs.append((overlap, side_index, k))
        for _, side_index, k in sorted(pairs, key=lambda pair: (-pair[0], pair[1], pair[2])):
            side = HAND_SIDES[side_index]
            if side in decided or k in claimed:
                continue
            claimed.add(k)
            decided[side] = TrackBox(cam, side, candidates[k].box, "chained", candidates[k].score)
            next_chains[(cam, side)] = Chain(candidates[k].box, 0)
        for side in chained_sides:
            chain: Chain = chains[(cam, side)]
            if side not in decided and chain.held < MAX_HELD:
                decided[side] = TrackBox(cam, side, chain.box, "held", None)
                next_chains[(cam, side)] = Chain(chain.box, chain.held + 1)
        for side in HAND_SIDES:
            if side in decided:
                continue
            confident: list[int] = [k for k, p in enumerate(candidates) if k not in claimed and p.side == side and p.score >= DETECT_SCORE]
            if not confident:
                decided[side] = TrackBox(cam, side, None, "none", None)
                continue
            best: int = max(confident, key=lambda k: (candidates[k].score, -k))
            claimed.add(best)
            decided[side] = TrackBox(cam, side, candidates[best].box, "detected", candidates[best].score)
            next_chains[(cam, side)] = Chain(candidates[best].box, 0)
        boxes.extend(decided[side] for side in HAND_SIDES)
    return TrackStep(TrackFrame(t_ns, boxes), next_chains)


def propagate(
    times_ns: list[int], cams: list[int], proposals_at: Callable[[int], list[Proposal]], labels_at: Callable[[int], list[ViewLabel]]
) -> list[TrackFrame]:
    """The machine track of every grid time, from the first one.

    Args:
        times_ns: The grid, ascending, ns.
        cams: Camera indices, rig order.
        proposals_at: Grid time -> every proposal at it.
        labels_at: Grid time -> stored labels at it.
    """
    frames: list[TrackFrame] = []
    chains: dict[TrackKey, Chain] = {}
    for t_ns in times_ns:
        step: TrackStep = track_step(t_ns, cams, proposals_at(t_ns), labels_at(t_ns), chains)
        frames.append(step.frame)
        chains = step.chains
    return frames


class TrackCache:
    """The machine track of one segment, computed lazily from the first grid time and kept until a label write."""

    def __init__(self, segment_id: str, *, times_ns: list[int], cams: list[int], proposals: ProposalProvider, store: LabelStore) -> None:
        """Bind the cache to a segment's grid, proposals and labels.

        Args:
            segment_id: Catalog segment id.
            times_ns: The 5 fps grid, ascending.
            cams: Camera indices, rig order.
            proposals: Proposal source.
            store: The segment's labels.
        """
        self.segment_id: str = segment_id
        self.times_ns: list[int] = sorted(times_ns)
        self.cams: list[int] = cams
        self.proposals: ProposalProvider = proposals
        self.store: LabelStore = store
        self._index: dict[int, int] = {t: i for i, t in enumerate(self.times_ns)}
        self._steps: list[TrackStep] = []
        """Computed prefix: ``_steps[i]`` is grid time ``times_ns[i]``."""
        self._lock = threading.Lock()

    def index(self, t_ns: int) -> int:
        """Grid index of ``t_ns``.

        Raises:
            ValueError: If ``t_ns`` is not a grid time.
        """
        found: int | None = self._index.get(t_ns)
        if found is None:
            raise ValueError(f"t_ns {t_ns} is not on the 5 fps grid of {self.segment_id}")
        return found

    def frames(self, t0_ns: int, n: int) -> list[TrackFrame]:
        """The machine track of ``n`` consecutive grid times from ``t0_ns`` (fewer at the end of the segment)."""
        if n < 1:
            raise ValueError("n must be at least 1")
        start: int = self.index(t0_ns)
        stop: int = min(start + n, len(self.times_ns))
        while True:
            # A few steps per lock hold, so a request is not stuck behind the background warm-up of a long segment.
            with self._lock:
                if len(self._steps) >= stop:
                    return [step.frame for step in self._steps[start:stop]]
                for _ in range(min(STEPS_PER_LOCK, stop - len(self._steps))):
                    t_ns: int = self.times_ns[len(self._steps)]
                    chains: dict[TrackKey, Chain] = self._steps[-1].chains if self._steps else {}
                    self._steps.append(track_step(t_ns, self.cams, self.proposals.proposals(self.segment_id, t_ns), self.store.get(t_ns), chains))

    def frame(self, t_ns: int) -> TrackFrame:
        """The machine track at one grid time."""
        return self.frames(t_ns, 1)[0]

    def invalidate_from(self, t_ns: int) -> None:
        """Forget every computed time at or after ``t_ns`` (call after the label write at ``t_ns`` has landed)."""
        with self._lock:
            del self._steps[bisect_left(self.times_ns, t_ns) :]
