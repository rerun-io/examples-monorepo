"""Machine propagation of hand boxes along the 5 fps grid (MEgATrack section 4.2, the reviewer acts only on failures).

For every (cam, side) and grid time the machine box is, in this order:

1. ``reviewed``: the stored label at that exact (t, cam, side) (its box, or None for ``no_hand``/``rejected``).
2. ``chained``: the track of the same (cam, side) at the previous grid time had a box (any origin): the EgoHand
   proposal of this camera, of ANY side, with the highest IoU >= ``CHAIN_IOU`` to that box. The track keeps its
   side, so a hand EgoHand mis-sides for a few frames stays on its track.
   ``held``: no such proposal for ONE grid step holds the previous box; a second miss ends the chain.
   ``tracked`` (only with a visual tracker, ``BoxCarrier``): a chain that started from a reviewed box (a person's or
   agent's label, or an EgoHand/hands-layer proposal a reviewer picked; ``tracker``/``range`` labels only continue
   such a chain, they never start one) does not hold on a miss: the visual tracker, initialised on the last frame
   where the chain had a box, carries that box to this frame. An EgoHand proposal with IoU >= ``CHAIN_IOU`` to the
   carried box takes over (origin ``chained``, a re-snap). The chain ends after ``CARRY_STEPS`` carried steps without
   a re-snap, or when the tracker loses the box (failure, or the box left the image).
3. ``detected``: no chain: the best EgoHand proposal of that side with score >= ``DETECT_SCORE``. This is how a
   box "propagates across views": by EgoHand's per-view side (no 3D fit runs on aarch64).
4. ``none``: no hand.

A ``no_hand`` label ends the chain at its time; a ``rejected`` label hides only that time (the chain passes through).

**Negative tracks.** A ``no_hand`` label at (t, cam, side) over a box the machine would have shown there starts a negative
track on that box, so the false box does not come back at the next grid time. That box is the continuation of the side's
live chain (the proposal of any side with the highest IoU >= ``CHAIN_IOU`` to the chain's box, or the chain's own box if
none), else the best unclaimed proposal of that side with score >= ``DETECT_SCORE``. No negative track starts on a box
that overlaps (IoU >= ``CLAIM_IOU``) a stored box of the other side in that camera at t: that hand is real, the reviewer
only swapped its side. At later grid times a negative track first claims every unclaimed proposal with IoU >=
``CHAIN_IOU`` to its box (any side), follows the best of them, holds ``MAX_HELD`` missed steps, then ends; a stored label
with a box for its (cam, side) ends it.

One proposal feeds at most one track per frame: reviewed boxes claim the proposals they overlap (IoU >= ``CLAIM_IOU``),
then negative tracks claim theirs, then chains take proposals greedily by IoU, then carried boxes re-snap (left before
right), then detections take what is left.

``propagate`` is a pure, deterministic function of (proposals, stored labels) and, with a carrier, of the decoded
frames (the carrier must return the same box for the same run). ``TrackCache`` keeps the computed
prefix of one segment and recomputes from the earliest time a label write touched.

A ``TrackCache`` runs any ``Propagator``: ``ChainPropagator`` (this module's rules, the default) or the 3D one of
``posekit.labeler.megatrack`` (``--propagator megatrack``).
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

from posekit.labeler.proposals import ProposalProvider, iou
from posekit.labeler.schema import HAND_SIDES, Box, HandSide, LabelSource, Proposal, TrackBox, TrackFrame, ViewLabel
from posekit.labeler.store import LabelStore

CHAIN_IOU: float = 0.3
"""IoU a proposal needs with the previous box to continue a chain."""
DETECT_SCORE: float = 0.5
"""EgoHand score from which a proposal starts a track when there is no chain."""
CLAIM_IOU: float = 0.5
"""IoU with a reviewed box from which a proposal belongs to that box (no other track may take it)."""
MAX_HELD: int = 1
"""Grid steps a chain may hold its box without a matching proposal."""
CARRY_STEPS: int = 10
"""Grid steps (2 s at 5 fps) a visual tracker may carry a reviewed box without an EgoHand re-snap."""
MACHINE_SOURCES: frozenset[LabelSource] = frozenset({"tracker", "range"})
"""Label sources that continue a reviewed chain but never start one (the machine made the box)."""
STEPS_PER_LOCK: int = 50
"""Grid times ``TrackCache`` computes per hold of its lock."""

TrackKey: TypeAlias = tuple[int, HandSide]
"""(cam, side)."""


@runtime_checkable
class BoxCarrier(Protocol):
    """A visual single-object tracker over one segment's decoded frames (``posekit.labeler.carry.VisualCarrier``)."""

    def carry(self, cam: int, start_ns: int, start_box: Box, t_ns: int) -> Box | None:
        """The box at grid time ``t_ns`` of the run initialised on camera ``cam``'s frame at ``start_ns`` with ``start_box``.

        Deterministic: the same run always gives the same boxes. None when the tracker lost the box at or before ``t_ns``.
        """
        ...


@dataclass(frozen=True, slots=True)
class Carry:
    """A visual tracker run that carries a reviewed chain's box."""

    start_ns: int
    """Grid time of the frame the tracker was initialised on."""
    start_box: Box
    """Box it was initialised with."""
    steps: int
    """Grid steps carried so far without a re-snap."""


@dataclass(frozen=True, slots=True)
class Chain:
    """The state a track carries to the next grid time."""

    box: Box
    """Last box of the track (a held step keeps it)."""
    held: int
    """Consecutive held steps so far."""
    seen_ns: int
    """Grid time ``box`` was seen at (a held step keeps it)."""
    reviewed: bool = False
    """The chain started from a reviewed box, so a visual tracker may carry it."""
    carry: Carry | None = None
    """The tracker run carrying the box since the last proposal or label; None when the last box was one."""


@dataclass(frozen=True, slots=True)
class Negative:
    """A negative track: a box the reviewer said is not this side's hand, followed so it is not shown again."""

    box: Box
    """Last box of the negative track."""
    held: int
    """Consecutive steps without an overlapping proposal."""


@runtime_checkable
class MachineStep(Protocol):
    """One grid time of a propagator: the machine boxes and whatever state the next time needs."""

    @property
    def frame(self) -> TrackFrame:
        """Boxes of every (cam, side)."""
        ...


@runtime_checkable
class Propagator(Protocol):
    """Machine propagation over the grid, one time after the other (``TrackCache`` keeps the steps)."""

    def step(self, t_ns: int, cams: list[int], proposals: list[Proposal], labels: list[ViewLabel], previous: MachineStep | None) -> MachineStep:
        """The machine boxes at ``t_ns`` given the step before it (None at the first grid time).

        Deterministic: the same inputs and previous step give the same boxes.
        """
        ...


@dataclass(frozen=True, slots=True)
class TrackStep:
    """The machine boxes of one grid time and the chains they leave for the next."""

    frame: TrackFrame
    """Boxes of every (cam, side)."""
    chains: dict[TrackKey, Chain]
    """Live chains after this time."""
    negatives: dict[TrackKey, Negative] = field(default_factory=dict)
    """Live negative tracks after this time."""


def track_step(
    t_ns: int,
    cams: list[int],
    proposals: list[Proposal],
    labels: list[ViewLabel],
    chains: dict[TrackKey, Chain],
    carrier: BoxCarrier | None = None,
    negatives: dict[TrackKey, Negative] | None = None,
) -> TrackStep:
    """Propagate the tracks one grid time; see the module docstring.

    Args:
        t_ns: Grid time, ns.
        cams: Camera indices, rig order.
        proposals: Every proposal at ``t_ns`` (only EgoHand ones are used).
        labels: Stored labels at ``t_ns``.
        chains: Live chains after the previous grid time.
        carrier: Visual tracker for reviewed chains; None = chains only hold (``MAX_HELD``).
        negatives: Live negative tracks after the previous grid time.
    """
    stored: dict[TrackKey, ViewLabel] = {(label.cam, label.side): label for label in labels}
    live_negatives: dict[TrackKey, Negative] = negatives or {}
    boxes: list[TrackBox] = []
    next_chains: dict[TrackKey, Chain] = {}
    next_negatives: dict[TrackKey, Negative] = {}
    for cam in cams:
        candidates: list[Proposal] = [p for p in proposals if p.model == "egohand" and p.cam == cam]
        claimed: set[int] = set()
        decided: dict[HandSide, TrackBox] = {}
        for side in HAND_SIDES:
            label: ViewLabel | None = stored.get((cam, side))
            if label is None:
                continue
            decided[side] = TrackBox(cam, side, label.box, "derived" if label.source == "rig" else "reviewed", None)
            if label.box is not None:
                prior: Chain | None = chains.get((cam, side))
                continues: bool = prior is not None and prior.reviewed and iou(prior.box, label.box) >= CHAIN_IOU
                next_chains[(cam, side)] = Chain(label.box, 0, t_ns, label.source not in MACHINE_SOURCES or continues)
                claimed.update(k for k, p in enumerate(candidates) if iou(p.box, label.box) >= CLAIM_IOU)
            elif label.status == "rejected" and (cam, side) in chains:
                next_chains[(cam, side)] = chains[(cam, side)]
        for side in HAND_SIDES:
            negative: Negative | None = live_negatives.get((cam, side))
            label = stored.get((cam, side))
            if negative is None or (label is not None and label.box is not None):
                continue
            overlapping: list[int] = [k for k, p in enumerate(candidates) if k not in claimed and iou(negative.box, p.box) >= CHAIN_IOU]
            claimed.update(overlapping)
            if overlapping:
                k = max(overlapping, key=lambda k: (iou(negative.box, candidates[k].box), -k))
                next_negatives[(cam, side)] = Negative(candidates[k].box, 0)
            elif negative.held < MAX_HELD:
                next_negatives[(cam, side)] = Negative(negative.box, negative.held + 1)
        for side in HAND_SIDES:
            label = stored.get((cam, side))
            if label is None or label.status != "no_hand" or (cam, side) in next_negatives:
                continue
            shown: Box | None = None
            prior_chain: Chain | None = chains.get((cam, side))
            if prior_chain is not None:
                continuing: list[int] = [k for k, p in enumerate(candidates) if k not in claimed and iou(prior_chain.box, p.box) >= CHAIN_IOU]
                shown = candidates[max(continuing, key=lambda k: (iou(prior_chain.box, candidates[k].box), -k))].box if continuing else prior_chain.box
            else:
                confident: list[int] = [k for k, p in enumerate(candidates) if k not in claimed and p.side == side and p.score >= DETECT_SCORE]
                shown = candidates[max(confident, key=lambda k: (candidates[k].score, -k))].box if confident else None
            other: ViewLabel | None = stored.get((cam, "right" if side == "left" else "left"))
            if shown is None or (other is not None and other.box is not None and iou(other.box, shown) >= CLAIM_IOU):
                continue
            next_negatives[(cam, side)] = Negative(shown, 0)
            claimed.update(k for k, p in enumerate(candidates) if iou(shown, p.box) >= CHAIN_IOU)
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
            next_chains[(cam, side)] = Chain(candidates[k].box, 0, t_ns, chains[(cam, side)].reviewed)
        for side in chained_sides:
            chain: Chain = chains[(cam, side)]
            if side in decided:
                continue
            if carrier is not None and chain.reviewed:
                run: Carry = chain.carry or Carry(chain.seen_ns, chain.box, 0)
                carried: Box | None = carrier.carry(cam, run.start_ns, run.start_box, t_ns) if run.steps < CARRY_STEPS else None
                if carried is None:
                    continue
                snaps: list[int] = [k for k, p in enumerate(candidates) if k not in claimed and iou(carried, p.box) >= CHAIN_IOU]
                if snaps:
                    k = max(snaps, key=lambda k: (iou(carried, candidates[k].box), -k))
                    claimed.add(k)
                    decided[side] = TrackBox(cam, side, candidates[k].box, "chained", candidates[k].score)
                    next_chains[(cam, side)] = Chain(candidates[k].box, 0, t_ns, True)
                else:
                    decided[side] = TrackBox(cam, side, carried, "tracked", None)
                    next_chains[(cam, side)] = Chain(carried, 0, t_ns, True, Carry(run.start_ns, run.start_box, run.steps + 1))
            elif chain.held < MAX_HELD:
                decided[side] = TrackBox(cam, side, chain.box, "held", None)
                next_chains[(cam, side)] = Chain(chain.box, chain.held + 1, chain.seen_ns, chain.reviewed)
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
            next_chains[(cam, side)] = Chain(candidates[best].box, 0, t_ns)
        boxes.extend(decided[side] for side in HAND_SIDES)
    return TrackStep(TrackFrame(t_ns, boxes), next_chains, next_negatives)


def propagate(
    times_ns: list[int],
    cams: list[int],
    proposals_at: Callable[[int], list[Proposal]],
    labels_at: Callable[[int], list[ViewLabel]],
    carrier: BoxCarrier | None = None,
) -> list[TrackFrame]:
    """The machine track of every grid time, from the first one.

    Args:
        times_ns: The grid, ascending, ns.
        cams: Camera indices, rig order.
        proposals_at: Grid time -> every proposal at it.
        labels_at: Grid time -> stored labels at it.
        carrier: Visual tracker for reviewed chains; None = chains only hold.
    """
    frames: list[TrackFrame] = []
    chains: dict[TrackKey, Chain] = {}
    negatives: dict[TrackKey, Negative] = {}
    for t_ns in times_ns:
        step: TrackStep = track_step(t_ns, cams, proposals_at(t_ns), labels_at(t_ns), chains, carrier, negatives)
        frames.append(step.frame)
        chains = step.chains
        negatives = step.negatives
    return frames


@dataclass(frozen=True, slots=True)
class ChainPropagator:
    """This module's rules (EgoHand chains, optionally carried by a visual tracker) as a ``Propagator``."""

    carrier: BoxCarrier | None = None
    """Visual tracker for reviewed chains; None = chains only hold."""

    def step(self, t_ns: int, cams: list[int], proposals: list[Proposal], labels: list[ViewLabel], previous: MachineStep | None) -> TrackStep:
        """``track_step`` from the previous step's chains."""
        if previous is not None and not isinstance(previous, TrackStep):
            raise TypeError(f"ChainPropagator continues a TrackStep, not {type(previous).__name__}")
        if previous is None:
            return track_step(t_ns, cams, proposals, labels, {}, self.carrier)
        return track_step(t_ns, cams, proposals, labels, previous.chains, self.carrier, previous.negatives)


class TrackCache:
    """The machine track of one segment, computed lazily from the first grid time and kept until a label write."""

    def __init__(
        self,
        segment_id: str,
        *,
        times_ns: list[int],
        cams: list[int],
        proposals: ProposalProvider,
        store: LabelStore,
        carrier: BoxCarrier | None = None,
        propagator: Propagator | None = None,
    ) -> None:
        """Bind the cache to a segment's grid, proposals and labels.

        Args:
            segment_id: Catalog segment id.
            times_ns: The 5 fps grid, ascending.
            cams: Camera indices, rig order.
            proposals: Proposal source.
            store: The segment's labels.
            carrier: Visual tracker for reviewed chains (it keeps its own runs; a label write does not drop them).
            propagator: The propagation rules; None = ``ChainPropagator(carrier)``.
        """
        self.segment_id: str = segment_id
        self.times_ns: list[int] = sorted(times_ns)
        self.cams: list[int] = cams
        self.proposals: ProposalProvider = proposals
        self.store: LabelStore = store
        self.propagator: Propagator = ChainPropagator(carrier) if propagator is None else propagator
        self._index: dict[int, int] = {t: i for i, t in enumerate(self.times_ns)}
        self._steps: list[MachineStep] = []
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
                    previous: MachineStep | None = self._steps[-1] if self._steps else None
                    self._steps.append(
                        self.propagator.step(t_ns, self.cams, self.proposals.proposals(self.segment_id, t_ns), self.store.get(t_ns), previous)
                    )

    def frame(self, t_ns: int) -> TrackFrame:
        """The machine track at one grid time."""
        return self.frames(t_ns, 1)[0]

    def invalidate_from(self, t_ns: int) -> None:
        """Forget every computed time at or after ``t_ns`` (call after the label write at ``t_ns`` has landed)."""
        with self._lock:
            del self._steps[bisect_left(self.times_ns, t_ns) :]
