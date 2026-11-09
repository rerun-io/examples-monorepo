"""Which frames a reviewer sees next, the range fill between reviewed keyframes, and the per-frame flags.

Review order: 1 fps keyframes that are not fully reviewed (every (cam, side) has a label) come first, reason
``keyframe``; then 5 fps times that a range fill could not decide, reason ``range_gap``. Each queue item also
carries the frame's flags.

Range fill (MEgATrack "accept the tracked run"), for each 5 fps time strictly between two reviewed keyframes
and each (cam, side) without a reviewed label:

- both ends ``no_hand`` and no confident proposal (score >= 0.5, any model) of that side -> ``no_hand``;
- both ends have a box and the EgoHand proposal of that side overlaps the linear interpolation of the two
  end boxes with IoU >= 0.3 -> ``accepted`` with the proposal box;
- otherwise the label stays empty and the time is queued as ``range_gap``.

Range labels carry source ``range``; a later fill may overwrite them, never a reviewed label.

Flags (shown in ``FrameState`` and on queue items):

- ``side_flip``: in one camera, a box of one side at t overlaps (IoU >= 0.5) a box of the other side at the
  previous grid time more than any box of its own side there (boxes = stored labels where present, else confident EgoHand proposals);
- ``views_disagree``: EgoHand and the ``hands_layer`` both have a box of one side in one camera, one of them
  confident, and their IoU < 0.3;
- ``low_score``: an EgoHand proposal with 0.3 <= score < 0.5.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from posekit.labeler.schema import HAND_SIDES, Box, FrameLabels, HandSide, Proposal, QueueItem, RangeRequest, RangeResult, ViewLabel
from posekit.labeler.store import LabelStore

CONFIDENT: float = 0.5
"""Proposal score at or above which a box counts as a confident detection."""
LOW_SCORE: float = 0.3
"""Lower end of the ``low_score`` band [LOW_SCORE, CONFIDENT)."""
RANGE_IOU: float = 0.3
"""IoU an EgoHand proposal needs with the interpolated end boxes to be range-accepted."""
DISAGREE_IOU: float = 0.3
"""IoU under which EgoHand and the hands layer disagree."""
FLIP_IOU: float = 0.5
"""IoU over which two boxes of consecutive times are the same hand."""


@runtime_checkable
class ProposalProvider(Protocol):
    """Model proposals for one time of one segment (EgoHand and, on Cap B segments, the ``hands`` layer)."""

    def proposals(self, segment_id: str, t_ns: int) -> list[Proposal]:
        """Every proposal of every camera at ``t_ns``."""
        ...


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    width: float = min(a.x1, b.x1) - max(a.x0, b.x0)
    height: float = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    inter: float = width * height
    union: float = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union


def lerp_box(a: Box, b: Box, alpha: float) -> Box:
    """Linear interpolation of two boxes, ``alpha`` in [0, 1] from ``a`` to ``b``."""
    return Box(
        a.x0 + (b.x0 - a.x0) * alpha,
        a.y0 + (b.y0 - a.y0) * alpha,
        a.x1 + (b.x1 - a.x1) * alpha,
        a.y1 + (b.y1 - a.y1) * alpha,
    )


def is_reviewed(store: LabelStore, t_ns: int, cams: list[int]) -> bool:
    """Every (cam, side) at ``t_ns`` has a label."""
    have: set[tuple[int, HandSide]] = {(label.cam, label.side) for label in store.get(t_ns)}
    return all((cam, side) in have for cam in cams for side in HAND_SIDES)


def frame_flags(proposals: list[Proposal], previous_boxes: dict[tuple[int, HandSide], Box], current_boxes: dict[tuple[int, HandSide], Box]) -> list[str]:
    """Flags of one frame set; see the module docstring.

    Args:
        proposals: Every proposal at this time.
        previous_boxes: Effective box per (cam, side) at the previous grid time.
        current_boxes: Effective box per (cam, side) at this time.
    """
    flags: list[str] = []
    for (cam, side), box in current_boxes.items():
        other: HandSide = "right" if side == "left" else "left"
        before_other: Box | None = previous_boxes.get((cam, other))
        before_same: Box | None = previous_boxes.get((cam, side))
        overlap_other: float = 0.0 if before_other is None else iou(box, before_other)
        overlap_same: float = 0.0 if before_same is None else iou(box, before_same)
        if overlap_other >= FLIP_IOU and overlap_other > overlap_same:
            flags.append("side_flip")
            break
    egohand: list[Proposal] = [p for p in proposals if p.model == "egohand"]
    hands_layer: list[Proposal] = [p for p in proposals if p.model == "hands_layer"]
    for mine in egohand:
        for theirs in hands_layer:
            if mine.cam == theirs.cam and mine.side == theirs.side and max(mine.score, theirs.score) >= CONFIDENT and iou(mine.box, theirs.box) < DISAGREE_IOU:
                flags.append("views_disagree")
                break
        if "views_disagree" in flags:
            break
    if any(LOW_SCORE <= p.score < CONFIDENT for p in egohand):
        flags.append("low_score")
    return flags


def effective_boxes(labels: list[ViewLabel], proposals: list[Proposal]) -> dict[tuple[int, HandSide], Box]:
    """Box per (cam, side): the stored label where one exists (none for no_hand/rejected), else the best confident EgoHand proposal."""
    boxes: dict[tuple[int, HandSide], Box] = {}
    labelled: set[tuple[int, HandSide]] = set()
    for label in labels:
        labelled.add((label.cam, label.side))
        if label.box is not None:
            boxes[(label.cam, label.side)] = label.box
    for proposal in sorted(proposals, key=lambda p: p.score):
        key: tuple[int, HandSide] = (proposal.cam, proposal.side)
        if proposal.model == "egohand" and proposal.score >= CONFIDENT and key not in labelled:
            boxes[key] = proposal.box
    return boxes


class ReviewQueue:
    """Review queue and range fill of one segment."""

    def __init__(self, segment_id: str, *, times_ns: list[int], keyframes_ns: list[int], cams: list[int], store: LabelStore, proposals: ProposalProvider) -> None:
        """Bind the queue to a segment's grid, its store and the proposals.

        Args:
            segment_id: Catalog segment id.
            times_ns: The 5 fps grid, ascending.
            keyframes_ns: The 1 fps keyframes, a subset of ``times_ns``.
            cams: Camera indices in rig order.
            store: The segment's labels.
            proposals: Proposal source.
        """
        self.segment_id: str = segment_id
        self.times_ns: list[int] = sorted(times_ns)
        self.keyframes_ns: list[int] = sorted(keyframes_ns)
        self.cams: list[int] = cams
        self.store: LabelStore = store
        self.proposals: ProposalProvider = proposals
        self._grid_index: dict[int, int] = {t: i for i, t in enumerate(self.times_ns)}
        self._gaps: set[int] = set()
        self._lock = threading.Lock()

    def on_grid(self, t_ns: int) -> bool:
        """``t_ns`` is a time of the 5 fps grid."""
        return t_ns in self._grid_index

    def flags(self, t_ns: int) -> list[str]:
        """Flags of the frame set at ``t_ns``."""
        proposals: list[Proposal] = self.proposals.proposals(self.segment_id, t_ns)
        current: dict[tuple[int, HandSide], Box] = effective_boxes(self.store.get(t_ns), proposals)
        index: int | None = self._grid_index.get(t_ns)
        previous: dict[tuple[int, HandSide], Box] = {}
        if index is not None and index > 0:
            before_ns: int = self.times_ns[index - 1]
            previous = effective_boxes(self.store.get(before_ns), self.proposals.proposals(self.segment_id, before_ns))
        return frame_flags(proposals, previous, current)

    def items(self, limit: int) -> list[QueueItem]:
        """Up to ``limit`` frame sets to review: unreviewed keyframes first, then range gaps, both in time order."""
        found: list[QueueItem] = []
        for t_ns in self.keyframes_ns:
            if len(found) >= limit:
                return found
            if not is_reviewed(self.store, t_ns, self.cams):
                found.append(QueueItem(self.segment_id, t_ns, ["keyframe", *self.flags(t_ns)]))
        with self._lock:
            open_gaps: list[int] = sorted(t for t in self._gaps if not is_reviewed(self.store, t, self.cams))
            self._gaps = set(open_gaps)
        for t_ns in open_gaps:
            if len(found) >= limit:
                break
            found.append(QueueItem(self.segment_id, t_ns, ["range_gap", *self.flags(t_ns)]))
        return found

    def n_reviewed_keyframes(self) -> int:
        """Keyframes with a label for every (cam, side)."""
        return sum(is_reviewed(self.store, t_ns, self.cams) for t_ns in self.keyframes_ns)

    def fill_range(self, request: RangeRequest) -> RangeResult:
        """Fill the grid times strictly between two reviewed keyframes; see the module docstring.

        Raises:
            ValueError: The ends are not keyframes of this segment, are out of order, or are not fully reviewed.
        """
        if request.segment_id != self.segment_id:
            raise ValueError(f"range for segment {request.segment_id!r} sent to {self.segment_id!r}")
        keyframes: set[int] = set(self.keyframes_ns)
        if request.t0_ns not in keyframes or request.t1_ns not in keyframes:
            raise ValueError("t0_ns and t1_ns must both be keyframes of the segment")
        if request.t0_ns >= request.t1_ns:
            raise ValueError("t0_ns must be before t1_ns")
        for end in (request.t0_ns, request.t1_ns):
            if not is_reviewed(self.store, end, self.cams):
                raise ValueError(f"keyframe {end} is not fully reviewed")
        filled: int = 0
        queued: list[int] = []
        span: float = float(request.t1_ns - request.t0_ns)
        for t_ns in (t for t in self.times_ns if request.t0_ns < t < request.t1_ns):
            alpha: float = (t_ns - request.t0_ns) / span
            proposals: list[Proposal] = self.proposals.proposals(self.segment_id, t_ns)
            new_labels: list[ViewLabel] = []
            open_views: int = 0
            for cam in self.cams:
                for side in HAND_SIDES:
                    existing: ViewLabel | None = self.store.get_view(t_ns, cam, side)
                    if existing is not None and existing.source != "range":
                        continue
                    decided: ViewLabel | None = self._range_label(request, cam, side, alpha, proposals)
                    if decided is None:
                        open_views += 1
                    else:
                        new_labels.append(decided)
            if new_labels:
                filled += self.store.put(FrameLabels(self.segment_id, t_ns, new_labels))
            if open_views:
                queued.append(t_ns)
        with self._lock:
            self._gaps.update(queued)
        return RangeResult(filled, queued)

    def _range_label(self, request: RangeRequest, cam: int, side: HandSide, alpha: float, proposals: list[Proposal]) -> ViewLabel | None:
        """The range decision for one (cam, side) at one time, or None when it needs a reviewer."""
        start: ViewLabel | None = self.store.get_view(request.t0_ns, cam, side)
        end: ViewLabel | None = self.store.get_view(request.t1_ns, cam, side)
        if start is None or end is None:
            return None
        same_side: list[Proposal] = [p for p in proposals if p.cam == cam and p.side == side]
        if start.status == "no_hand" and end.status == "no_hand":
            if any(p.score >= CONFIDENT for p in same_side):
                return None
            return ViewLabel(cam, side, "no_hand", None, "range", request.reviewer)
        if start.box is None or end.box is None:
            return None
        expected: Box = lerp_box(start.box, end.box, alpha)
        candidates: list[Proposal] = [p for p in same_side if p.model == "egohand"]
        if not candidates:
            return None
        best: Proposal = max(candidates, key=lambda p: iou(p.box, expected))
        if iou(best.box, expected) < RANGE_IOU:
            return None
        return ViewLabel(cam, side, "accepted", best.box, "range", request.reviewer)
