"""What a reviewer's post did, measured against the machine track, and the effort summary of a segment.

Each posted label is compared with the stored label at its (t, cam, side) when there is one, else with the machine
box there (``posekit.labeler.tracker``), BEFORE the post lands. A label that differs gives one ``ActionEvent``:

- a box where the machine had none, or IoU < ``MOVE_IOU`` with the machine box: ``draw_box``, or ``pick_proposal``
  when the box equals an EgoHand proposal of that camera within ``SAME_PX``;
- the machine box of the OTHER side of that camera (within ``SAME_PX``): ``set_side`` (the companion ``no_hand``
  posted for that other side is part of the same action and is not logged again);
- IoU >= ``MOVE_IOU`` with the machine box but not equal: ``move_box``;
- ``no_hand``: ``no_hand``; ``rejected``: ``reject``.

When no posted label differs and at least one of them confirms a machine box that was not already a stored label,
the post logs one ``accept_frame``.

Effort (``effort_summary``): ``frames`` are grid times with at least one stored label and ``video_s`` is
``frames`` x the grid period; interventions are actions other than ``accept_range``/``accept_frame``;
``reviewer_wall_s`` sums the wall-clock gaps between consecutive actions, each capped at ``IDLE_CAP_S`` (so a
pause does not count as work; the time before the first action is not measured).
"""

from __future__ import annotations

from itertools import pairwise

from posekit.labeler.proposals import iou
from posekit.labeler.schema import (
    PASSIVE_ACTIONS,
    ActionEvent,
    ActionKind,
    Box,
    EffortSummary,
    FrameLabels,
    HandSide,
    Proposal,
    TrackBox,
    TrackFrame,
    ViewLabel,
)
from posekit.labeler.store import LabelRow

SAME_PX: float = 1.0
"""Largest corner difference, px, at which two boxes are the same box."""
MOVE_IOU: float = 0.5
"""IoU with the machine box from which a changed box is a move, not a new box."""
IDLE_CAP_S: float = 300.0
"""Longest gap between two actions that counts as reviewer work, s."""


def same_box(a: Box | None, b: Box | None) -> bool:
    """Both boxes exist and every corner coordinate differs by at most ``SAME_PX``."""
    if a is None or b is None:
        return False
    return max(abs(a.x0 - b.x0), abs(a.y0 - b.y0), abs(a.x1 - b.x1), abs(a.y1 - b.y1)) <= SAME_PX


def same_label(a: ViewLabel, b: ViewLabel) -> bool:
    """Same decision: same status family (box / no hand / rejected) and the same box."""
    if a.box is not None or b.box is not None:
        return same_box(a.box, b.box)
    return a.status == b.status


def classify_labels(frame: FrameLabels, machine: TrackFrame, proposals: list[Proposal], stored: list[ViewLabel], wall_ns: int) -> list[ActionEvent]:
    """The actions of one post; see the module docstring.

    Args:
        frame: The post.
        machine: The machine track at ``frame.t_ns`` before the post.
        proposals: Every proposal at ``frame.t_ns``.
        stored: The stored labels at ``frame.t_ns`` before the post.
        wall_ns: Wall-clock time of the post, ns.
    """
    track: dict[tuple[int, HandSide], TrackBox] = {(box.cam, box.side): box for box in machine.boxes}
    before: dict[tuple[int, HandSide], ViewLabel] = {(label.cam, label.side): label for label in stored}
    kinds: dict[tuple[int, HandSide], ActionKind] = {}
    confirms_machine: bool = False
    for label in frame.labels:
        key: tuple[int, HandSide] = (label.cam, label.side)
        old: ViewLabel | None = before.get(key)
        machine_box: Box | None = track[key].box if key in track else None
        if old is not None:
            if same_label(label, old):
                continue
        elif (label.box is None and label.status == "no_hand" and machine_box is None) or same_box(label.box, machine_box):
            confirms_machine = True
            continue
        other: HandSide = "right" if label.side == "left" else "left"
        other_box: Box | None = track[(label.cam, other)].box if (label.cam, other) in track else None
        if label.status == "rejected":
            kinds[key] = "reject"
        elif label.box is None:
            kinds[key] = "no_hand"
        elif same_box(label.box, other_box):
            kinds[key] = "set_side"
        elif machine_box is not None and iou(label.box, machine_box) >= MOVE_IOU:
            kinds[key] = "move_box"
        elif any(p.model == "egohand" and p.cam == label.cam and same_box(p.box, label.box) for p in proposals):
            kinds[key] = "pick_proposal"
        else:
            kinds[key] = "draw_box"
    by_key: dict[tuple[int, HandSide], ViewLabel] = {(label.cam, label.side): label for label in frame.labels}
    for (cam, side), kind in list(kinds.items()):
        other_side: HandSide = "right" if side == "left" else "left"
        flipped: ViewLabel | None = by_key.get((cam, other_side))
        was: TrackBox | None = track.get((cam, side))
        if (
            kind == "no_hand"
            and kinds.get((cam, other_side)) == "set_side"
            and flipped is not None
            and was is not None
            and same_box(flipped.box, was.box)
        ):
            del kinds[(cam, side)]
    if not kinds:
        if not confirms_machine:
            return []
        return [ActionEvent(wall_ns, frame.segment_id, frame.t_ns, None, None, None, "accept_frame", frame.labels[0].reviewer, len(frame.labels))]
    return [
        ActionEvent(wall_ns, frame.segment_id, frame.t_ns, None, cam, side, kind, by_key[(cam, side)].reviewer, 1)
        for (cam, side), kind in sorted(kinds.items())
    ]


def effort_summary(actions: list[ActionEvent], rows: list[LabelRow], period_ns: int) -> EffortSummary:
    """Effort of one segment from its action log and stored labels; see the module docstring.

    Args:
        actions: Every logged action.
        rows: Every stored label.
        period_ns: Grid period, ns.
    """
    frames: int = len({row.t_ns for row in rows})
    video_s: float = frames * period_ns / 1e9
    interventions: list[ActionEvent] = [action for action in actions if action.kind not in PASSIVE_ACTIONS]
    walls: list[int] = sorted(action.wall_ns for action in actions)
    reviewer_wall_s: float = sum((min((b - a) / 1e9, IDLE_CAP_S) for a, b in pairwise(walls)), 0.0)
    frames_corrected: int = len({action.t_ns for action in interventions})
    n_boxes: int = sum(1 for row in rows if row.label.box is not None)
    return EffortSummary(
        video_s=video_s,
        n_actions=len(actions),
        n_interventions=len(interventions),
        interventions_per_min=len(interventions) / (video_s / 60.0) if video_s > 0.0 else None,
        frames=frames,
        frames_corrected=frames_corrected,
        share_corrected=frames_corrected / frames if frames else None,
        boxes_accepted_by_watching=sum(1 for row in rows if row.label.source == "tracker" and row.label.box is not None),
        reviewer_wall_s=reviewer_wall_s,
        boxes_per_reviewer_s=n_boxes / reviewer_wall_s if reviewer_wall_s > 0.0 else None,
    )
