"""Classifying reviewer posts against the machine track, and the effort summary (posekit.labeler.actions)."""

from __future__ import annotations

import pytest

from posekit.labeler.actions import classify_labels, effort_summary
from posekit.labeler.schema import (
    ActionEvent,
    ActionKind,
    Box,
    EffortSummary,
    FrameLabels,
    HandSide,
    LabelStatus,
    Proposal,
    TrackBox,
    TrackFrame,
    ViewLabel,
)
from posekit.labeler.store import LabelRow

T: int = 1_000_000_000
WALL: int = 1_700_000_000_000_000_000
LEFT_BOX: Box = Box(100.0, 100.0, 200.0, 200.0)
RIGHT_BOX: Box = Box(400.0, 100.0, 500.0, 200.0)


def machine(*overrides: TrackBox) -> TrackFrame:
    """Cam 0 has a chained left box and a detected right box; cam 1 has no hand; ``overrides`` replace entries."""
    boxes: dict[tuple[int, HandSide], TrackBox] = {
        (0, "left"): TrackBox(0, "left", LEFT_BOX, "chained", 0.8),
        (0, "right"): TrackBox(0, "right", RIGHT_BOX, "detected", 0.9),
        (1, "left"): TrackBox(1, "left", None, "none", None),
        (1, "right"): TrackBox(1, "right", None, "none", None),
    }
    for box in overrides:
        boxes[(box.cam, box.side)] = box
    return TrackFrame(T, list(boxes.values()))


PROPOSALS: list[Proposal] = [Proposal(1, "right", Box(700.0, 300.0, 800.0, 420.0), 0.4, "egohand")]


def kinds(
    labels: list[ViewLabel], track: TrackFrame | None = None, stored: list[ViewLabel] | None = None
) -> list[tuple[ActionKind, int | None, HandSide | None, int]]:
    events: list[ActionEvent] = classify_labels(FrameLabels("seg", T, labels), track or machine(), PROPOSALS, stored or [], WALL)
    assert all((event.wall_ns, event.segment_id, event.t_ns, event.t1_ns) == (WALL, "seg", T, None) for event in events)
    return [(event.kind, event.cam, event.side, event.n_labels) for event in events]


def label(cam: int, side: HandSide, box: Box | None, status: LabelStatus = "corrected") -> ViewLabel:
    """An agent label; without a box, ``corrected`` means ``no_hand``."""
    return ViewLabel(cam, side, "no_hand" if box is None and status == "corrected" else status, box, "agent", "astra")


def test_box_where_the_machine_had_none_is_drawn_or_picked() -> None:
    assert kinds([label(1, "left", Box(10.0, 10.0, 90.0, 90.0))]) == [("draw_box", 1, "left", 1)]
    picked: Box = Box(700.4, 299.6, 800.0, 420.9)  # an EgoHand proposal within 1 px (of either side)
    assert kinds([label(1, "left", picked)]) == [("pick_proposal", 1, "left", 1)]


def test_box_far_from_the_machine_box_is_drawn() -> None:
    assert kinds([label(0, "left", Box(150.0, 150.0, 260.0, 260.0))]) == [("draw_box", 0, "left", 1)]  # IoU 0.13


def test_overlapping_change_is_a_move() -> None:
    assert kinds([label(0, "left", Box(110.0, 100.0, 210.0, 200.0))]) == [("move_box", 0, "left", 1)]


def test_equal_box_is_nothing_and_a_whole_equal_post_is_one_accept_frame() -> None:
    nudged: Box = Box(100.5, 99.5, 200.0, 200.0)
    assert kinds([label(0, "left", nudged, "accepted"), label(0, "right", RIGHT_BOX, "accepted")]) == [("accept_frame", None, None, 2)]
    assert kinds([label(0, "left", nudged), label(1, "left", Box(10.0, 10.0, 90.0, 90.0))]) == [("draw_box", 1, "left", 1)]


def test_the_other_sides_box_is_a_side_change() -> None:
    # The machine's right box is really the left hand: one set_side; the companion no_hand on the right is part of it.
    assert kinds([label(0, "left", RIGHT_BOX), label(0, "right", None)]) == [("set_side", 0, "left", 1)]


def test_no_hand_and_reject() -> None:
    assert kinds([label(0, "left", None)]) == [("no_hand", 0, "left", 1)]
    assert kinds([label(1, "left", None)]) == [("accept_frame", None, None, 1)]
    assert kinds([label(1, "left", None, "rejected")]) == [("reject", 1, "left", 1)]


def test_reposting_stored_labels_logs_nothing() -> None:
    stored: list[ViewLabel] = [label(0, "left", LEFT_BOX), label(1, "left", None, "rejected")]
    track: TrackFrame = machine(TrackBox(0, "left", LEFT_BOX, "reviewed", None), TrackBox(1, "left", None, "reviewed", None))
    assert kinds(stored, track, stored) == []
    # no_hand over a stored rejected label is a change, though the machine shows no box either way.
    assert kinds([label(1, "left", None)], track, stored) == [("no_hand", 1, "left", 1)]


def test_effort_summary() -> None:
    def event(kind: ActionKind, t_ns: int, wall_s: float, n_labels: int = 1) -> ActionEvent:
        per_view: bool = kind not in ("accept_range", "accept_frame")
        return ActionEvent(
            WALL + int(wall_s * 1e9), "seg", t_ns, None, 0 if per_view else None, "left" if per_view else None, kind, "astra", n_labels
        )

    actions: list[ActionEvent] = [
        event("accept_range", T, 0.0, 22),
        event("draw_box", T + 400_000_000, 10.0),
        event("move_box", T + 400_000_000, 12.0),
        event("no_hand", T + 600_000_000, 2_000.0),  # a long idle gap counts as IDLE_CAP_S
    ]
    rows: list[LabelRow] = [LabelRow(T + i * 200_000_000, label(0, "left", LEFT_BOX, "accepted")) for i in range(10)]
    rows = [LabelRow(row.t_ns, ViewLabel(0, "left", "accepted", LEFT_BOX, "tracker", "astra")) for row in rows[:8]] + rows[8:]
    rows.append(LabelRow(T, ViewLabel(1, "left", "no_hand", None, "tracker", "astra")))
    summary: EffortSummary = effort_summary(actions, rows, 200_000_000)
    assert (summary.frames, summary.video_s, summary.n_actions, summary.n_interventions) == (10, pytest.approx(2.0), 4, 3)
    assert summary.interventions_per_min == pytest.approx(90.0)
    assert (summary.frames_corrected, summary.share_corrected) == (2, pytest.approx(0.2))
    assert summary.boxes_accepted_by_watching == 8
    assert summary.reviewer_wall_s == pytest.approx(312.0)
    assert summary.boxes_per_reviewer_s == pytest.approx(10 / 312.0)


def test_effort_summary_of_nothing() -> None:
    summary: EffortSummary = effort_summary([], [], 200_000_000)
    assert (summary.frames, summary.interventions_per_min, summary.share_corrected, summary.boxes_per_reviewer_s) == (0, None, None, None)
