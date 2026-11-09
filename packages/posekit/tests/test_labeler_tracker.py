"""Machine propagation of hand boxes on synthetic proposals (posekit.labeler.tracker)."""

from __future__ import annotations

import pytest

from posekit.labeler.schema import Box, FrameLabels, HandSide, Proposal, TrackBox, TrackFrame, ViewLabel
from posekit.labeler.store import LabelStore
from posekit.labeler.tracker import TrackCache, propagate

STEP: int = 200_000_000
TIMES: list[int] = [i * STEP for i in range(8)]
CAMS: list[int] = [0, 1]


def ego(cam: int, side: HandSide, x: float, score: float = 0.9) -> Proposal:
    """A 100x100 EgoHand box whose left edge is at ``x``."""
    return Proposal(cam, side, Box(x, 100.0, x + 100.0, 200.0), score, "egohand")


def at(frames: list[TrackFrame], index: int, cam: int, side: HandSide) -> TrackBox:
    found: list[TrackBox] = [box for box in frames[index].boxes if (box.cam, box.side) == (cam, side)]
    assert len(found) == 1
    return found[0]


def run(proposals: dict[int, list[Proposal]], labels: dict[int, list[ViewLabel]] | None = None) -> list[TrackFrame]:
    """Propagate over ``TIMES``; both dicts are keyed by grid index."""
    stored: dict[int, list[ViewLabel]] = labels or {}
    return propagate(TIMES, CAMS, lambda t: proposals.get(TIMES.index(t), []), lambda t: stored.get(TIMES.index(t), []))


def reviewed(cam: int, side: HandSide, x: float) -> ViewLabel:
    return ViewLabel(cam, side, "corrected", Box(x, 100.0, x + 100.0, 200.0), "person", "tester")


def test_frames_list_every_cam_and_side_in_rig_order() -> None:
    frames: list[TrackFrame] = run({})
    assert [frame.t_ns for frame in frames] == TIMES
    assert [(box.cam, box.side, box.box, box.origin) for box in frames[0].boxes] == [
        (0, "left", None, "none"),
        (0, "right", None, "none"),
        (1, "left", None, "none"),
        (1, "right", None, "none"),
    ]


def test_detected_fallback_needs_a_confident_proposal_of_that_side() -> None:
    frames: list[TrackFrame] = run({0: [ego(0, "left", 10.0, 0.8), ego(0, "left", 500.0, 0.6), ego(1, "right", 10.0, 0.4)]})
    first: TrackBox = at(frames, 0, 0, "left")
    assert (first.origin, first.box, first.score) == ("detected", Box(10.0, 100.0, 110.0, 200.0), 0.8)
    assert at(frames, 0, 1, "right").origin == "none"  # score 0.4 < 0.5
    assert at(frames, 0, 0, "right").origin == "none"


def test_chain_follows_overlapping_proposals_and_keeps_its_side() -> None:
    # Reviewed left box at t0; EgoHand calls the same moving hand "right" at t1 and t2 with a low score.
    proposals: dict[int, list[Proposal]] = {1: [ego(0, "right", 30.0, 0.35)], 2: [ego(0, "right", 50.0, 0.35)]}
    frames: list[TrackFrame] = run(proposals, {0: [reviewed(0, "left", 10.0)]})
    assert at(frames, 0, 0, "left").origin == "reviewed"
    second: TrackBox = at(frames, 1, 0, "left")
    assert (second.origin, second.box, second.score) == ("chained", Box(30.0, 100.0, 130.0, 200.0), 0.35)
    assert at(frames, 2, 0, "left").box == Box(50.0, 100.0, 150.0, 200.0)
    # The proposal belongs to the left track; the right side gets no box from it.
    assert at(frames, 1, 0, "right").origin == "none"


def test_chain_ignores_a_proposal_below_the_chain_iou() -> None:
    # IoU of [10,110] and [90,190] is 20/180 < 0.3: not the same hand.
    frames: list[TrackFrame] = run({1: [ego(0, "left", 90.0, 0.4)]}, {0: [reviewed(0, "left", 10.0)]})
    assert at(frames, 1, 0, "left").origin == "held"


def test_one_miss_holds_two_misses_end_the_chain() -> None:
    proposals: dict[int, list[Proposal]] = {3: [ego(0, "left", 10.0, 0.4)], 5: [ego(0, "left", 10.0, 0.4)]}
    frames: list[TrackFrame] = run(proposals, {0: [reviewed(0, "left", 10.0)]})
    held: TrackBox = at(frames, 1, 0, "left")
    assert (held.origin, held.box, held.score) == ("held", Box(10.0, 100.0, 110.0, 200.0), None)
    assert at(frames, 2, 0, "left").origin == "none"  # second miss: the chain ends
    # Without a chain, a low-score proposal is not enough.
    assert at(frames, 3, 0, "left").origin == "none"


def test_a_held_chain_resumes_on_the_next_detection() -> None:
    frames: list[TrackFrame] = run({2: [ego(0, "left", 20.0, 0.4)]}, {0: [reviewed(0, "left", 10.0)]})
    assert [at(frames, i, 0, "left").origin for i in range(4)] == ["reviewed", "held", "chained", "held"]


def test_a_detected_box_starts_a_chain() -> None:
    frames: list[TrackFrame] = run({0: [ego(1, "right", 10.0, 0.9)], 1: [ego(1, "left", 15.0, 0.45)]})
    assert at(frames, 0, 1, "right").origin == "detected"
    nxt: TrackBox = at(frames, 1, 1, "right")
    assert (nxt.origin, nxt.box) == ("chained", Box(15.0, 100.0, 115.0, 200.0))
    assert at(frames, 1, 1, "left").origin == "none"


def test_reviewed_label_overrides_the_machine_and_restarts_the_chain() -> None:
    proposals: dict[int, list[Proposal]] = {i: [ego(0, "left", 10.0)] for i in range(len(TIMES))}
    proposals[3] = [ego(0, "left", 10.0), ego(0, "left", 400.0, 0.3)]
    frames: list[TrackFrame] = run(proposals, {2: [reviewed(0, "left", 395.0)]})
    assert at(frames, 1, 0, "left").origin == "chained"
    fixed: TrackBox = at(frames, 2, 0, "left")
    assert (fixed.origin, fixed.box, fixed.score) == ("reviewed", Box(395.0, 100.0, 495.0, 200.0), None)
    assert at(frames, 3, 0, "left").box == Box(400.0, 100.0, 500.0, 200.0)  # chained from the fix, not the confident box
    assert at(frames, 3, 0, "left").origin == "chained"


def test_no_hand_ends_a_chain_and_rejected_does_not() -> None:
    proposals: dict[int, list[Proposal]] = {i: [ego(0, "left", 10.0, 0.4), ego(1, "left", 10.0, 0.4)] for i in range(len(TIMES))}
    labels: dict[int, list[ViewLabel]] = {
        0: [reviewed(0, "left", 10.0), reviewed(1, "left", 10.0)],
        2: [ViewLabel(0, "left", "no_hand", None, "person", "tester"), ViewLabel(1, "left", "rejected", None, "person", "tester")],
    }
    frames: list[TrackFrame] = run(proposals, labels)
    assert (at(frames, 2, 0, "left").origin, at(frames, 2, 0, "left").box) == ("reviewed", None)
    assert at(frames, 3, 0, "left").origin == "none"  # the chain ended; 0.4 is not a confident detection
    assert (at(frames, 2, 1, "left").origin, at(frames, 2, 1, "left").box) == ("reviewed", None)
    assert at(frames, 3, 1, "left").origin == "chained"  # rejected only at that time


def test_two_tracks_never_share_one_proposal() -> None:
    # Both hands reviewed side by side; at t1 only the left hand is detected and it overlaps both previous boxes.
    labels: dict[int, list[ViewLabel]] = {0: [reviewed(0, "left", 0.0), reviewed(0, "right", 60.0)]}
    frames: list[TrackFrame] = run({1: [ego(0, "left", 10.0)]}, labels)
    assert at(frames, 1, 0, "left").origin == "chained"
    assert at(frames, 1, 0, "right").origin == "held"


def test_track_cache_recomputes_after_a_label_write() -> None:
    store: LabelStore = LabelStore("seg", split="train", application_id="test")

    class Proposals:
        def proposals(self, segment_id: str, t_ns: int) -> list[Proposal]:
            return [ego(0, "left", 10.0, 0.9)]

    cache: TrackCache = TrackCache("seg", times_ns=TIMES, cams=CAMS, proposals=Proposals(), store=store)
    window: list[TrackFrame] = cache.frames(TIMES[2], 3)
    assert [frame.t_ns for frame in window] == TIMES[2:5]
    assert at(window, 0, 0, "left").origin == "chained"
    store.put(FrameLabels("seg", TIMES[3], [ViewLabel(0, "left", "no_hand", None, "person", "tester")]))
    cache.invalidate_from(TIMES[3])
    again: list[TrackFrame] = cache.frames(TIMES[2], 3)
    assert [at(again, i, 0, "left").origin for i in range(3)] == ["chained", "reviewed", "detected"]
    assert len(cache.frames(TIMES[-2], 10)) == 2
    with pytest.raises(ValueError, match="grid"):
        cache.frames(TIMES[0] + 1, 2)
