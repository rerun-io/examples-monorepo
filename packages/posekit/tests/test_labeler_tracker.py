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


def no_hand(cam: int, side: HandSide) -> ViewLabel:
    return ViewLabel(cam, side, "no_hand", None, "agent", "astra")


def test_a_false_box_removed_once_does_not_come_back() -> None:
    # EgoHand keeps proposing a false "left" box that drifts 5 px per step; the reviewer says no hand at t1.
    proposals: dict[int, list[Proposal]] = {i: [ego(0, "left", 500.0 + 5.0 * i, 0.9)] for i in range(6)}
    frames: list[TrackFrame] = run(proposals, {1: [no_hand(0, "left")]})
    assert at(frames, 0, 0, "left").origin == "detected"
    assert (at(frames, 1, 0, "left").origin, at(frames, 1, 0, "left").box) == ("reviewed", None)
    assert [at(frames, i, 0, "left").origin for i in (2, 3, 4)] == ["none"] * 3
    assert [at(frames, i, 0, "right").origin for i in (2, 3, 4)] == ["none"] * 3  # claimed for every side
    # Without the no_hand label the same box is shown every frame.
    assert [at(run(proposals), i, 0, "left").origin for i in (2, 3, 4)] == ["chained"] * 3


def test_a_false_box_on_a_chain_does_not_come_back() -> None:
    # A reviewed chain walks onto a false box; the reviewer says no hand at t2: the chain's continuation becomes negative.
    proposals: dict[int, list[Proposal]] = {i: [ego(0, "right", 10.0 + 10.0 * i, 0.9)] for i in range(6)}
    frames: list[TrackFrame] = run(proposals, {0: [reviewed(0, "left", 10.0)], 2: [no_hand(0, "left")]})
    assert at(frames, 1, 0, "left").origin == "chained"
    assert [at(frames, i, 0, side).origin for i in (3, 4, 5) for side in ("left", "right")] == ["none"] * 6


def test_a_real_hand_elsewhere_in_the_view_is_still_detected() -> None:
    proposals: dict[int, list[Proposal]] = {i: [ego(0, "left", 500.0, 0.9)] + ([ego(0, "left", 10.0, 0.7)] if i >= 2 else []) for i in range(6)}
    frames: list[TrackFrame] = run(proposals, {1: [no_hand(0, "left")]})
    for i in (2, 3, 4):
        found: TrackBox = at(frames, i, 0, "left")
        assert (found.origin, found.box) == (("detected", Box(10.0, 100.0, 110.0, 200.0)) if i == 2 else ("chained", Box(10.0, 100.0, 110.0, 200.0)))


def test_a_reviewer_box_ends_the_negative_track() -> None:
    proposals: dict[int, list[Proposal]] = {i: [ego(0, "left", 500.0, 0.9)] for i in range(6)}
    frames: list[TrackFrame] = run(proposals, {1: [no_hand(0, "left")], 3: [reviewed(0, "left", 500.0)]})
    assert at(frames, 2, 0, "left").origin == "none"
    assert at(frames, 3, 0, "left").origin == "reviewed"
    assert [at(frames, i, 0, "left").origin for i in (4, 5)] == ["chained", "chained"]


def test_a_negative_track_holds_one_miss_then_ends() -> None:
    once: dict[int, list[Proposal]] = {i: [ego(0, "left", 500.0)] for i in (0, 1, 3)}
    assert at(run(once, {1: [no_hand(0, "left")]}), 3, 0, "left").origin == "none"  # one miss (t2) holds the negative box
    twice: dict[int, list[Proposal]] = {i: [ego(0, "left", 500.0)] for i in (0, 1, 4)}
    assert at(run(twice, {1: [no_hand(0, "left")]}), 4, 0, "left").origin == "detected"  # two misses (t2, t3) end it


def test_no_negative_track_on_a_side_swap() -> None:
    # The reviewer moves a hand from left to right in one post: the box is a real hand, the right chain keeps it.
    proposals: dict[int, list[Proposal]] = {i: [ego(0, "left", 10.0, 0.9)] for i in range(4)}
    frames: list[TrackFrame] = run(proposals, {1: [no_hand(0, "left"), reviewed(0, "right", 10.0)]})
    assert [at(frames, i, 0, "right").origin for i in (2, 3)] == ["chained", "chained"]


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
    # The no_hand at t3 makes the same box a negative track, so it is not detected again at t4.
    assert [at(again, i, 0, "left").origin for i in range(3)] == ["chained", "reviewed", "none"]
    assert len(cache.frames(TIMES[-2], 10)) == 2
    with pytest.raises(ValueError, match="grid"):
        cache.frames(TIMES[0] + 1, 2)


# -- visual carry (MEgATrack: a tracker carries a reviewed box until it truly fails) ----------------------------------

LONG: list[int] = [i * STEP for i in range(14)]


class Drift:
    """Fake visual tracker: the start box moves ``dx`` px right per grid step; lost from grid index ``lost_at`` on."""

    def __init__(self, dx: float = 20.0, lost_at: int | None = None) -> None:
        self.dx: float = dx
        self.lost_at: int | None = lost_at
        self.calls: list[tuple[int, int, Box, int]] = []

    def carry(self, cam: int, start_ns: int, start_box: Box, t_ns: int) -> Box | None:
        self.calls.append((cam, start_ns, start_box, t_ns))
        if self.lost_at is not None and t_ns >= LONG[self.lost_at]:
            return None
        shift: float = self.dx * (t_ns - start_ns) / STEP
        return Box(start_box.x0 + shift, start_box.y0, start_box.x1 + shift, start_box.y1)


def carried(proposals: dict[int, list[Proposal]], labels: dict[int, list[ViewLabel]], carrier: Drift | None) -> list[TrackFrame]:
    """Propagate over ``LONG`` with an optional visual tracker; both dicts are keyed by grid index."""
    return propagate(LONG, CAMS, lambda t: proposals.get(LONG.index(t), []), lambda t: labels.get(LONG.index(t), []), carrier=carrier)


def test_a_reviewed_box_is_carried_for_ten_steps_then_the_chain_ends() -> None:
    carrier: Drift = Drift()
    frames: list[TrackFrame] = carried({}, {0: [reviewed(0, "left", 10.0)]}, carrier)
    assert [at(frames, i, 0, "left").origin for i in range(12)] == ["reviewed"] + ["tracked"] * 10 + ["none"]
    third: TrackBox = at(frames, 3, 0, "left")
    assert (third.box, third.score) == (Box(70.0, 100.0, 170.0, 200.0), None)
    # One tracker run, initialised on the reviewed frame and box.
    assert {(cam, start, box) for cam, start, box, _ in carrier.calls} == {(0, LONG[0], Box(10.0, 100.0, 110.0, 200.0))}
    # Without the flag the same labels hold once and end.
    off: list[TrackFrame] = carried({}, {0: [reviewed(0, "left", 10.0)]}, None)
    assert [at(off, i, 0, "left").origin for i in range(3)] == ["reviewed", "held", "none"]


def test_the_tracked_box_snaps_back_to_an_egohand_proposal() -> None:
    # The tracker moves 80 px per step: the proposal at step 3 misses the step-2 box (IoU 0.11) but matches the tracked box.
    carrier: Drift = Drift(dx=80.0)
    frames: list[TrackFrame] = carried({3: [ego(0, "right", 250.0, 0.4)]}, {0: [reviewed(0, "left", 10.0)]}, carrier)
    assert [at(frames, i, 0, "left").origin for i in range(5)] == ["reviewed", "tracked", "tracked", "chained", "tracked"]
    snapped: TrackBox = at(frames, 3, 0, "left")
    assert (snapped.box, snapped.score) == (Box(250.0, 100.0, 350.0, 200.0), 0.4)
    assert at(frames, 3, 0, "right").origin == "none"  # the proposal belongs to the carried left track
    # After the re-snap a new run starts from the proposal, and the ten-step budget starts again.
    assert carrier.calls[-1][1:3] == (LONG[3], Box(250.0, 100.0, 350.0, 200.0))
    assert [at(frames, i, 0, "left").origin for i in range(4, 14)] == ["tracked"] * 10


def test_tracker_failure_ends_the_chain() -> None:
    frames: list[TrackFrame] = carried({3: [ego(0, "left", 700.0, 0.9)]}, {0: [reviewed(0, "left", 10.0)]}, Drift(lost_at=3))
    assert [at(frames, i, 0, "left").origin for i in range(4)] == ["reviewed", "tracked", "tracked", "detected"]


def test_no_hand_ends_a_carried_chain() -> None:
    labels: dict[int, list[ViewLabel]] = {0: [reviewed(0, "left", 10.0)], 2: [ViewLabel(0, "left", "no_hand", None, "agent", "astra")]}
    frames: list[TrackFrame] = carried({}, labels, Drift())
    assert [at(frames, i, 0, "left").origin for i in range(4)] == ["reviewed", "tracked", "reviewed", "none"]


def test_only_chains_from_a_reviewed_box_are_carried() -> None:
    # A detected box chains and holds as before; the tracker never runs for it.
    carrier: Drift = Drift()
    frames: list[TrackFrame] = carried({0: [ego(1, "right", 10.0, 0.9)]}, {}, carrier)
    assert [at(frames, i, 1, "right").origin for i in range(3)] == ["detected", "held", "none"]
    assert carrier.calls == []


def test_machine_labels_continue_a_carried_chain_but_never_start_one() -> None:
    accepted: ViewLabel = ViewLabel(0, "left", "accepted", Box(30.0, 100.0, 130.0, 200.0), "tracker", "astra")
    # The reviewer accepted the carried box at step 1: the chain stays a reviewed chain and the tracker restarts there.
    carrier: Drift = Drift()
    frames: list[TrackFrame] = carried({}, {0: [reviewed(0, "left", 10.0)], 1: [accepted]}, carrier)
    assert [at(frames, i, 0, "left").origin for i in range(3)] == ["reviewed", "reviewed", "tracked"]
    assert carrier.calls[-1][1:3] == (LONG[1], accepted.box)
    # A tracker or range label with no reviewed chain behind it only holds.
    alone: list[TrackFrame] = carried({}, {1: [accepted]}, Drift())
    assert [at(alone, i, 0, "left").origin for i in range(1, 4)] == ["reviewed", "held", "none"]
    ranged: ViewLabel = ViewLabel(0, "left", "accepted", Box(30.0, 100.0, 130.0, 200.0), "range", "astra")
    assert at(carried({}, {1: [ranged]}, Drift()), 2, 0, "left").origin == "held"


def test_picked_egohand_boxes_are_reviewed_boxes() -> None:
    picked: ViewLabel = ViewLabel(0, "left", "accepted", Box(10.0, 100.0, 110.0, 200.0), "egohand", "astra")
    assert at(carried({}, {0: [picked]}, Drift()), 1, 0, "left").origin == "tracked"


def test_track_cache_carries_and_recomputes_with_the_same_boxes() -> None:
    store: LabelStore = LabelStore("seg", split="train", application_id="test")

    class Nothing:
        def proposals(self, segment_id: str, t_ns: int) -> list[Proposal]:
            return []

    store.put(FrameLabels("seg", LONG[0], [reviewed(0, "left", 10.0)]))
    cache: TrackCache = TrackCache("seg", times_ns=LONG, cams=CAMS, proposals=Nothing(), store=store, carrier=Drift())
    first: list[TrackFrame] = cache.frames(LONG[0], 5)
    assert [at(first, i, 0, "left").origin for i in range(5)] == ["reviewed"] + ["tracked"] * 4
    store.put(FrameLabels("seg", LONG[6], [reviewed(1, "right", 500.0)]))
    cache.invalidate_from(LONG[6])
    assert cache.frames(LONG[0], 5) == first
    assert at(cache.frames(LONG[7], 1), 0, 1, "right").origin == "tracked"
