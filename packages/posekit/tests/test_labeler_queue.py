"""Review queue, range fill and frame flags."""

from __future__ import annotations

import pytest

from posekit.labeler.queue import ReviewQueue, effective_boxes, frame_flags, iou, lerp_box
from posekit.labeler.schema import Box, FrameLabels, HandSide, Proposal, QueueItem, RangeRequest, RangeResult, ViewLabel
from posekit.labeler.store import LabelStore, RangeWrite

SEGMENT: str = "seg"
STEP: int = 200_000_000
TIMES: list[int] = [i * STEP for i in range(11)]
KEYFRAMES: list[int] = [0, 5 * STEP, 10 * STEP]
CAMS: list[int] = [0, 1]


class FakeProposals:
    """Proposals keyed by time."""

    def __init__(self) -> None:
        self.by_time: dict[int, list[Proposal]] = {}

    def proposals(self, segment_id: str, t_ns: int) -> list[Proposal]:
        assert segment_id == SEGMENT
        return self.by_time.get(t_ns, [])


def make_queue() -> tuple[ReviewQueue, LabelStore, FakeProposals]:
    store: LabelStore = LabelStore(SEGMENT, split="train", application_id="test")
    proposals: FakeProposals = FakeProposals()
    queue: ReviewQueue = ReviewQueue(SEGMENT, times_ns=TIMES, keyframes_ns=KEYFRAMES, cams=CAMS, store=store, proposals=proposals)
    return queue, store, proposals


def box_at(x: float) -> Box:
    return Box(x, 100.0, x + 100.0, 200.0)


def review(store: LabelStore, t_ns: int, left: Box | None, right: Box | None) -> None:
    """Label every (cam, side) at t_ns: cam 0 gets the given boxes (None = no_hand), cam 1 is no_hand."""
    labels: list[ViewLabel] = []
    for side, box in (("left", left), ("right", right)):
        hand_side: HandSide = "left" if side == "left" else "right"
        labels.append(ViewLabel(0, hand_side, "corrected" if box else "no_hand", box, "person", "pablo"))
        labels.append(ViewLabel(1, hand_side, "no_hand", None, "person", "pablo"))
    store.put(FrameLabels(SEGMENT, t_ns, labels))


def test_iou_and_lerp() -> None:
    assert iou(box_at(0.0), box_at(0.0)) == pytest.approx(1.0)
    assert iou(box_at(0.0), box_at(50.0)) == pytest.approx(5000.0 / 15000.0)
    assert iou(box_at(0.0), box_at(500.0)) == 0.0
    assert lerp_box(box_at(0.0), box_at(100.0), 0.25) == box_at(25.0)


def test_unreviewed_keyframes_come_first() -> None:
    queue, store, _ = make_queue()
    assert [item.t_ns for item in queue.items(50)] == KEYFRAMES
    assert all(item.reasons[0] == "keyframe" for item in queue.items(50))
    assert len(queue.items(2)) == 2
    review(store, 0, None, None)
    assert [item.t_ns for item in queue.items(50)] == KEYFRAMES[1:]
    # A partly reviewed keyframe stays in the queue.
    store.put(FrameLabels(SEGMENT, 5 * STEP, [ViewLabel(0, "left", "no_hand", None, "person", "pablo")]))
    assert [item.t_ns for item in queue.items(50)] == KEYFRAMES[1:]
    assert queue.n_reviewed_keyframes() == 1


def test_fill_range_no_hand_and_tracked_box() -> None:
    queue, store, proposals = make_queue()
    review(store, 0, box_at(0.0), None)
    review(store, 5 * STEP, box_at(100.0), None)
    for step in range(1, 5):
        # EgoHand tracks the left hand along the interpolated path, slightly off.
        proposals.by_time[step * STEP] = [Proposal(0, "left", box_at(step * 20.0 + 3.0), 0.9, "egohand")]
    result: RangeResult = queue.fill_range(RangeRequest(SEGMENT, 0, 5 * STEP, "pablo"))
    assert result == RangeResult(filled=4 * 4, queued=[])
    label: ViewLabel | None = store.get_view(2 * STEP, 0, "left")
    assert label == ViewLabel(0, "left", "accepted", box_at(43.0), "range", "pablo")
    assert store.get_view(2 * STEP, 0, "right") == ViewLabel(0, "right", "no_hand", None, "range", "pablo")
    assert store.get_view(2 * STEP, 1, "left") == ViewLabel(1, "left", "no_hand", None, "range", "pablo")
    end: ViewLabel | None = store.get_view(5 * STEP, 0, "left")
    assert end is not None and end.source == "person"


def test_fill_range_queues_inconsistent_times() -> None:
    queue, store, proposals = make_queue()
    review(store, 0, box_at(0.0), None)
    review(store, 5 * STEP, box_at(100.0), None)
    proposals.by_time[1 * STEP] = [Proposal(0, "left", box_at(800.0), 0.9, "egohand")]  # far from the path
    proposals.by_time[2 * STEP] = [Proposal(0, "left", box_at(40.0), 0.9, "egohand"), Proposal(0, "right", box_at(400.0), 0.7, "egohand")]
    proposals.by_time[3 * STEP] = [Proposal(0, "left", box_at(60.0), 0.9, "egohand")]
    # 4 * STEP: no proposal at all for the left hand.
    # A reviewed label inside the range is never overwritten.
    reviewed: ViewLabel = ViewLabel(1, "left", "rejected", None, "person", "pablo")
    store.put(FrameLabels(SEGMENT, 3 * STEP, [reviewed]))
    result: RangeResult = queue.fill_range(RangeRequest(SEGMENT, 0, 5 * STEP, "astra"))
    assert result.queued == [1 * STEP, 2 * STEP, 4 * STEP]
    assert store.get_view(1 * STEP, 0, "left") is None
    assert store.get_view(2 * STEP, 0, "left") is not None  # left was consistent
    assert store.get_view(2 * STEP, 0, "right") is None  # confident right proposal vs no_hand ends
    assert store.get_view(3 * STEP, 1, "left") == reviewed
    assert result.filled == 3 + 3 + 3 + 3
    items: list[QueueItem] = queue.items(50)
    assert [item.t_ns for item in items] == [10 * STEP, 1 * STEP, 2 * STEP, 4 * STEP]
    assert items[1].reasons[0] == "range_gap"
    # Reviewing a gap removes it from the queue.
    review(store, 1 * STEP, box_at(20.0), None)
    assert [item.t_ns for item in queue.items(50)] == [10 * STEP, 2 * STEP, 4 * STEP]


def test_fill_range_overwrites_old_range_labels() -> None:
    queue, store, proposals = make_queue()
    review(store, 0, None, None)
    review(store, 5 * STEP, None, None)
    assert queue.fill_range(RangeRequest(SEGMENT, 0, 5 * STEP, "pablo")).filled == 16
    proposals.by_time[2 * STEP] = [Proposal(1, "right", box_at(0.0), 0.8, "egohand")]
    result: RangeResult = queue.fill_range(RangeRequest(SEGMENT, 0, 5 * STEP, "pablo"))
    assert result.queued == [2 * STEP]
    assert result.filled == 15


@pytest.mark.parametrize(
    ("t0", "t1", "message"),
    [(0, 3 * STEP, "keyframes"), (5 * STEP, 0, "before"), (0, 10 * STEP, "not fully reviewed")],
)
def test_fill_range_rejects_bad_ends(t0: int, t1: int, message: str) -> None:
    queue, store, _ = make_queue()
    review(store, 0, None, None)
    review(store, 5 * STEP, None, None)
    with pytest.raises(ValueError, match=message):
        queue.fill_range(RangeRequest(SEGMENT, t0, t1, "pablo"))


def test_flags() -> None:
    left_now: dict[tuple[int, HandSide], Box] = {(0, "left"): box_at(0.0)}
    right_before: dict[tuple[int, HandSide], Box] = {(0, "right"): box_at(5.0)}
    assert frame_flags([], right_before, left_now) == ["side_flip"]
    assert frame_flags([], {**right_before, (0, "left"): box_at(0.0)}, left_now) == []
    disagree: list[Proposal] = [Proposal(2, "left", box_at(0.0), 0.9, "egohand"), Proposal(2, "left", box_at(300.0), 1.0, "hands_layer")]
    assert frame_flags(disagree, {}, {}) == ["views_disagree"]
    agree: list[Proposal] = [Proposal(2, "left", box_at(0.0), 0.9, "egohand"), Proposal(2, "left", box_at(10.0), 1.0, "hands_layer")]
    assert frame_flags(agree, {}, {}) == []
    assert frame_flags([Proposal(1, "right", box_at(0.0), 0.4, "egohand")], {}, {}) == ["low_score"]
    assert frame_flags([Proposal(1, "right", box_at(0.0), 0.2, "egohand")], {}, {}) == []


def test_effective_boxes_prefer_labels() -> None:
    labels: list[ViewLabel] = [ViewLabel(0, "left", "no_hand", None, "person", "p"), ViewLabel(1, "left", "corrected", box_at(9.0), "person", "p")]
    proposals: list[Proposal] = [
        Proposal(0, "left", box_at(0.0), 0.9, "egohand"),
        Proposal(1, "left", box_at(0.0), 0.9, "egohand"),
        Proposal(2, "left", box_at(0.0), 0.4, "egohand"),
        Proposal(3, "right", box_at(0.0), 0.9, "egohand"),
    ]
    assert effective_boxes(labels, proposals) == {(1, "left"): box_at(9.0), (3, "right"): box_at(0.0)}


def test_queue_items_carry_flags() -> None:
    queue, _, proposals = make_queue()
    proposals.by_time[0] = [Proposal(0, "left", box_at(0.0), 0.35, "egohand")]
    assert queue.items(1) == [QueueItem(SEGMENT, 0, ["keyframe", "low_score"])]


def test_refill_removes_range_labels_it_can_no_longer_decide() -> None:
    queue, store, proposals = make_queue()
    review(store, 0, box_at(0.0), None)
    review(store, 5 * STEP, box_at(100.0), None)
    for step in range(1, 5):
        proposals.by_time[step * STEP] = [Proposal(0, "left", box_at(step * 20.0), 0.9, "egohand")]
    assert queue.fill_range(RangeRequest(SEGMENT, 0, 5 * STEP, "astra")).queued == []
    # The reviewer moves the end box far away: the middle proposals no longer follow the interpolated path.
    review(store, 5 * STEP, box_at(1000.0), None)
    result: RangeResult = queue.fill_range(RangeRequest(SEGMENT, 0, 5 * STEP, "astra"))
    assert result.queued == [1 * STEP, 2 * STEP, 3 * STEP, 4 * STEP]
    assert store.get_view(2 * STEP, 0, "left") is None  # the stale accepted box is gone
    assert store.get_view(2 * STEP, 0, "right") == ViewLabel(0, "right", "no_hand", None, "range", "astra")
    assert [item.t_ns for item in queue.items(50)] == [10 * STEP, 1 * STEP, 2 * STEP, 3 * STEP, 4 * STEP]


class RacingStore(LabelStore):
    """A store where a reviewer's post lands while a range fill runs: on the second read of the start keyframe.

    The first read is the fill's check that the ends are reviewed; the post lands after it, before any write.
    """

    def __init__(self, post: FrameLabels) -> None:
        super().__init__(SEGMENT, split="train", application_id="test")
        self.post: FrameLabels | None = post
        self.start_reads: int = 0

    def _race(self, t_ns: int) -> None:
        if t_ns != 0 or self.post is None:
            return
        self.start_reads += 1
        if self.start_reads == 2:
            post: FrameLabels = self.post
            self.post = None
            self.put(post)

    def get(self, t_ns: int) -> list[ViewLabel]:
        self._race(t_ns)
        return super().get(t_ns)

    def get_view(self, t_ns: int, cam: int, side: HandSide) -> ViewLabel | None:
        self._race(t_ns)
        return super().get_view(t_ns, cam, side)


def test_fill_never_overwrites_a_label_posted_during_the_fill() -> None:
    correction: ViewLabel = ViewLabel(0, "left", "corrected", box_at(7.0), "person", "pablo")
    store: RacingStore = RacingStore(FrameLabels(SEGMENT, 1 * STEP, [correction]))
    review(store, 0, None, None)
    review(store, 5 * STEP, None, None)
    queue: ReviewQueue = ReviewQueue(SEGMENT, times_ns=TIMES, keyframes_ns=KEYFRAMES, cams=CAMS, store=store, proposals=FakeProposals())
    store.start_reads = 0  # count the fill's reads only
    queue.fill_range(RangeRequest(SEGMENT, 0, 5 * STEP, "astra"))
    assert store.post is None
    assert store.get_view(1 * STEP, 0, "left") == correction


def test_range_write_keeps_reviewed_labels_and_drops_undecided_range_labels() -> None:
    _, store, _ = make_queue()
    reviewed: ViewLabel = ViewLabel(0, "left", "corrected", box_at(5.0), "person", "pablo")
    store.put(FrameLabels(SEGMENT, STEP, [reviewed, ViewLabel(1, "left", "no_hand", None, "range", "astra")]))
    decisions: dict[tuple[int, HandSide], ViewLabel | None] = {
        (0, "left"): ViewLabel(0, "left", "accepted", box_at(0.0), "range", "astra"),
        (0, "right"): ViewLabel(0, "right", "no_hand", None, "range", "astra"),
        (1, "left"): None,
        (1, "right"): None,
    }
    assert store.put_range(STEP, decisions) == RangeWrite(stored=1, open_views=2)
    assert store.get(STEP) == [reviewed, ViewLabel(0, "right", "no_hand", None, "range", "astra")]


def test_restart_keeps_open_range_gaps() -> None:
    queue, store, proposals = make_queue()
    review(store, 0, box_at(0.0), None)
    review(store, 5 * STEP, box_at(100.0), None)
    proposals.by_time[1 * STEP] = [Proposal(0, "left", box_at(800.0), 0.9, "egohand")]  # far from the path
    proposals.by_time[3 * STEP] = [Proposal(0, "left", box_at(60.0), 0.9, "egohand")]
    assert queue.fill_range(RangeRequest(SEGMENT, 0, 5 * STEP, "astra")).queued == [1 * STEP, 2 * STEP, 4 * STEP]
    review(store, 10 * STEP, None, None)  # reviewed, but no fill ran towards it: nothing to reconstruct there
    restarted: ReviewQueue = ReviewQueue(SEGMENT, times_ns=TIMES, keyframes_ns=KEYFRAMES, cams=CAMS, store=store, proposals=proposals)
    items: list[QueueItem] = restarted.items(50)
    assert [(item.t_ns, item.reasons[0]) for item in items] == [(1 * STEP, "range_gap"), (2 * STEP, "range_gap"), (4 * STEP, "range_gap")]
