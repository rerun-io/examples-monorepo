"""Detector box metrics on hand-built frames: a hand's box is the square around its circle, in net pixels."""
import pytest
import torch

from handtrack.eval.boxes import BoxCounts, box_counts


def circle(cx: float, cy: float, r: float) -> list[float]:
    """Net-pixel circle in the normalized target layout (cx / 640, cy / 480, r / 640)."""
    return [cx / 640, cy / 480, r / 640]


NONE: list[float] = [0.0, 0.0, 0.0]


def frame(pred: list[list[float]], prob: list[float], truth: list[list[float]], presence: list[float], scored: list[bool] | None = None) -> BoxCounts:
    return box_counts(torch.tensor([pred]), torch.tensor([prob]), torch.tensor([truth]), torch.tensor([presence]), torch.tensor([scored or [True, True]]))


def test_an_exact_box_on_a_present_hand_is_detected() -> None:
    counts = frame([circle(200, 240, 50), NONE], [0.9, 0.1], [circle(200, 240, 50), NONE], [1.0, 0.0])
    assert (counts.present, counts.detected, counts.predicted, counts.swapped) == (1, 1, 1, 0)
    assert counts.recall == 1.0 and counts.precision == 1.0
    assert counts.median_iou == pytest.approx(1.0, abs=0.01)


def test_a_box_shifted_by_half_its_side_has_iou_one_third_and_misses() -> None:
    # Squares of side 100 overlapping 50 x 100: IoU = 5000 / 15000.
    counts = frame([circle(250, 240, 50), NONE], [0.9, 0.1], [circle(200, 240, 50), NONE], [1.0, 0.0])
    assert (counts.present, counts.detected, counts.predicted) == (1, 0, 1)
    assert counts.median_iou == pytest.approx(1 / 3, abs=0.01)


def test_a_hand_found_in_the_other_slot_is_a_side_swap() -> None:
    counts = frame([circle(400, 240, 60), NONE], [0.9, 0.2], [NONE, circle(400, 240, 60)], [0.0, 1.0])
    assert (counts.present, counts.detected, counts.predicted, counts.swapped) == (1, 0, 1, 1)
    assert counts.side_accuracy == 0.0


def test_positives_on_a_hand_free_frame_are_false_alarms() -> None:
    counts = frame([circle(100, 100, 40), circle(500, 100, 40)], [0.7, 0.3], [NONE, NONE], [0.0, 0.0])
    assert (counts.absent_frames, counts.absent_frame_alarms, counts.predicted) == (1, 1, 1)
    assert counts.false_alarms_per_1k == 1000.0
    assert counts.recall is None


def test_an_unscored_partial_hand_counts_nowhere() -> None:
    counts = frame([circle(200, 240, 50), NONE], [0.9, 0.1], [NONE, NONE], [0.0, 0.0], scored=[False, True])
    assert (counts.present, counts.predicted, counts.absent_frames) == (0, 0, 0)


def test_counts_add_up_across_batches() -> None:
    hit = frame([circle(200, 240, 50), NONE], [0.9, 0.1], [circle(200, 240, 50), NONE], [1.0, 0.0])
    alarm = frame([circle(100, 100, 40), NONE], [0.7, 0.3], [NONE, NONE], [0.0, 0.0])
    total = hit + alarm
    assert (total.present, total.detected, total.predicted, total.absent_frames, total.absent_frame_alarms) == (1, 1, 2, 1, 1)
    assert total.precision == 0.5
