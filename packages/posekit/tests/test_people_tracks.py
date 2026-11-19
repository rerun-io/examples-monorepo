"""Unit tests for the people-track bookkeeping (synthetic arrays; no GPU, no weights)."""

import numpy as np
import torch
from jaxtyping import Bool, Float32
from numpy import ndarray
from torch import Tensor


def test_mask_iou_pairs_and_empty_masks() -> None:
    from posekit.ops.people_tracks import mask_iou

    a: Bool[Tensor, "n h w"] = torch.zeros((2, 4, 4), dtype=torch.bool)
    a[0, 0:2, 0:2] = True  # 4 px
    b: Bool[Tensor, "m h w"] = torch.zeros((3, 4, 4), dtype=torch.bool)
    b[0, 0:2, 0:4] = True  # 8 px, contains a[0]
    b[1, 2:4, 2:4] = True  # disjoint

    iou: Float32[Tensor, "n m"] = mask_iou(a, b)

    assert iou.shape == (2, 3)
    assert torch.allclose(iou[0], torch.tensor([0.5, 0.0, 0.0]))
    assert torch.equal(iou[1], torch.zeros(3))  # empty track mask


def test_greedy_match_is_one_to_one_highest_first() -> None:
    from posekit.ops.people_tracks import greedy_match

    iou: Float32[ndarray, "n m"] = np.array([[0.9, 0.8], [0.85, 0.1], [0.2, 0.2]], dtype=np.float32)

    assert greedy_match(iou, threshold=0.3) == [(0, 0)]
    pairs: list[tuple[int, int]] = greedy_match(iou, threshold=0.05)
    assert pairs == [(0, 0), (2, 1)]  # (2, 1) at 0.2 beats (1, 1) at 0.1 for column 1


def test_greedy_match_without_rows_or_columns() -> None:
    from posekit.ops.people_tracks import greedy_match

    assert greedy_match(np.zeros((0, 3), dtype=np.float32), threshold=0.3) == []
    assert greedy_match(np.zeros((2, 0), dtype=np.float32), threshold=0.3) == []


def test_plan_redetect_reanchors_starts_and_drops_duplicates() -> None:
    from posekit.ops.people_tracks import RedetectPlan, plan_redetect

    # Tracks 7 and 9 both follow detection 0; track 9 overlaps less and is a duplicate. Track 4 sees nothing (kept).
    # Detection 1 is new.
    iou: Float32[ndarray, "n m"] = np.array([[0.8, 0.0], [0.5, 0.0], [0.0, 0.1]], dtype=np.float32)

    plan: RedetectPlan = plan_redetect([7, 9, 4], iou, np.eye(2, dtype=np.float32), threshold=0.3, next_id=10, max_tracks=8)

    assert plan.reanchor == ((7, 0),)
    assert plan.drop == (9,)
    assert plan.start == ((10, 1),)
    assert plan.unmatched == (4,)
    assert plan.next_id == 11


def test_plan_redetect_first_frame_and_track_cap() -> None:
    from posekit.ops.people_tracks import RedetectPlan, plan_redetect

    plan: RedetectPlan = plan_redetect([], np.zeros((0, 3), dtype=np.float32), np.eye(3, dtype=np.float32), threshold=0.3, next_id=0, max_tracks=2)

    assert plan.reanchor == () and plan.drop == ()
    assert plan.start == ((0, 0), (1, 1))
    assert plan.next_id == 2


def test_plan_redetect_starts_one_track_per_person_when_tiles_see_them_twice() -> None:
    from posekit.ops.people_tracks import RedetectPlan, plan_redetect

    # Detections 0 (whole image) and 1 (a tile) are the same person; detection 2 is someone else.
    detection_iou: Float32[ndarray, "m m"] = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    plan: RedetectPlan = plan_redetect([], np.zeros((0, 3), dtype=np.float32), detection_iou, threshold=0.3, next_id=0, max_tracks=2)

    assert plan.start == ((0, 0), (1, 2))


def test_plan_redetect_without_detections_keeps_tracks() -> None:
    from posekit.ops.people_tracks import RedetectPlan, plan_redetect

    plan: RedetectPlan = plan_redetect([3, 5], np.zeros((2, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), threshold=0.3, next_id=6, max_tracks=8)

    assert plan == RedetectPlan(reanchor=(), start=(), drop=(), unmatched=(3, 5), next_id=6)


def test_count_misses_resets_counts_and_drops_after_max() -> None:
    from posekit.ops.people_tracks import count_misses

    misses: dict[int, int] = {1: 2, 2: 1}
    updated, dropped = count_misses(misses, {1: False, 2: True, 3: False}, max_misses=3)

    assert dropped == [1]
    assert updated == {2: 0, 3: 1}
