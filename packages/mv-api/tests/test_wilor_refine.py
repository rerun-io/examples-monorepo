"""Pure hand-box selection used by the WiLoR exo/ego tools."""

import numpy as np
import torch
from posekit.predictions import BoxDetections, HandDetections

from mv_api.wilor_refine import best_hand_per_side


def test_keeps_the_most_confident_box_of_each_side_in_one_frame() -> None:
    detections = HandDetections(
        BoxDetections(
            xyxy=torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5], [6, 6, 7, 7]], dtype=torch.float32),
            scores=torch.tensor([0.4, 0.9, 0.8, 0.95], dtype=torch.float32),
            frame_indices=torch.tensor([0, 0, 0, 1], dtype=torch.long),
        ),
        ("left", "left", "right", "right"),
    )
    best = best_hand_per_side(detections, frame=0)
    np.testing.assert_array_equal(best["left"], [2, 2, 3, 3])
    np.testing.assert_array_equal(best["right"], [4, 4, 5, 5])


def test_a_side_without_boxes_is_absent() -> None:
    best = best_hand_per_side(HandDetections.empty("cpu"), frame=0)
    assert best == {}
