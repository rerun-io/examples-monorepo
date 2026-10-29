"""Public frame-planning and detection-container seams, without GPU assets."""

import numpy as np
import torch
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.skeletons import COCO_133

from selfcap_curate.hands_gpu import detect_frames, pick_frames
from selfcap_curate.records import HandTrack


def test_two_gops_pick_twelve_spaced_motion_peaks() -> None:
    times = np.arange(600, dtype=np.float64) / 25.0
    speed = np.zeros(600)
    expected = [80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 330, 334, 338, 342, 346, 350, 354, 358, 362, 366, 370, 374]
    speed[expected] = np.arange(1, 25)
    speed[:50] = 1000.0  # Margin excludes these even though they move fastest.
    hand = HandTrack(times, np.zeros((600, 21, 3)), np.ones(600, dtype=bool), speed)
    assert pick_frames({"left": hand, "right": hand}, 600, 25.0).tolist() == expected


def test_top_person_batches_and_empty_frames_keep_container_shape() -> None:
    class Detector:
        def __call__(self, frames: torch.Tensor) -> BoxDetections:
            return BoxDetections(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), torch.tensor([0.4, 0.9]), torch.tensor([0, 0]))

    class Pose:
        def __call__(self, frames: torch.Tensor, boxes: BoxDetections) -> Keypoints2d:
            assert boxes.xyxy.tolist() == [[5.0, 6.0, 7.0, 8.0]]
            return Keypoints2d(torch.ones((1, 133, 2)), torch.full((1, 133), 0.7), boxes.frame_indices, COCO_133)

    result = detect_frames(torch.zeros((10, 4, 4, 3), dtype=torch.uint8), np.arange(10, dtype=np.float64), Detector(), Pose())
    assert result.xy.shape == (10, 133, 2)
    assert result.scores.shape == (10, 133)
    assert np.isnan(result.xy[1:8]).all()
    assert result.xy[[0, 8]].tolist() == np.ones((2, 133, 2)).tolist()
    assert result.scores[0, 0] == float(np.float16(0.7))
