"""Full-frame hand producer public seams without CUDA or downloaded models."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from posekit.predictions import Keypoints2d
from posekit.skeletons import COCO_133

import selfcap_curate.hands_gpu as module
from selfcap_curate.hands_gpu import detect_frames, full_frame_boxes, pick_oak_frames
from selfcap_curate.records import HAND_LATENCY_MS, HandTrack


@pytest.mark.parametrize("flag", [None, "quest_unsynced_kept_as_cut", "quest_pose_unsynced"])
def test_oak_motion_selection_uses_aligned_hands_latency_spacing_and_margin(flag, fixture_sync) -> None:
    sync = fixture_sync()
    sync = replace(sync, offsets_ms={"quest_pose": 100.0}, oak_rgb_model=replace(sync.oak_rgb_model, a_ms=0.0, b_ms_per_s=0.0))
    times = np.arange(100, dtype=np.float64) / 10.0 + 0.048 - 0.1
    speed = np.zeros(100)
    speed[[0, 20, 22, 30, 98]] = [1.0, 0.9, 0.8, 0.7, 1.0]
    hand = HandTrack(times, np.zeros((100, 21, 3)), np.ones(100, dtype=bool), speed)
    sync = replace(sync, flags=[] if flag is None else [flag])
    assert pick_oak_frames({"left": hand, "right": hand}, 100, 10.0, sync).tolist() == ([20, 30] if flag is None else [])


def test_full_frame_pose_keeps_every_frame_and_frozen_score_precision() -> None:
    class Pose:
        def __call__(self, images, boxes):
            assert boxes.xyxy.tolist() == [[0.0, 0.0, 7.0, 5.0]] * len(images)
            return Keypoints2d(torch.full((len(images), 133, 2), 4.0), torch.full((len(images), 133), 0.7), boxes.frame_indices, COCO_133)

    frames = torch.zeros((10, 5, 7, 3), dtype=torch.uint8)
    result = detect_frames(frames, np.arange(10, dtype=np.float64), full_frame_boxes, Pose())
    assert result.xy.shape == (10, 133, 2)
    assert np.all(result.xy == 4.0)
    assert np.all(result.scores == float(np.float16(0.7)))


def test_oak_picker_uses_the_shared_hand_latency(monkeypatch, fixture_sync) -> None:

    assert (HAND_LATENCY_MS / 1000.0).hex() == (0.048).hex()
    monkeypatch.setattr(module, "HAND_LATENCY_MS", 60.0, raising=False)
    sync = fixture_sync(flags=[], offsets_ms={"quest_pose": 100.0}, oak_rgb_model=replace(fixture_sync().oak_rgb_model, a_ms=0.0, b_ms_per_s=0.0))
    times = np.arange(1000, dtype=np.float64) / 100.0 + 0.06 - 0.1
    speed = np.zeros(1000)
    speed[300] = 1.0
    hand = HandTrack(times, np.zeros((1000, 21, 3)), np.ones(1000, dtype=bool), speed)
    assert module.pick_oak_frames({"left": hand}, 1000, 100.0, sync).tolist() == [300]
