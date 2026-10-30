"""Seeded visibility sampling and independent-frame EgoHand inference."""

import numpy as np
import pytest
import torch

pytest.importorskip("egohand_inference", reason="EgoHand inference environment is required")

from posekit.models.egohand import HandDetections, HandReconstruction
from posekit.models.egohand_geometry import OAKCamera
from posekit.predictions import BoxDetections, Keypoints3d
from posekit.skeletons import SOMA_HAND

from selfcap_curate.oak_egohand import infer_egohand_frames, select_egohand_frames


def test_sampler_is_seeded_uniform_and_rejects_invisible_fast_untracked_hands():
    times = np.arange(200, dtype=np.float64) / 20.0
    pixels = np.full((200, 133, 2), np.nan, np.float32)
    pixels[:, 91:112] = np.stack([30.0 + times, np.full(200, 50.0)], axis=1)[:, None]
    pixels[:, 112:133] = [220.0, 50.0]
    pixels[30:50, 91:112, 0] = -10.0
    pixels[100:110, 91:112, 0] = np.arange(10)[:, None] * 25.0
    tracked = np.ones((200, 2), bool)
    tracked[150:160, 0] = False
    result = select_egohand_frames(times, pixels, tracked, (200, 100))
    repeat = select_egohand_frames(times, pixels, tracked, (200, 100))
    other = select_egohand_frames(times, pixels, tracked, (200, 100), seed=7)
    assert len(result.indices) == 64
    np.testing.assert_array_equal(result.indices, repeat.indices)
    assert not np.array_equal(result.indices, other.indices)
    assert np.all(np.diff(result.indices) > 0)
    assert result.hands[:, 0].all() and not result.hands[:, 1].any()
    assert not np.isin(result.indices, np.r_[30:50, 101:110, 150:160]).any()
    assert all(np.sum((result.indices >= lo) & (result.indices < lo + 50)) >= 5 for lo in range(0, 200, 50))
    tracked[:] = False
    assert len(select_egohand_frames(times, pixels, tracked, (200, 100)).indices) == 0


def test_inference_detects_each_frame_independently_and_preserves_box_score():
    calls = []

    def detector(frames):
        calls.append(len(frames))
        return HandDetections(BoxDetections(torch.zeros((1, 4)), torch.tensor([0.75]), torch.tensor([0])), ("left",))

    def predictor(frames, detections, camera):
        n = len(detections.sides)
        xy = torch.arange(50, dtype=torch.float32).reshape(1, 25, 2).repeat(n, 1, 1)
        points = Keypoints3d(xy, torch.zeros((n, 25)), detections.boxes.scores[:, None].repeat(1, 25), detections.boxes.frame_indices,
                            SOMA_HAND, (0,), torch.zeros((n, 25, 3)))
        return HandReconstruction(points, detections.sides, torch.zeros((n, 25, 3, 3)), torch.zeros((n, 1)),
                                  torch.ones(n), torch.zeros((n, 3)), None, tuple(torch.zeros((1, 3)) for _ in range(n)))

    camera = OAKCamera(np.eye(3), np.zeros(14), 30, 20)
    result = infer_egohand_frames(torch.zeros((6, 20, 30, 3), dtype=torch.uint8), np.arange(6, dtype=np.float64),
                                  np.tile([True, False], (6, 1)), camera, detector, predictor)
    assert calls == [1] * 6
    assert np.all(result.scores[:, 91:112] == 0.75)
    assert np.all(result.scores[:, 112:] == 0.0)
    np.testing.assert_array_equal(result.xy[0, 91:96], np.arange(10).reshape(5, 2))
    np.testing.assert_array_equal(result.xy[0, 96], [12.0, 13.0])
