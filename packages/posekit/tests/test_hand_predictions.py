"""Public boundary tests for the hand prediction containers."""

import pytest
import torch

from posekit.predictions import BoxDetections, HandDetections, HandPoses3d, Keypoints3d
from posekit.skeletons import HAND_21, SOMA_HAND


def _boxes(count: int) -> BoxDetections:
    return BoxDetections(
        xyxy=torch.zeros((count, 4), dtype=torch.float32),
        scores=torch.ones((count,), dtype=torch.float32),
        frame_indices=torch.zeros((count,), dtype=torch.long),
    )


def test_hand_detections_pair_every_box_with_a_side() -> None:
    """A side per box row; a missing or extra side is a construction error."""
    detections = HandDetections(_boxes(2), ("left", "right"))
    assert detections.num_detections == 2
    with pytest.raises(ValueError, match="sides"):
        HandDetections(_boxes(2), ("left",))


def test_empty_hand_detections_have_no_rows() -> None:
    empty = HandDetections.empty("cpu")
    assert empty.num_detections == 0
    assert empty.sides == ()


def _soma_hands(count: int) -> HandPoses3d:
    # Joint i sits at x = i so the HAND_21 view is checkable by value.
    xyz = torch.arange(25, dtype=torch.float32)[None, :, None].expand(count, 25, 3).contiguous()
    return HandPoses3d(
        keypoints=Keypoints3d(
            xy=xyz[..., :2].contiguous(),
            z_root_relative=xyz[..., 2] - xyz[:, :1, 2],
            scores=torch.ones((count, 25), dtype=torch.float32),
            frame_indices=torch.zeros((count,), dtype=torch.long),
            skeleton=SOMA_HAND,
            root_indices=(0,),
            xyz_camera=xyz,
        ),
        sides=("left",) * count,
    )


def test_soma_hands_expose_the_hand21_view_without_metacarpals() -> None:
    """SOMA-25 drops the four non-thumb metacarpals (5, 10, 15, 20) to become HAND_21."""
    hand21 = _soma_hands(2).as_hand21()
    assert hand21.keypoints.skeleton == HAND_21
    assert hand21.keypoints.xyz_camera is not None
    expected = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    assert hand21.keypoints.xyz_camera[0, :, 0].tolist() == expected
    assert hand21.keypoints.xy[1, :, 0].tolist() == expected
    assert hand21.sides == ("left", "left")


def test_hand_poses_reject_misaligned_sides() -> None:
    with pytest.raises(ValueError, match="sides"):
        HandPoses3d(keypoints=_soma_hands(2).keypoints, sides=("left",))
