import math

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from mvs.pose_stream import Keyframe, KeyframeBuffer, PoseDistance, pose_distance


def _translation_pose(x: float) -> Float[Tensor, "4 4"]:
    """Return an identity-orientation pose translated along x."""
    pose_44: Float[Tensor, "4 4"] = torch.eye(4, dtype=torch.float64)
    pose_44[0, 3] = x
    return pose_44


def _keyframe(frame_index: int, x: float, image_3hw: Tensor) -> Keyframe:
    """Return a keyframe translated along x with identity calibration."""
    pose_44: Float[Tensor, "4 4"] = _translation_pose(x)
    return Keyframe(frame_index, pose_44, torch.linalg.inv(pose_44), torch.eye(4, dtype=torch.float32), image_3hw)


def test_pose_distance_decomposes_identity_translation_and_rotation() -> None:
    """Pose distance reports the specified translation and trace-based rotation measures."""

    identity_44: Float[Tensor, "4 4"] = torch.eye(4, dtype=torch.float64)
    translated_44: Float[Tensor, "4 4"] = identity_44.clone()
    translated_44[0, 3] = 3.0
    rotated_44: Float[Tensor, "4 4"] = identity_44.clone()
    rotated_44[:2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)

    same: PoseDistance = pose_distance(identity_44, identity_44)
    translated: PoseDistance = pose_distance(identity_44, translated_44)
    rotated: PoseDistance = pose_distance(identity_44, rotated_44)

    assert same.combined == 0.0
    assert same.rotation == 0.0
    assert same.translation == 0.0
    assert translated.translation == 3.0
    assert translated.combined == 3.0
    assert translated.rotation == 0.0
    assert rotated.translation == 0.0
    assert rotated.rotation == rotated.combined == pytest.approx(math.sqrt(4.0 / 3.0))


def test_keyframe_buffer_admits_only_motion_above_threshold() -> None:
    """The first frame is retained and later frames must move more than 0.1."""

    buffer: KeyframeBuffer = KeyframeBuffer()
    image_3hw: Float[Tensor, "3 384 512"] = torch.zeros((3, 384, 512), dtype=torch.float32)

    assert buffer.add(_keyframe(0, 0.0, image_3hw))
    assert not buffer.add(_keyframe(1, 0.1, image_3hw))
    assert buffer.add(_keyframe(2, 0.1001, image_3hw))
    assert len(buffer) == 2


def test_keyframe_buffer_selects_seven_best_sources_in_score_order() -> None:
    """Selection returns the seven baselines nearest the 0.15 m optimum."""

    buffer: KeyframeBuffer = KeyframeBuffer()
    image_3hw: Float[Tensor, "3 384 512"] = torch.zeros((3, 384, 512), dtype=torch.float32)
    frame_index: int
    for frame_index in range(8):
        assert buffer.add(_keyframe(frame_index, (frame_index + 1) * 0.15, image_3hw))

    selected_indices: list[int] = [keyframe.frame_index for keyframe in buffer.select_sources(_translation_pose(0.0))]

    assert selected_indices == list(range(7))
