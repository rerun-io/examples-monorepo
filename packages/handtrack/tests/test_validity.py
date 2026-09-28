import pytest
import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from handtrack.labels.validity import MIN_VISIBLE_KEYPOINTS, SHOW3D_CONFIDENCE_THRESHOLD, HandLabel, classify_visibility, show3d_hands, umetrack_hands


def test_visibility_thresholds() -> None:
    assert MIN_VISIBLE_KEYPOINTS == 17
    assert classify_visibility(torch.tensor([[0, 1, 16, 17, 21]])).tolist() == [[HandLabel.ABSENT, HandLabel.PARTIAL, HandLabel.PARTIAL, HandLabel.PRESENT, HandLabel.PRESENT]]


def test_umetrack_binary_labels_and_headset() -> None:
    result: tuple[Bool[Tensor, 'f'], Bool[Tensor, 'f 2']] = umetrack_hands(torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), torch.tensor([True, False, True]))
    assert result[0].tolist() == [True, False, True]
    assert result[1].tolist() == [[False, True], [True, False], [True, True]]


@pytest.mark.parametrize('confidence', [0.1, -1.0, 2.0, float('nan')])
def test_umetrack_rejects_nonbinary(confidence: float) -> None:
    with pytest.raises(ValueError, match='binary'):
        umetrack_hands(torch.tensor([[confidence, 1.0]]), torch.tensor([True]))


def test_show3d_per_camera_missing_pose_and_headset_rules() -> None:
    assert SHOW3D_CONFIDENCE_THRESHOLD == 0.1
    confidence: Float32[Tensor, '5 2'] = torch.tensor([[0.9, 0.1], [0.0, 0.1], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]])
    has_pose: Bool[Tensor, '5 2'] = torch.tensor([[True, True], [True, True], [False, True], [True, True], [True, True]])
    inside: Int64[Tensor, '5 2 2'] = torch.tensor([[[0, 0], [21, 1]], [[0, 0], [1, 0]], [[0, 21], [0, 0]], [[21, 21], [0, 0]], [[21, 21], [0, 0]]])
    result: tuple[Bool[Tensor, '5 2'], Bool[Tensor, '5 2 2']] = show3d_hands(confidence, has_pose, inside, torch.tensor([True, True, True, False, True]))
    assert result[0].tolist() == [[True, False], [True, False], [False, False], [False, False], [True, True]]
    assert result[1].tolist() == [[[True, False]] * 2, [[False, False]] * 2, [[False, True]] * 2, [[True, True]] * 2, [[True, True]] * 2]
