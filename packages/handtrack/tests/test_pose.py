import numpy as np
import torch
from serde.json import from_json
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy, wrist_for_hand
from simplecv.umetrack_temp.generic_hand_model_numpy import skin_landmarks as skin_landmarks_numpy
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch

from handtrack.hand.pose import GENERIC_HAND_MODEL, HandPose, Side, extrapolate, generic_hand_model, landmarks


def _rotation_z(angle: float) -> torch.Tensor:
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def test_extrapolate_is_constant_velocity() -> None:
    before: HandPose = HandPose(_rotation_z(0.1), torch.tensor([0.0, 0.0, 0.3]), torch.zeros(22))
    previous: HandPose = HandPose(_rotation_z(0.3), torch.tensor([0.01, 0.0, 0.3]), torch.full((22,), 0.2))
    guess: HandPose = extrapolate(previous, before)
    torch.testing.assert_close(guess.rotation, _rotation_z(0.5))
    torch.testing.assert_close(guess.translation, torch.tensor([0.02, 0.0, 0.3]))
    torch.testing.assert_close(guess.joint_angles, torch.full((22,), 0.4))


def test_landmarks_match_simplecv_numpy_skinning_for_both_hands() -> None:
    model: HandModelTorch = generic_hand_model()
    numpy_model: HandModelNumpy = from_json(HandModelNumpy, GENERIC_HAND_MODEL.read_text())
    angles: torch.Tensor = torch.linspace(-0.2, 0.6, 22)
    pose: HandPose = HandPose(_rotation_z(0.4), torch.tensor([0.05, -0.02, 0.35]), angles)
    for side in Side:
        wrist: np.ndarray = np.eye(4, dtype=np.float32)
        wrist[:3, :3] = pose.rotation.numpy()
        wrist[:3, 3] = pose.translation.numpy() * 1000.0
        expected: np.ndarray = skin_landmarks_numpy(numpy_model, angles.numpy(), wrist_for_hand(wrist, int(side))) / 1000.0
        np.testing.assert_allclose(landmarks(model, pose, side).numpy(), expected, atol=1e-6)
    assert landmarks(model, pose, Side.LEFT).shape == (21, 3)
