import numpy as np
import torch
from serde.json import from_json
from simplecv.umetrack_temp.generic_hand_model_numpy import HandModelNumpy, wrist_for_hand
from simplecv.umetrack_temp.generic_hand_model_numpy import skin_landmarks as skin_landmarks_numpy
from simplecv.umetrack_temp.generic_hand_model_numpy import skin_mesh as skin_mesh_numpy
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch

from handtrack.hand.pose import GENERIC_HAND_MODEL, HandPose, Side, extrapolate, generic_hand_model, hand_model_from_profile, landmarks, mesh_vertices


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


def test_extrapolate_gain_damps_and_the_step_clamp_bounds_the_wrist() -> None:
    before: HandPose = HandPose(_rotation_z(0.1), torch.tensor([0.0, 0.0, 0.3]), torch.zeros(22))
    previous: HandPose = HandPose(_rotation_z(0.3), torch.tensor([0.02, 0.0, 0.3]), torch.full((22,), 0.2))
    half: HandPose = extrapolate(previous, before, gain=0.5)
    torch.testing.assert_close(half.rotation, _rotation_z(0.4), atol=2e-3, rtol=0.0)  # 0.1 rad of the 0.2 rad step, to first order
    torch.testing.assert_close(torch.linalg.det(half.rotation), torch.tensor(1.0))
    torch.testing.assert_close(half.rotation @ half.rotation.T, torch.eye(3), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(half.translation, torch.tensor([0.03, 0.0, 0.3]))
    torch.testing.assert_close(half.joint_angles, torch.full((22,), 0.3))
    torch.testing.assert_close(extrapolate(previous, before, gain=0.0).rotation, previous.rotation)
    clamped: HandPose = extrapolate(previous, before, max_step_m=0.005)
    torch.testing.assert_close(clamped.translation, torch.tensor([0.025, 0.0, 0.3]))
    torch.testing.assert_close(clamped.rotation, _rotation_z(0.5))  # the clamp bounds the wrist only


def test_landmarks_and_mesh_match_simplecv_numpy_skinning_for_both_hands() -> None:
    model: HandModelTorch = generic_hand_model()
    numpy_model: HandModelNumpy = from_json(HandModelNumpy, GENERIC_HAND_MODEL.read_text())
    angles: torch.Tensor = torch.linspace(-0.2, 0.6, 22)
    pose: HandPose = HandPose(_rotation_z(0.4), torch.tensor([0.05, -0.02, 0.35]), angles)
    wrist: np.ndarray = np.eye(4, dtype=np.float32)
    wrist[:3, :3] = pose.rotation.numpy()
    wrist[:3, 3] = pose.translation.numpy() * 1000.0
    for side in (Side.LEFT, Side.RIGHT):
        mirrored: np.ndarray = wrist_for_hand(wrist, int(side))
        np.testing.assert_allclose(landmarks(model, pose, side).numpy(), skin_landmarks_numpy(numpy_model, angles.numpy(), mirrored) / 1000.0, atol=1e-6)
        np.testing.assert_allclose(mesh_vertices(model, pose, side).numpy(), skin_mesh_numpy(numpy_model, angles.numpy(), mirrored) / 1000.0, atol=1e-6)


def test_profile_reader_accepts_bare_and_enveloped_models() -> None:
    bare: str = GENERIC_HAND_MODEL.read_text()
    enveloped: tuple[str, ...] = ('{"hand_model":' + bare + "}", '\n\n   {' + " " * 80 + '"hand_model":' + bare + "}", '{"meta": {"name": "subject"}, "hand_model":' + bare + "}")
    for text in (bare, *enveloped):
        torch.testing.assert_close(hand_model_from_profile(text).landmark_rest_positions, generic_hand_model().landmark_rest_positions)
