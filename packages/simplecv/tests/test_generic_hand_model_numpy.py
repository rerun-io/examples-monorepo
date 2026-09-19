"""Parity tests between the torch and NumPy hand model implementations."""

from __future__ import annotations

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Float32
from umetrack_models import synthetic_hand_model, to_torch

torch = pytest.importorskip("torch")

from simplecv.umetrack_temp import generic_hand_model_numpy as np_mod  # noqa: E402
from simplecv.umetrack_temp import generic_hand_model_torch as torch_mod  # noqa: E402

RNG = np.random.default_rng(1234)


def test_so3_exp_map_matches_torch() -> None:
    log_rot: Float32[np.ndarray, "12 3"] = RNG.normal(size=(12, 3)).astype(np.float32)

    numpy_result = np_mod.so3_exp_map(log_rot)
    torch_result = torch_mod.so3_exp_map(torch.from_numpy(log_rot)).numpy()

    np.testing.assert_allclose(numpy_result, torch_result, rtol=1e-5, atol=1e-5)


def test_skin_landmarks_matches_torch() -> None:
    hand_model: np_mod.HandModelNumpy = synthetic_hand_model()
    torch_hand_model: torch_mod.HandModelTorch = to_torch(hand_model)


    joint_angles: Float32[np.ndarray, "n_joints=22"] = RNG.normal(size=(22,)).astype(np.float32)
    wrist_transforms: Float32[np.ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    wrist_transforms[:3, 3] = RNG.normal(size=(3,)).astype(np.float32)

    numpy_landmarks = np_mod.skin_landmarks(hand_model, joint_angles, wrist_transforms)
    torch_landmarks = torch_mod.skin_landmarks(
        torch_hand_model,
        torch.from_numpy(joint_angles),
        torch.from_numpy(wrist_transforms),
    ).numpy()

    np.testing.assert_allclose(numpy_landmarks, torch_landmarks, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("hand_idx", (np_mod.LEFT_HAND_INDEX, np_mod.RIGHT_HAND_INDEX))
def test_landmarks_from_hand_pose_matches_torch(
    hand_idx: int
) -> None:
    hand_model: np_mod.HandModelNumpy = synthetic_hand_model()
    torch_hand_model: torch_mod.HandModelTorch = to_torch(hand_model)


    joint_angles: Float32[np.ndarray, "n_joints=22"] = RNG.normal(size=(22,)).astype(np.float32)
    wrist_xform: Float32[np.ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    wrist_xform[:3, 3] = RNG.normal(size=(3,)).astype(np.float32)

    numpy_pose = np_mod.SingleHandPose(joint_angles=joint_angles, wrist_xform=wrist_xform)
    torch_pose = torch_mod.SingleHandPose(joint_angles=joint_angles, wrist_xform=wrist_xform)

    numpy_landmarks = np_mod.landmarks_from_hand_pose(hand_model, numpy_pose, hand_idx)
    torch_landmarks = torch_mod.landmarks_from_hand_pose(torch_hand_model, torch_pose, hand_idx)

    np.testing.assert_allclose(numpy_landmarks, torch_landmarks, rtol=1e-5, atol=1e-5)


@st.composite
def _so3_input(draw: st.DrawFn) -> Float32[np.ndarray, "batch 3"]:
    batch: int = draw(st.integers(min_value=1, max_value=16))
    elements = st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False)
    log_rot: Float32[np.ndarray, "batch 3"] = draw(hnp.arrays(dtype=np.float32, shape=(batch, 3), elements=elements))
    return log_rot


@st.composite
def _joint_angles_and_wrist(
    draw: st.DrawFn,
) -> tuple[Float32[np.ndarray, "n_joints=22"], Float32[np.ndarray, "4 4"]]:
    angle_elements = st.floats(min_value=-2.0 * np.pi, max_value=2.0 * np.pi, allow_nan=False, allow_infinity=False)
    joint_angles: Float32[np.ndarray, "n_joints=22"] = draw(
        hnp.arrays(dtype=np.float32, shape=(np_mod.NUM_JOINTS_PER_HAND,), elements=angle_elements)
    )

    translation_elements = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    translation: Float32[np.ndarray, "3"] = draw(
        hnp.arrays(dtype=np.float32, shape=(3,), elements=translation_elements)
    )

    wrist_template: Float32[np.ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    wrist_template[:3, 3] = translation

    return joint_angles, wrist_template


@given(log_rot=_so3_input())
@settings(max_examples=64)
def test_so3_exp_map_matches_torch_hypothesis(log_rot: Float32[np.ndarray, "batch 3"]) -> None:
    numpy_result: Float32[np.ndarray, "batch 3 3"] = np_mod.so3_exp_map(log_rot)
    torch_log_rot: Float32[torch.Tensor, "batch 3"] = torch.from_numpy(log_rot)
    torch_result: Float32[np.ndarray, "batch 3 3"] = torch_mod.so3_exp_map(torch_log_rot).numpy()
    np.testing.assert_allclose(numpy_result, torch_result, rtol=1e-5, atol=1e-5)


@given(sample=_joint_angles_and_wrist())
@settings(max_examples=32)
def test_skin_landmarks_matches_torch_hypothesis(
    sample: tuple[Float32[np.ndarray, "n_joints=22"], Float32[np.ndarray, "4 4"]]
) -> None:
    hand_model: np_mod.HandModelNumpy = synthetic_hand_model()
    torch_hand_model: torch_mod.HandModelTorch = to_torch(hand_model)

    joint_angles, wrist_transforms = sample
    joint_angles_arr: Float32[np.ndarray, "n_joints=22"] = joint_angles
    wrist_transforms_arr: Float32[np.ndarray, "4 4"] = wrist_transforms

    torch_joint_angles: Float32[torch.Tensor, "n_joints=22"] = torch.from_numpy(joint_angles_arr)
    torch_wrist: Float32[torch.Tensor, "4 4"] = torch.from_numpy(wrist_transforms_arr)

    numpy_landmarks: Float32[np.ndarray, "... num_landmarks 3"] = np_mod.skin_landmarks(
        hand_model, joint_angles_arr, wrist_transforms_arr
    )
    torch_landmarks: Float32[np.ndarray, "... num_landmarks 3"] = torch_mod.skin_landmarks(
        torch_hand_model, torch_joint_angles, torch_wrist
    ).numpy()

    np.testing.assert_allclose(numpy_landmarks, torch_landmarks, rtol=1e-5, atol=1e-5)
