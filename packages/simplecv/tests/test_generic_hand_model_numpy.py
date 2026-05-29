"""Parity tests between the torch and NumPy hand model implementations."""

from __future__ import annotations

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Float32

torch = pytest.importorskip("torch")

from simplecv.umetrack_temp import generic_hand_model_numpy as np_mod  # noqa: E402
from simplecv.umetrack_temp import generic_hand_model_torch as torch_mod  # noqa: E402

RNG = np.random.default_rng(1234)


def _random_hand_model() -> tuple[np_mod.HandModelNumpy, torch_mod.HandModelTorch]:
    """Generate matching random hand models for both backends."""

    num_joint_frames = np_mod.NUM_JOINT_FRAMES

    joint_rotation_axes: Float32[np.ndarray, "22 3"] = RNG.normal(size=(22, 3)).astype(np.float32)
    joint_rest_positions: Float32[np.ndarray, "22 3"] = RNG.normal(size=(22, 3)).astype(np.float32)
    joint_frame_index = RNG.integers(low=0, high=num_joint_frames, size=(22,), dtype=np.int64)
    joint_parent = RNG.integers(low=-1, high=22, size=(22,), dtype=np.int64)
    joint_first_child = RNG.integers(low=-1, high=22, size=(22,), dtype=np.int64)
    joint_next_sibling = RNG.integers(low=-1, high=22, size=(22,), dtype=np.int64)
    landmark_rest_positions: Float32[np.ndarray, "21 3"] = RNG.normal(size=(21, 3)).astype(np.float32)
    landmark_rest_bone_weights: Float32[np.ndarray, "21 3"] = RNG.random(size=(21, 3), dtype=np.float32)
    landmark_rest_bone_indices = RNG.integers(low=0, high=num_joint_frames, size=(21, 3), dtype=np.int64)
    hand_scale: Float32[np.ndarray, ""] = np.array(1.0, dtype=np.float32)
    mesh_vertices: Float32[np.ndarray, "10 3"] = RNG.normal(size=(10, 3)).astype(np.float32)
    mesh_triangles = RNG.integers(low=0, high=10, size=(5, 3), dtype=np.int64)
    dense_bone_weights: Float32[np.ndarray, "10 num_joint_frames"] = RNG.random(
        size=(10, num_joint_frames), dtype=np.float32
    )
    joint_limits: Float32[np.ndarray, "22 2"] = RNG.normal(size=(22, 2)).astype(np.float32)

    numpy_model = np_mod.HandModelNumpy(
        joint_rotation_axes=joint_rotation_axes,
        joint_rest_positions=joint_rest_positions,
        joint_frame_index=joint_frame_index,
        joint_parent=joint_parent,
        joint_first_child=joint_first_child,
        joint_next_sibling=joint_next_sibling,
        landmark_rest_positions=landmark_rest_positions,
        landmark_rest_bone_weights=landmark_rest_bone_weights,
        landmark_rest_bone_indices=landmark_rest_bone_indices,
        hand_scale=hand_scale,
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        dense_bone_weights=dense_bone_weights,
        joint_limits=joint_limits,
    )

    torch_model = torch_mod.HandModelTorch(
        joint_rotation_axes=torch.from_numpy(joint_rotation_axes),
        joint_rest_positions=torch.from_numpy(joint_rest_positions),
        joint_frame_index=torch.from_numpy(joint_frame_index),
        joint_parent=torch.from_numpy(joint_parent),
        joint_first_child=torch.from_numpy(joint_first_child),
        joint_next_sibling=torch.from_numpy(joint_next_sibling),
        landmark_rest_positions=torch.from_numpy(landmark_rest_positions),
        landmark_rest_bone_weights=torch.from_numpy(landmark_rest_bone_weights),
        landmark_rest_bone_indices=torch.from_numpy(landmark_rest_bone_indices),
        hand_scale=torch.from_numpy(hand_scale),
        mesh_vertices=torch.from_numpy(mesh_vertices),
        mesh_triangles=torch.from_numpy(mesh_triangles),
        dense_bone_weights=torch.from_numpy(dense_bone_weights),
        joint_limits=torch.from_numpy(joint_limits),
    )

    return numpy_model, torch_model


HAND_MODEL_NUMPY_PROP, HAND_MODEL_TORCH_PROP = _random_hand_model()


def test_so3_exp_map_matches_torch() -> None:
    log_rot: Float32[np.ndarray, "12 3"] = RNG.normal(size=(12, 3)).astype(np.float32)

    numpy_result = np_mod.so3_exp_map(log_rot)
    torch_result = torch_mod.so3_exp_map(torch.from_numpy(log_rot)).numpy()

    np.testing.assert_allclose(numpy_result, torch_result, rtol=1e-5, atol=1e-5)


def test_skin_landmarks_matches_torch() -> None:
    numpy_model, torch_model = _random_hand_model()

    joint_angles: Float32[np.ndarray, "n_joints=22"] = RNG.normal(size=(22,)).astype(np.float32)
    wrist_transforms: Float32[np.ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    wrist_transforms[:3, 3] = RNG.normal(size=(3,)).astype(np.float32)

    numpy_landmarks = np_mod.skin_landmarks(numpy_model, joint_angles, wrist_transforms)
    torch_landmarks = torch_mod.skin_landmarks(
        torch_model,
        torch.from_numpy(joint_angles),
        torch.from_numpy(wrist_transforms),
    ).numpy()

    np.testing.assert_allclose(numpy_landmarks, torch_landmarks, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("hand_idx", (np_mod.LEFT_HAND_INDEX, np_mod.RIGHT_HAND_INDEX))
def test_landmarks_from_hand_pose_matches_torch(hand_idx: int) -> None:
    numpy_model, torch_model = _random_hand_model()

    joint_angles: Float32[np.ndarray, "n_joints=22"] = RNG.normal(size=(22,)).astype(np.float32)
    wrist_xform: Float32[np.ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    wrist_xform[:3, 3] = RNG.normal(size=(3,)).astype(np.float32)

    numpy_pose = np_mod.SingleHandPose(joint_angles=joint_angles, wrist_xform=wrist_xform)
    torch_pose = torch_mod.SingleHandPose(joint_angles=joint_angles, wrist_xform=wrist_xform)

    numpy_landmarks = np_mod.landmarks_from_hand_pose(numpy_model, numpy_pose, hand_idx)
    torch_landmarks = torch_mod.landmarks_from_hand_pose(torch_model, torch_pose, hand_idx)

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
    joint_angles, wrist_transforms = sample
    joint_angles_arr: Float32[np.ndarray, "n_joints=22"] = joint_angles
    wrist_transforms_arr: Float32[np.ndarray, "4 4"] = wrist_transforms

    torch_joint_angles: Float32[torch.Tensor, "n_joints=22"] = torch.from_numpy(joint_angles_arr)
    torch_wrist: Float32[torch.Tensor, "4 4"] = torch.from_numpy(wrist_transforms_arr)

    numpy_landmarks: Float32[np.ndarray, "... num_landmarks 3"] = np_mod.skin_landmarks(
        HAND_MODEL_NUMPY_PROP, joint_angles_arr, wrist_transforms_arr
    )
    torch_landmarks: Float32[np.ndarray, "... num_landmarks 3"] = torch_mod.skin_landmarks(
        HAND_MODEL_TORCH_PROP, torch_joint_angles, torch_wrist
    ).numpy()

    np.testing.assert_allclose(numpy_landmarks, torch_landmarks, rtol=1e-5, atol=1e-5)
