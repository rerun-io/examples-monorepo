"""Shared UmeTrack models skin whole pose batches without changing frame results."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray
from serde import serde
from serde.json import from_json
from umetrack_models import synthetic_hand_model, to_torch

torch = pytest.importorskip("torch", reason="UmeTrack parity checks require torch")

from simplecv.umetrack_temp import generic_hand_model_numpy as numpy_model  # noqa: E402
from simplecv.umetrack_temp import generic_hand_model_torch as torch_model  # noqa: E402

Backend: TypeAlias = Literal["numpy", "torch"]


def _random_poses(
    seed: int, batch_shape: tuple[int, ...]
) -> tuple[Float32[ndarray, "*batch 22"], Float32[ndarray, "*batch 4 4"]]:
    rng: np.random.Generator = np.random.default_rng(seed)
    angles: Float32[ndarray, "*batch 22"] = rng.uniform(-0.8, 0.8, (*batch_shape, 22)).astype(np.float32)
    wrists: Float32[ndarray, "*batch 4 4"] = np.broadcast_to(np.eye(4, dtype=np.float32), (*batch_shape, 4, 4)).copy()
    wrists[..., :3, :3] = numpy_model.so3_exp_map(rng.normal(size=(math.prod(batch_shape), 3)).astype(np.float32)).reshape(*batch_shape, 3, 3)
    wrists[..., :3, 3] = rng.normal(size=(*batch_shape, 3)).astype(np.float32)
    return angles, wrists


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)])
@pytest.mark.parametrize("numpy_fn,torch_fn", [
    (numpy_model.skin_landmarks, torch_model.skin_landmarks),
    (numpy_model.skin_mesh, torch_model.skin_mesh),
])
def test_batch_matches_frames(
    batch_shape: tuple[int, ...],
    numpy_fn: Callable[
        [numpy_model.HandModelNumpy, Float32[ndarray, "*batch 22"], Float32[ndarray, "*batch 4 4"]],
        Float32[ndarray, "*batch points 3"],
    ],
    torch_fn: Callable[
        [torch_model.HandModelTorch, Float32[torch.Tensor, "*batch 22"], Float32[torch.Tensor, "*batch 4 4"]],
        Float32[torch.Tensor, "*batch points 3"],
    ],
) -> None:
    hand_model: numpy_model.HandModelNumpy = synthetic_hand_model()
    angles, wrists = _random_poses(7, batch_shape)
    expected: Float32[ndarray, "*batch points 3"] = np.stack(
        [numpy_fn(hand_model, angle, wrist) for angle, wrist in zip(angles.reshape(-1, 22), wrists.reshape(-1, 4, 4), strict=True)]
    ).reshape(*batch_shape, -1, 3)
    for actual in (
        numpy_fn(hand_model, angles, wrists),
        torch_fn(to_torch(hand_model), torch.from_numpy(angles), torch.from_numpy(wrists)).numpy(),
    ):
        assert actual.shape == expected.shape
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_identity_pose_preserves_rest_geometry(backend: Backend) -> None:
    hand_model: numpy_model.HandModelNumpy = synthetic_hand_model()
    torch_hand_model: torch_model.HandModelTorch = to_torch(hand_model)
    angles: Float32[ndarray, "22"] = np.zeros(22, dtype=np.float32)
    wrist: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    landmarks: Float32[ndarray, "21 3"]
    mesh: Float32[ndarray, "31 3"]
    if backend == "numpy":
        landmarks = numpy_model.skin_landmarks(hand_model, angles, wrist)
        mesh = numpy_model.skin_mesh(hand_model, angles, wrist)
    else:
        landmarks = torch_model.skin_landmarks_np(torch_hand_model, angles, wrist)
        mesh = torch_model.skin_mesh(torch_hand_model, torch.from_numpy(angles), torch.from_numpy(wrist)).numpy()
    np.testing.assert_allclose(landmarks, hand_model.landmark_rest_positions, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(mesh, hand_model.mesh_vertices, atol=1e-7, rtol=0.0)


def test_right_hand_batch_matches_hand_pose() -> None:
    hand_model: numpy_model.HandModelNumpy = synthetic_hand_model()
    angles, wrists = _random_poses(11, (5,))
    original: Float32[ndarray, "5 4 4"] = wrists.copy()
    mirrored: Float32[ndarray, "5 4 4"] = numpy_model.wrist_for_hand(wrists, numpy_model.RIGHT_HAND_INDEX)
    np.testing.assert_array_equal(mirrored[..., :, 0], -original[..., :, 0])
    np.testing.assert_array_equal(mirrored[..., :, 1:], original[..., :, 1:])
    left: Float32[ndarray, "5 4 4"] = numpy_model.wrist_for_hand(wrists, numpy_model.LEFT_HAND_INDEX)
    np.testing.assert_array_equal(left, original)
    assert not np.shares_memory(left, wrists)
    expected: Float32[ndarray, "5 21 3"] = np.stack(
        [
            numpy_model.landmarks_from_hand_pose(hand_model, numpy_model.SingleHandPose(angle, wrist), numpy_model.RIGHT_HAND_INDEX)
            for angle, wrist in zip(angles, wrists, strict=True)
        ]
    )
    np.testing.assert_allclose(numpy_model.skin_landmarks(hand_model, angles, mirrored), expected, atol=1e-5, rtol=0.0)
    np.testing.assert_array_equal(wrists, original)


@serde
@dataclass(frozen=True, slots=True)
class ProfileRecord:
    """Subject profile from SHOW3D."""

    hand_model: numpy_model.HandModelNumpy
    """Rest geometry and skinning parameters in source units."""


@serde
@dataclass(frozen=True, slots=True)
class HandPoseRecord:
    """One source hand pose and its independent landmark reference."""

    confidence: float
    """Source pose confidence."""
    joint_angles: Float32[ndarray, "22"]
    """Joint angles in radians."""
    wrist_rotation: Float32[ndarray, "3 3"]
    """World-from-wrist rotation."""
    wrist_translation: Float32[ndarray, "3"]
    """World-from-wrist translation in millimetres."""
    landmarks_3d_mm: Float32[ndarray, "21 3"]
    """Reference world landmarks in millimetres."""


@serde
@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Hand poses supplied for one source frame."""

    hand_poses: dict[str, HandPoseRecord]
    """Poses keyed by the source hand index."""


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("hand_idx", [0, 1])
def test_show3d_frame_matches_shipped_landmarks(backend: Backend, hand_idx: int) -> None:
    """Compare SHOW3D SPI102 frame 180 with the shipped millimetre landmarks."""
    fixtures: Path = Path(__file__).parent / "fixtures" / "umetrack"
    profile: ProfileRecord = from_json(ProfileRecord, (fixtures / "show3d_SPI102_profile_umetrack.json").read_text())
    frame: FrameRecord = from_json(FrameRecord, (fixtures / "show3d_SPI102_keyboard_frame180.json").read_text())
    pose: HandPoseRecord = frame.hand_poses[str(hand_idx)]
    wrist: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    wrist[:3, :3] = pose.wrist_rotation
    wrist[:3, 3] = pose.wrist_translation
    mirrored: Float32[ndarray, "4 4"] = numpy_model.wrist_for_hand(wrist, hand_idx)
    actual: Float32[ndarray, "21 3"]
    if backend == "numpy":
        actual = numpy_model.skin_landmarks(profile.hand_model, pose.joint_angles, mirrored)
    else:
        actual = torch_model.skin_landmarks_np(to_torch(profile.hand_model), pose.joint_angles, mirrored)
    assert float(np.max(np.abs(actual - pose.landmarks_3d_mm))) < 1e-3
