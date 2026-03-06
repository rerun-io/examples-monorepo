from __future__ import annotations

import math

from beartype.roar import BeartypeCallHintParamViolation
import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st
from jaxtyping import Float
from numpy import ndarray

from monopriors.camera_utils import (
    auto_orient_and_center_poses as torch_auto_orient_and_center_poses,
    focus_of_attention as torch_focus_of_attention,
)
from monopriors.camera_numpy_utils import (
    auto_orient_and_center_poses as np_auto_orient_and_center_poses,
    focus_of_attention as np_focus_of_attention,
    rotation_matrix_between as np_rotation_matrix_between,
)


# ------------------------------ Helpers ------------------------------ #


def random_rotation_matrix() -> Float[ndarray, "3 3"]:
    axis: Float[ndarray, "3"] = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle: float = float(np.random.rand() * 2 * math.pi)
    K: Float[ndarray, "3 3"] = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    eye_mat: Float[ndarray, "3 3"] = np.eye(3)
    R: Float[ndarray, "3 3"] = eye_mat + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R.astype(np.float64)


def random_poses(n: int) -> Float[ndarray, "n 4 4"]:
    mats: list[Float[ndarray, "4 4"]] = []
    for _ in range(n):
        R = random_rotation_matrix()
        t: Float[ndarray, "3"] = np.random.randn(3) * 0.5
        P: Float[ndarray, "4 4"] = np.eye(4)
        P[:3, :3] = R
        P[:3, 3] = t
        mats.append(P)
    return np.stack(mats, axis=0)


rot_method_strat = st.sampled_from(["pca", "up", "vertical", "none"])  # type: ignore
center_method_strat = st.sampled_from(["poses", "focus", "none"])  # type: ignore


# ------------------------------ Tests ------------------------------ #


def test_rotation_matrix_parallel_and_antiparallel():
    a: Float[ndarray, "3"] = np.array([1.0, 0.0, 0.0])
    b: Float[ndarray, "3"] = np.array([1.0, 0.0, 0.0])
    R_same = np_rotation_matrix_between(a, b)
    assert np.allclose(R_same @ a, b, atol=1e-6)

    b2: Float[ndarray, "3"] = np.array([-1.0, 0.0, 0.0])
    R_opposite = np_rotation_matrix_between(a, b2)
    a_rot: Float[ndarray, "3"] = R_opposite @ a
    assert np.allclose(a_rot / np.linalg.norm(a_rot), b2 / np.linalg.norm(b2), atol=1e-6)
    assert np.isclose(np.linalg.det(R_opposite), 1.0, atol=1e-6)


@given(
    a=st.lists(st.floats(-10, 10), min_size=3, max_size=3),
    b=st.lists(st.floats(-10, 10), min_size=3, max_size=3),
)
@settings(max_examples=50)
def test_rotation_matrix_properties(a: list[float], b: list[float]):
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    if np.linalg.norm(a_arr) < 1e-8 or np.linalg.norm(b_arr) < 1e-8:
        return
    R_np = np_rotation_matrix_between(a_arr, b_arr)
    # Orthonormal
    should_I = R_np.T @ R_np
    assert np.allclose(should_I, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R_np), 1.0, atol=1e-5)
    # Direction mapping
    a_dir = a_arr / np.linalg.norm(a_arr)
    b_dir = b_arr / np.linalg.norm(b_arr)
    mapped = R_np @ a_dir
    # Allow small sign ambiguity when anti-parallel
    if np.allclose(a_dir, -b_dir, atol=1e-5):
        assert np.allclose(mapped, b_dir, atol=1e-5) or np.allclose(mapped, -b_dir, atol=1e-5)
    else:
        assert np.allclose(mapped, b_dir, atol=1e-5)


@given(st.integers(min_value=2, max_value=6))
@settings(max_examples=15)
def test_focus_of_attention_parity(n: int):
    poses_np = random_poses(n)
    focus_init: Float[ndarray, "3"] = np.mean(poses_np[:, :3, 3], axis=0)
    poses_torch = torch.tensor(poses_np, dtype=torch.float64)
    torch_focus = torch_focus_of_attention(poses_torch, torch.tensor(focus_init))
    np_focus = np_focus_of_attention(poses_np, focus_init)
    assert np.allclose(np_focus, torch_focus.numpy(force=True), atol=5e-5, rtol=1e-5)


@given(
    n=st.integers(min_value=2, max_value=6),
    method=rot_method_strat,
    center=center_method_strat,
)
@settings(max_examples=40, deadline=None)
def test_auto_orient_and_center_parity(n: int, method: str, center: str):
    poses_np = random_poses(n)
    # Vertical method needs at least 3 poses for stable SVD parity (torch version may be degenerate with n<3)
    if method == "vertical" and n < 3:
        pytest.skip("Skip vertical orientation parity for n < 3 (degenerate SVD case)")

    poses_torch = torch.tensor(poses_np, dtype=torch.float32)
    torch_res = torch_auto_orient_and_center_poses(poses_torch, method=method, center_method=center)
    np_res = np_auto_orient_and_center_poses(poses_np, method=method, center_method=center)

    torch_transform = torch_res.transform.numpy(force=True)
    torch_oriented = torch_res.oriented_poses.numpy(force=True)

    # Allow row-wise sign flips (common eigen/SVD ambiguity). Helper:
    def rows_equivalent(a: np.ndarray, b: np.ndarray, atol_rot=8e-5, atol_trans=1e-4) -> bool:
        if a.shape != b.shape:
            return False
        # a,b : (3,4) or (3,4) slices
        for i in range(3):
            row_a = a[i]
            row_b = b[i]
            if np.allclose(row_a, row_b, atol=atol_trans):
                continue
            if np.allclose(row_a, -row_b, atol=atol_trans):
                continue
            return False
        return True

    assert rows_equivalent(np_res.transform, torch_transform), "Transform rows differ beyond sign ambiguity"

    # Check each pose (N,3,4)
    for i in range(np_res.oriented_poses.shape[0]):
        assert rows_equivalent(np_res.oriented_poses[i], torch_oriented[i]), f"Pose {i} mismatch beyond sign"


def test_invalid_method():
    poses_np = random_poses(3)
    with pytest.raises(BeartypeCallHintParamViolation):
        np_auto_orient_and_center_poses(poses_np, method="bad")  # type: ignore


def test_single_camera_focus_graceful():
    poses_np = random_poses(1)
    init_focus = poses_np[0, :3, 3]
    # Should just return something finite without error
    focus = np_focus_of_attention(poses_np, init_focus)
    assert np.isfinite(focus).all()
