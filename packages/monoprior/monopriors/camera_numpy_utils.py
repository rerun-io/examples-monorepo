"""NumPy equivalents of functions in `camera_utils`.

The implementations mirror the PyTorch versions for numerical parity while
operating purely on `numpy.ndarray` inputs/outputs. All intermediate variables
are annotated with jaxtyping shapes/dtypes for consistency with the rest of
the codebase. See `camera_utils.py` for original reference.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple, TypeAlias

import numpy as np
from jaxtyping import Float
from numpy import ndarray


def _normalize(v: Float[ndarray, "3"]) -> Float[ndarray, "3"]:
    norm: float = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("Zero-length vector cannot be normalized")
    return v / norm


def rotation_matrix_between(a: Float[ndarray, "3"], b: Float[ndarray, "3"]) -> Float[ndarray, "3 3"]:
    """Compute rotation matrix that rotates vector ``a`` onto vector ``b``.

    Args:
        a: Source vector (shape (3,)).
        b: Target vector (shape (3,)).
    Returns:
        3x3 rotation matrix.
    """
    a_n: Float[ndarray, "3"] = _normalize(a.astype(np.float64))
    b_n: Float[ndarray, "3"] = _normalize(b.astype(np.float64))

    v: Float[ndarray, "3"] = np.cross(a_n, b_n)  # axis of rotation

    eps: float = 1e-6
    if float(np.sum(np.abs(v))) < eps:
        # Vectors are parallel or anti-parallel; pick an orthogonal axis.
        x: Float[ndarray, "3"] = np.array([1.0, 0.0, 0.0]) if abs(a_n[0]) < eps else np.array([0.0, 1.0, 0.0])
        v = np.cross(a_n, x)

    v_n: Float[ndarray, "3"] = _normalize(v)
    vx: Float[ndarray, "3 3"] = np.array(
        [
            [0.0, -v_n[2], v_n[1]],
            [v_n[2], 0.0, -v_n[0]],
            [-v_n[1], v_n[0], 0.0],
        ],
        dtype=np.float64,
    )
    theta: float = float(np.arccos(np.clip(np.dot(a_n, b_n), -1.0, 1.0)))

    eye_mat: Float[ndarray, "3 3"] = np.eye(3, dtype=np.float64)
    rot: Float[ndarray, "3 3"] = eye_mat + math.sin(theta) * vx + (1.0 - math.cos(theta)) * (vx @ vx)
    return rot.astype(np.float64)


def focus_of_attention(poses: Float[ndarray, "*n 4 4"], initial_focus: Float[ndarray, "3"]) -> Float[ndarray, "3"]:
    """Compute focus of attention (closest point to optical axes) for active cameras.

    Mirrors the iterative pruning strategy from the torch implementation.
    Only cameras that see the current focus in front of them are retained.
    """
    poses_f64: Float[ndarray, "*n 4 4"] = poses.astype(np.float64)
    active_directions: Float[ndarray, "*n 3 1"] = -poses_f64[:, :3, 2:3]
    active_origins: Float[ndarray, "*n 3 1"] = poses_f64[:, :3, 3:4]
    focus_pt: Float[ndarray, "3"] = initial_focus.astype(np.float64)

    dots: Float[ndarray, "*n"] = np.sum(
        active_directions.squeeze(-1) * (focus_pt[None, :] - active_origins.squeeze(-1)), axis=-1
    )
    active_mask: ndarray = dots > 0
    done: bool = False

    while int(np.sum(active_mask.astype(int))) > 1 and not done:
        active_directions = active_directions[active_mask]
        active_origins = active_origins[active_mask]
        # m_i = I - d d^T for each direction
        d: Float[ndarray, "m 3 1"] = active_directions
        I3: Float[ndarray, "3 3"] = np.eye(3)
        m: Float[ndarray, "m 3 3"] = I3[None, :, :] - d @ np.transpose(d, (0, 2, 1))
        mt_m: Float[ndarray, "m 3 3"] = np.transpose(m, (0, 2, 1)) @ m
        mt_m_mean: Float[ndarray, "3 3"] = np.mean(mt_m, axis=0)
        focus_pt = np.linalg.inv(mt_m_mean) @ (np.mean(mt_m @ active_origins, axis=0)[:, 0])

        dots = np.sum(active_directions.squeeze(-1) * (focus_pt[None, :] - active_origins.squeeze(-1)), axis=-1)
        new_active_mask: ndarray = dots > 0
        if new_active_mask.all():
            done = True
        active_mask = new_active_mask

    return focus_pt.astype(np.float64)


OrientedPoses: TypeAlias = Float[ndarray, "*n 3 4"]
Transform34: TypeAlias = Float[ndarray, "3 4"]


class OrientResults(NamedTuple):
    oriented_poses: OrientedPoses
    transform: Transform34


def auto_orient_and_center_poses(
    world_T_cam_gl: Float[ndarray, "*n 4 4"],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> OrientResults:
    """Orient and center camera poses (NumPy version).

    See `camera_utils.auto_orient_and_center_poses` for detailed documentation.
    """
    poses_f64: Float[ndarray, "*n 4 4"] = world_T_cam_gl.astype(np.float64)
    origins: Float[ndarray, "*n 3"] = poses_f64[..., :3, 3]
    mean_origin: Float[ndarray, "3"] = np.mean(origins, axis=0)
    translation_diff: Float[ndarray, "*n 3"] = origins - mean_origin

    if center_method == "poses":
        translation: Float[ndarray, "3"] = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses_f64, mean_origin)
    elif center_method == "none":
        translation = np.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        cov: Float[ndarray, "3 3"] = translation_diff.T @ translation_diff
        eigvals_eigh, eigvec = np.linalg.eigh(cov)
        eigvec = np.flip(eigvec, axis=-1)  # descending order (eigvals unused)
        if np.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]
        transform: Float[ndarray, "3 4"] = np.concatenate([eigvec, (eigvec @ -translation[..., None])], axis=-1)
        oriented_poses: Float[ndarray, "*n 3 4"] = transform @ poses_f64
        if np.mean(oriented_poses, axis=0)[2, 1] < 0:
            oriented_poses[:, 1:3, :] = -oriented_poses[:, 1:3, :]
            transform[1:3, :] = -transform[1:3, :]
    elif method in ("up", "vertical"):
        up: Float[ndarray, "3"] = np.mean(poses_f64[:, :3, 1], axis=0)
        up = up / np.linalg.norm(up)
        if method == "vertical":
            x_axis_matrix: Float[ndarray, "*n 3"] = poses_f64[:, :3, 0]
            U, S, Vh = np.linalg.svd(x_axis_matrix, full_matrices=False)
            if S.shape[0] > 1 and S[1] > 0.17 * math.sqrt(poses_f64.shape[0]):
                up_vertical: Float[ndarray, "3"] = Vh[2, :]
                up = up_vertical if float(np.dot(up_vertical, up)) > 0 else -up_vertical
            elif S.shape[0] > 0:
                up = up - Vh[0, :] * float(np.dot(up, Vh[0, :]))
                up = up / np.linalg.norm(up)
        rotation: Float[ndarray, "3 3"] = rotation_matrix_between(up, np.array([0.0, 0.0, 1.0]))
        transform = np.concatenate([rotation, rotation @ -translation[..., None]], axis=-1)
        oriented_poses = transform @ poses_f64
    elif method == "none":
        transform4 = np.eye(4)
        transform4[:3, 3] = -translation
        transform = transform4[:3, :]
        oriented_poses = transform @ poses_f64
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return OrientResults(oriented_poses=oriented_poses, transform=transform[:3, :])


__all__ = [
    "rotation_matrix_between",
    "focus_of_attention",
    "auto_orient_and_center_poses",
    "OrientResults",
]
