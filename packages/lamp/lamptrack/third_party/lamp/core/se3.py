# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny SE(3) helpers backed by 4x4 numpy float32 matrices."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_4x4_f32(any_se3: Any) -> np.ndarray:
    """Coerce an SE(3)-like value to a `(4, 4)` `np.float32` matrix."""
    raw: Any = any_se3
    if hasattr(any_se3, "to_matrix"):
        raw = any_se3.to_matrix()
    elif hasattr(any_se3, "matrix"):
        raw = any_se3.matrix()

    m: np.ndarray = np.asarray(raw, dtype=np.float32)  # pyright: ignore[reportUnknownArgumentType]
    if m.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = m
        return out
    if m.shape != (4, 4):
        raise ValueError(f"expected a 4x4 (or 3x4) SE(3) matrix; got shape {m.shape}")
    return m


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Right-multiply: `T_a @ T_b` (a then b applied in the matrix-product sense).

    With the `T_dest_src` naming convention used elsewhere in the codebase,
    `compose(T_a_b, T_b_c) -> T_a_c`.
    """
    return (a @ b).astype(np.float32, copy=False)


def invert(t: np.ndarray) -> np.ndarray:
    """Invert an SE(3) transform analytically (`R^T`, `-R^T t`).

    Faster + more numerically stable than `np.linalg.inv` for SE(3) matrices.
    """
    out = np.eye(4, dtype=np.float32)
    R = t[:3, :3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t[:3, 3]
    return out


def slerp_so3_batched(
    R_old: np.ndarray,
    R_new: np.ndarray,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """Per-joint SO3 slerp on a `(J, 3, 3)` batch."""
    if R_old.shape != R_new.shape:
        raise ValueError(
            f"R_old / R_new shapes must match; got {R_old.shape} vs {R_new.shape}"
        )
    if R_old.size == 0:
        # Defensive: allow callers to pass an empty rotation stack.
        return R_old.astype(np.float32, copy=False)
    if isinstance(alpha, np.ndarray):
        if alpha.shape != (R_old.shape[0],):
            raise ValueError(
                f"alpha array shape must be ({R_old.shape[0]},); got {alpha.shape}"
            )
        if not (np.all(alpha >= 0.0) and np.all(alpha <= 1.0)):
            raise ValueError("alpha values must all be in [0, 1]")
        alpha_arr = alpha.astype(np.float32, copy=False)
    else:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        alpha_arr = float(alpha)  # scalar; numpy broadcasts

    # Batched matrix → quaternion (xyzw) via Shepperd's method, vectorized.
    # Avoids constructing per-call rotation objects (which dominate the
    # per-call cost on small batches).
    q_old = _mat_to_quat_xyzw_batched(R_old)
    q_new = _mat_to_quat_xyzw_batched(R_new)

    # Sign-flip new quaternions whose dot product with old is negative so the
    # interpolation takes the short path (q and -q are the same rotation).
    dots = np.sum(q_old * q_new, axis=1)
    q_new = np.where(dots[:, None] < 0.0, -q_new, q_new)
    dots = np.abs(dots)

    # Stable theta from clipped dot product.
    dots_clipped = np.clip(dots, -1.0, 1.0)
    theta = np.arccos(dots_clipped)
    sin_theta = np.sin(theta)

    near_parallel = sin_theta < 1e-6
    s_old = np.where(
        near_parallel,
        1.0 - alpha_arr,
        np.sin((1.0 - alpha_arr) * theta) / np.where(near_parallel, 1.0, sin_theta),
    )
    s_new = np.where(
        near_parallel,
        alpha_arr,
        np.sin(alpha_arr * theta) / np.where(near_parallel, 1.0, sin_theta),
    )
    q_interp = s_old[:, None] * q_old + s_new[:, None] * q_new

    # Re-normalize (covers both the lerp fallback and any fp noise).
    norms = np.linalg.norm(q_interp, axis=1, keepdims=True)
    q_interp = q_interp / np.maximum(norms, 1e-12)

    return _quat_xyzw_to_mat_batched(q_interp)


def slerp_se3_batched(
    T_old: np.ndarray,
    T_new: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Per-row SE(3) slerp on `(N, 4, 4)` batches with per-row alphas."""
    if T_old.shape != T_new.shape:
        raise ValueError(
            f"T_old / T_new shapes must match; got {T_old.shape} vs {T_new.shape}"
        )
    if T_old.size == 0:
        return T_old.astype(np.float32, copy=False)
    if T_old.ndim != 3 or T_old.shape[1:] != (4, 4):
        raise ValueError(f"T_old must be (N, 4, 4); got {T_old.shape}")
    if alphas.shape != (T_old.shape[0],):
        raise ValueError(
            f"alphas shape must be ({T_old.shape[0]},); got {alphas.shape}"
        )

    R_old = T_old[:, :3, :3]
    R_new = T_new[:, :3, :3]
    R_interp = slerp_so3_batched(R_old, R_new, alphas)

    # Translation: plain lerp on the (N, 3) translation column.
    t_old = T_old[:, :3, 3]
    t_new = T_new[:, :3, 3]
    a = alphas.astype(np.float32, copy=False)[:, None]
    t_interp = (1.0 - a) * t_old + a * t_new

    out = np.tile(np.eye(4, dtype=np.float32), (T_old.shape[0], 1, 1))
    out[:, :3, :3] = R_interp.astype(np.float32, copy=False)
    out[:, :3, 3] = t_interp.astype(np.float32, copy=False)
    return out


def _mat_to_quat_xyzw_batched(R: np.ndarray) -> np.ndarray:
    """Vectorized rotation matrix → quaternion (xyzw) via Shepperd's method."""
    R32 = R.astype(np.float32, copy=False)
    m00 = R32[:, 0, 0]
    m01 = R32[:, 0, 1]
    m02 = R32[:, 0, 2]
    m10 = R32[:, 1, 0]
    m11 = R32[:, 1, 1]
    m12 = R32[:, 1, 2]
    m20 = R32[:, 2, 0]
    m21 = R32[:, 2, 1]
    m22 = R32[:, 2, 2]
    trace = m00 + m11 + m22

    # Case 1: trace > 0
    s1 = 2.0 * np.sqrt(np.maximum(trace + 1.0, 1e-12))
    w1 = 0.25 * s1
    x1 = (m21 - m12) / s1
    y1 = (m02 - m20) / s1
    z1 = (m10 - m01) / s1

    # Case 2: m00 > m11 and m00 > m22
    s2 = 2.0 * np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12))
    w2 = (m21 - m12) / s2
    x2 = 0.25 * s2
    y2 = (m01 + m10) / s2
    z2 = (m02 + m20) / s2

    # Case 3: m11 > m22
    s3 = 2.0 * np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12))
    w3 = (m02 - m20) / s3
    x3 = (m01 + m10) / s3
    y3 = 0.25 * s3
    z3 = (m12 + m21) / s3

    # Case 4: else
    s4 = 2.0 * np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12))
    w4 = (m10 - m01) / s4
    x4 = (m02 + m20) / s4
    y4 = (m12 + m21) / s4
    z4 = 0.25 * s4

    case1 = trace > 0.0
    case2 = (~case1) & (m00 > m11) & (m00 > m22)
    case3 = (~case1) & (~case2) & (m11 > m22)
    # case4 implied otherwise.

    w = np.where(case1, w1, np.where(case2, w2, np.where(case3, w3, w4)))
    x = np.where(case1, x1, np.where(case2, x2, np.where(case3, x3, x4)))
    y = np.where(case1, y1, np.where(case2, y2, np.where(case3, y3, y4)))
    z = np.where(case1, z1, np.where(case2, z2, np.where(case3, z3, z4)))
    return np.stack([x, y, z, w], axis=1).astype(np.float32, copy=False)


def _quat_xyzw_to_mat_batched(q: np.ndarray) -> np.ndarray:
    """Vectorized quaternion (xyzw) → rotation matrix. Returns `(J, 3, 3)`."""
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    out[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[:, 0, 1] = 2.0 * (xy - wz)
    out[:, 0, 2] = 2.0 * (xz + wy)
    out[:, 1, 0] = 2.0 * (xy + wz)
    out[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[:, 1, 2] = 2.0 * (yz - wx)
    out[:, 2, 0] = 2.0 * (xz - wy)
    out[:, 2, 1] = 2.0 * (yz + wx)
    out[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out
