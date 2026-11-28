"""Pose math as CASPAR's generated kernels do it: symforce Pose3 storage [qx, qy, qz, qw, tx, ty, tz].

The tangent is [wx, wy, wz, dtx, dty, dtz] with the right-rotation retraction of kernel_PinholePose_retract.cu:
q' = q (x) [w sin(a/2)/a, cos(a/2)], a = sqrt(w.w + 1e-12), t' = t + dt. Rotation matrices use the unit-quaternion
formula on the stored (not renormalised) quaternion, as symforce's Rot3 does, because CASPAR normalises only at
write-back.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from jaxtyping import Float
from numpy import ndarray


def quaternion_matrix(q: Float[ndarray, "n 4"]) -> Float[ndarray, "n 3 3"]:
    """Rotation matrix of [qx, qy, qz, qw] rows (symforce's Rot3.to_rotation_matrix)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrix: Float[ndarray, "n 3 3"] = np.empty((len(q), 3, 3), dtype=q.dtype)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def quaternion_multiply(a: Float[ndarray, "n 4"], b: Float[ndarray, "n 4"]) -> Float[ndarray, "n 4"]:
    """Hamilton product a (x) b of [qx, qy, qz, qw] rows."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=1,
    )


def compose(a: Float[ndarray, "n 7"], b: Float[ndarray, "n 7"]) -> Float[ndarray, "n 7"]:
    """a * b for pose rows: rotation a.q (x) b.q, translation R(a.q) b.t + a.t (symforce Pose3.compose)."""
    rotated: Float[ndarray, "n 3"] = np.einsum("nij,nj->ni", quaternion_matrix(a[:, :4]), b[:, 4:])
    return np.concatenate([quaternion_multiply(a[:, :4], b[:, :4]), rotated + a[:, 4:]], axis=1)


def retract_pose(pose: Float[ndarray, "n 7"], tangent: Float[ndarray, "n 6"]) -> Float[ndarray, "n 7"]:
    """CASPAR's Pose3 retraction (kernel_PinholePose_retract.cu); no renormalisation."""
    w = tangent[:, :3]
    angle = np.sqrt(np.sum(w * w, axis=1) + 1e-12)
    delta = np.concatenate([w * (np.sin(angle / 2) / angle)[:, None], np.cos(angle / 2)[:, None]], axis=1).astype(pose.dtype)
    return np.concatenate([quaternion_multiply(pose[:, :4], delta), pose[:, 4:] + tangent[:, 3:]], axis=1)


def normalize_pose(pose: Float[ndarray, "n 7"]) -> Float[ndarray, "n 7"]:
    """Unit quaternion, as COLMAP's write-back does."""
    return np.concatenate([pose[:, :4] / np.linalg.norm(pose[:, :4], axis=1, keepdims=True), pose[:, 4:]], axis=1)


def skew(v: Float[ndarray, "n 3"]) -> Float[ndarray, "n 3 3"]:
    """[v]_x, so that skew(v) @ x = v x x."""
    zero = np.zeros_like(v[:, 0])
    return np.stack(
        [np.stack([zero, -v[:, 2], v[:, 1]], 1), np.stack([v[:, 2], zero, -v[:, 0]], 1), np.stack([-v[:, 1], v[:, 0], zero], 1)],
        axis=1,
    )


class CameraPoint(NamedTuple):
    """A world point in the camera frame and its derivatives w.r.t. the three tangents."""

    point_cam: Float[ndarray, "n 3"]
    """R_s (R_r X + t_r) + t_s."""
    d_rig_from_world: Float[ndarray, "n 3 6"]
    """d point_cam / d tangent of rig_from_world."""
    d_sensor_from_rig: Float[ndarray, "n 3 6"]
    """d point_cam / d tangent of sensor_from_rig."""
    d_point: Float[ndarray, "n 3 3"]
    """d point_cam / d world point."""


def camera_point(rig_from_world: Float[ndarray, "n 7"], sensor_from_rig: Float[ndarray, "n 7"], point: Float[ndarray, "n 3"]) -> CameraPoint:
    """cam_from_world * point with Jacobians consistent with `retract_pose` (right rotation, additive translation).

    With cam_from_world = sensor_from_rig * rig_from_world and the tangent applied on the right of each rotation:
    d/d w_r = -R_c [X]_x, d/d t_r = R_s, d/d w_s = -R_s [Y]_x with Y = R_r X + t_r, d/d t_s = I, d/d X = R_c.
    """
    rig_rotation = quaternion_matrix(rig_from_world[:, :4])
    sensor_rotation = quaternion_matrix(sensor_from_rig[:, :4])
    # Like the generated code, project with the composed cam_from_world, not through the rig frame.
    cam_from_world = compose(sensor_from_rig, rig_from_world)
    cam_rotation = quaternion_matrix(cam_from_world[:, :4])
    in_rig: Float[ndarray, "n 3"] = np.einsum("nij,nj->ni", rig_rotation, point) + rig_from_world[:, 4:]
    point_cam: Float[ndarray, "n 3"] = np.einsum("nij,nj->ni", cam_rotation, point) + cam_from_world[:, 4:]
    n = len(point)
    d_rig = np.empty((n, 3, 6), dtype=point.dtype)
    d_rig[:, :, :3] = -cam_rotation @ skew(point)
    d_rig[:, :, 3:] = sensor_rotation
    d_sensor = np.empty((n, 3, 6), dtype=point.dtype)
    d_sensor[:, :, :3] = -sensor_rotation @ skew(in_rig)
    d_sensor[:, :, 3:] = np.eye(3, dtype=point.dtype)
    return CameraPoint(point_cam, d_rig, d_sensor, cam_rotation)
