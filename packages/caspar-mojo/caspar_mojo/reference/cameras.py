"""CASPAR's four camera models (caspar_generate.py: pinhole_core, simple_radial_core, opencv_fisheye_core,
full_opencv_project) with analytic Jacobians, vectorised over observations.

Intrinsics arrive in COLMAP's `camera.params` order. The focal-and-extra Jacobian columns follow CASPAR's
FocalAndExtra node: PINHOLE [fx, fy], SIMPLE_RADIAL [f, k], OPENCV_FISHEYE [fx, fy, k1..k4],
FULL_OPENCV [fx, fy, k1, k2, p1, p2, k3, k4, k5, k6].
"""

from typing import NamedTuple

import numpy as np
from jaxtyping import Float, Int64
from numpy import ndarray

from caspar_mojo.problem import CameraModel

EPSILON: float = 1e-6
"""symforce's `sf.epsilon()` in CASPAR's f32 tree; the Mojo solver uses the same value in every precision."""

FOCAL_AND_EXTRA_COLUMNS: dict[CameraModel, tuple[int, ...]] = {
    CameraModel.PINHOLE: (0, 1),
    CameraModel.SIMPLE_RADIAL: (0, 3),
    CameraModel.OPENCV_FISHEYE: (0, 1, 4, 5, 6, 7),
    CameraModel.FULL_OPENCV: (0, 1, 4, 5, 6, 7, 8, 9, 10, 11),
}
"""Columns of a `camera.params` row that form the refinable focal-and-extra block, in CASPAR's tangent order."""


class Projection(NamedTuple):
    """Pixels of camera-frame points and their derivatives."""

    pixel: Float[ndarray, "n 2"]
    """Projected pixel."""
    d_point: Float[ndarray, "n 2 3"]
    """d pixel / d point_cam."""
    d_focal_and_extra: Float[ndarray, "n 2 k"]
    """d pixel / d focal-and-extra, columns in `FOCAL_AND_EXTRA_COLUMNS` order."""


def _sign_no_zero(x: Float[ndarray, " n"]) -> Float[ndarray, " n"]:
    return np.where(x >= 0, 1.0, -1.0).astype(x.dtype)


def _perspective(point_cam: Float[ndarray, "n 3"]) -> tuple[Float[ndarray, " n"], Float[ndarray, " n"], Float[ndarray, "n 2 3"]]:
    """(u, v) = (X, Y) / (Z + copysign(eps, Z)) and d(u, v)/d point_cam."""
    x, y, z = point_cam[:, 0], point_cam[:, 1], point_cam[:, 2]
    inv: Float[ndarray, " n"] = 1.0 / (z + EPSILON * _sign_no_zero(z))
    u: Float[ndarray, " n"] = x * inv
    v: Float[ndarray, " n"] = y * inv
    d_uv: Float[ndarray, "n 2 3"] = np.zeros((len(x), 2, 3), dtype=point_cam.dtype)
    d_uv[:, 0, 0] = inv
    d_uv[:, 0, 2] = -u * inv
    d_uv[:, 1, 1] = inv
    d_uv[:, 1, 2] = -v * inv
    return u, v, d_uv


def project(model: CameraModel, point_cam: Float[ndarray, "n 3"], params: Float[ndarray, "n 12"]) -> Projection:
    """Project camera-frame points with one camera model.

    Args:
        model: The camera model of every row.
        point_cam: Points in the camera frame.
        params: COLMAP `camera.params` per row, zero-padded to 12 (only the model's leading columns are read).

    Returns:
        The pixels and the analytic Jacobians.
    """
    dtype = point_cam.dtype
    n: int = len(point_cam)
    if model == CameraModel.SIMPLE_RADIAL:
        fx, fy, cx, cy = params[:, 0], params[:, 0], params[:, 1], params[:, 2]
    else:
        fx, fy, cx, cy = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    pixel: Float[ndarray, "n 2"] = np.empty((n, 2), dtype=dtype)

    if model == CameraModel.PINHOLE:
        u, v, d_uv = _perspective(point_cam)
        pixel[:, 0] = fx * u + cx
        pixel[:, 1] = fy * v + cy
        d_point = d_uv * np.stack([fx, fy], axis=1)[:, :, None]
        d_fae = np.zeros((n, 2, 2), dtype=dtype)
        d_fae[:, 0, 0] = u
        d_fae[:, 1, 1] = v
        return Projection(pixel, d_point, d_fae)

    if model == CameraModel.SIMPLE_RADIAL:
        k = params[:, 3]
        u, v, d_uv = _perspective(point_cam)
        q = u * u + v * v
        radial = 1.0 + k * q
        pixel[:, 0] = fx * radial * u + cx
        pixel[:, 1] = fx * radial * v + cy
        uv = np.stack([u, v], axis=1)
        d_pix_d_uv = fx[:, None, None] * (radial[:, None, None] * np.eye(2, dtype=dtype) + 2.0 * k[:, None, None] * uv[:, :, None] * uv[:, None, :])
        d_point = d_pix_d_uv @ d_uv
        d_fae = np.stack([radial[:, None] * uv, (fx * q)[:, None] * uv], axis=2)
        return Projection(pixel, d_point, d_fae)

    if model == CameraModel.OPENCV_FISHEYE:
        k1, k2, k3, k4 = params[:, 4], params[:, 5], params[:, 6], params[:, 7]
        norm = np.sqrt(np.sum(point_cam * point_cam, axis=1) + EPSILON)
        ray = point_cam / norm[:, None]
        d_ray = np.eye(3, dtype=dtype)[None] / norm[:, None, None] - point_cam[:, :, None] * point_cam[:, None, :] / (norm**3)[:, None, None]
        x, y, zr = ray[:, 0], ray[:, 1], ray[:, 2]
        rho = np.sqrt(x * x + y * y + EPSILON)
        w = zr + EPSILON * _sign_no_zero(zr)
        theta = np.arctan2(rho, w)
        t2 = theta * theta
        radial = t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))
        theta_d = theta * (1.0 + radial)
        d_theta_d = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
        scale = theta_d / rho
        denominator = rho * rho + w * w
        # d theta / d ray through rho (x, y) and w (z); d rho / d ray = (x, y, 0) / rho.
        d_rho = np.stack([x / rho, y / rho, np.zeros_like(x)], axis=1)
        d_theta = (w / denominator)[:, None] * d_rho
        d_theta[:, 2] += -rho / denominator
        d_scale = (d_theta_d / rho)[:, None] * d_theta - (theta_d / (rho * rho))[:, None] * d_rho
        d_pix_d_ray = np.empty((n, 2, 3), dtype=dtype)
        d_pix_d_ray[:, 0] = fx[:, None] * (x[:, None] * d_scale)
        d_pix_d_ray[:, 0, 0] += fx * scale
        d_pix_d_ray[:, 1] = fy[:, None] * (y[:, None] * d_scale)
        d_pix_d_ray[:, 1, 1] += fy * scale
        pixel[:, 0] = fx * scale * x + cx
        pixel[:, 1] = fy * scale * y + cy
        d_point = d_pix_d_ray @ d_ray
        d_fae = np.zeros((n, 2, 6), dtype=dtype)
        d_fae[:, 0, 0] = scale * x
        d_fae[:, 1, 1] = scale * y
        power = theta / rho
        for column in range(2, 6):
            power = power * t2
            d_fae[:, 0, column] = fx * x * power
            d_fae[:, 1, column] = fy * y * power
        return Projection(pixel, d_point, d_fae)

    k1, k2, p1, p2, k3, k4, k5, k6 = (params[:, i] for i in range(4, 12))
    u, v, d_uv = _perspective(point_cam)
    u2, uv, v2 = u * u, u * v, v * v
    q = u2 + v2
    numerator = 1.0 + q * (k1 + q * (k2 + q * k3))
    denominator = 1.0 + q * (k4 + q * (k5 + q * k6))
    a = numerator / denominator
    d_numerator = k1 + q * (2.0 * k2 + q * 3.0 * k3)
    d_denominator = k4 + q * (2.0 * k5 + q * 3.0 * k6)
    da = (d_numerator * denominator - numerator * d_denominator) / (denominator * denominator)
    xd = u * a + 2.0 * p1 * uv + p2 * (q + 2.0 * u2)
    yd = v * a + 2.0 * p2 * uv + p1 * (q + 2.0 * v2)
    pixel[:, 0] = fx * xd + cx
    pixel[:, 1] = fy * yd + cy
    d_xy_d_uv = np.empty((n, 2, 2), dtype=dtype)
    d_xy_d_uv[:, 0, 0] = a + 2.0 * u2 * da + 2.0 * p1 * v + 6.0 * p2 * u
    d_xy_d_uv[:, 0, 1] = 2.0 * uv * da + 2.0 * p1 * u + 2.0 * p2 * v
    d_xy_d_uv[:, 1, 0] = 2.0 * uv * da + 2.0 * p2 * v + 2.0 * p1 * u
    d_xy_d_uv[:, 1, 1] = a + 2.0 * v2 * da + 2.0 * p2 * u + 6.0 * p1 * v
    d_point = (np.stack([fx, fy], axis=1)[:, :, None] * d_xy_d_uv) @ d_uv
    d_fae = np.zeros((n, 2, 10), dtype=dtype)
    d_fae[:, 0, 0] = xd
    d_fae[:, 1, 1] = yd
    # Columns 2..9: k1, k2, p1, p2, k3, k4, k5, k6.
    d_a = {
        2: q / denominator,
        3: q * q / denominator,
        6: q**3 / denominator,
        7: -a * q / denominator,
        8: -a * q * q / denominator,
        9: -a * q**3 / denominator,
    }
    for column, value in d_a.items():
        d_fae[:, 0, column] = fx * u * value
        d_fae[:, 1, column] = fy * v * value
    d_fae[:, 0, 4] = fx * 2.0 * uv
    d_fae[:, 1, 4] = fy * (q + 2.0 * v2)
    d_fae[:, 0, 5] = fx * (q + 2.0 * u2)
    d_fae[:, 1, 5] = fy * 2.0 * uv
    return Projection(pixel, d_point, d_fae)


def focal_and_extra_columns(model: CameraModel) -> Int64[ndarray, " k"]:
    """Columns of `camera.params` that the focal-and-extra tangent of `model` moves."""
    return np.asarray(FOCAL_AND_EXTRA_COLUMNS[model], dtype=np.int64)
