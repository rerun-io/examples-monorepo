"""Project Aria's FISHEYE624 lens: projection, Newton unprojection, and the calibration edits the SDK applies.

Every formula is projectaria-tools' (2.3, ``FisheyeRadTanThinPrism.h`` and
``CameraCalibration.cpp``), evaluated in the same order, so projections agree
with ``CameraCalibration.project`` to the last bit. The arctangents go through
libm (``math.atan``/``math.atan2``) point by point for the same reason: numpy's
SIMD ``arctan`` is within an ulp of libm but not equal to it on AVX-512 hosts.

Parameters are ``[f, cx, cy, k1..k6, p1, p2, s1..s4]`` (see ``Fisheye624Parameters``).
Pixel centres sit at integer coordinates, so the image spans
``[-0.5, width - 0.5] x [-0.5, height - 0.5]``.
"""

import math
from collections.abc import Callable

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray

from simplecv.camera_parameters import Fisheye624Parameters
from simplecv.se3 import SE3

NEWTON_MAX_ITERATIONS: int = 50
"""``CameraNewtonsMethod::kMaxIterations``."""
NEWTON_TOLERANCE: float = float(np.float32(1e-7))
"""``CameraNewtonsMethod::kDoubleTolerance``: a float32 constant, widened where it is compared."""
NEWTON_TOLERANCE_SQ: float = float(np.float32(1e-7) * np.float32(1e-7))
"""Its square, computed in float32 as the SDK does, for the 2-vector convergence test."""

SOPHUS_EPSILON: float = 1e-10
"""``Sophus::Constants<double>::epsilon()``, the floor on the angle solve's derivative."""



def _libm(function: Callable[..., float], *arguments: Float64[ndarray, "n"]) -> Float64[ndarray, "n"]:
    """Apply a libm function point by point (see the module docstring for why not numpy's)."""
    return np.fromiter(map(function, *arguments), dtype=np.float64, count=len(arguments[0]))


def _distort(params: Float64[ndarray, "15"], xr: Float64[ndarray, "n"], yr: Float64[ndarray, "n"]) -> tuple[Float64[ndarray, "n"], Float64[ndarray, "n"]]:
    """Tangential and thin-prism terms on the radially distorted normalised point."""
    p1, p2, s1, s2, s3, s4 = (float(value) for value in params[9:15])
    squared_norm: Float64[ndarray, "n"] = xr * xr + yr * yr
    temp: Float64[ndarray, "n"] = 2.0 * (xr * p1 + yr * p2)
    u: Float64[ndarray, "n"] = xr + (temp * xr + squared_norm * p1)
    v: Float64[ndarray, "n"] = yr + (temp * yr + squared_norm * p2)
    fourth: Float64[ndarray, "n"] = squared_norm * squared_norm
    return u + (s1 * squared_norm + s2 * fourth), v + (s3 * squared_norm + s4 * fourth)


def _distortion_jacobian(
    params: Float64[ndarray, "15"], xr: Float64[ndarray, "n"], yr: Float64[ndarray, "n"]
) -> tuple[Float64[ndarray, "n"], Float64[ndarray, "n"], Float64[ndarray, "n"], Float64[ndarray, "n"]]:
    """``compute_duvDistorted_dxryr``: the 2x2 Jacobian of ``_distort`` as (00, 01, 10, 11)."""
    p1, p2, s1, s2, s3, s4 = (float(value) for value in params[9:15])
    squared_norm: Float64[ndarray, "n"] = xr * xr + yr * yr
    offdiag: Float64[ndarray, "n"] = 2.0 * (xr * p2 + yr * p1)
    temp1: Float64[ndarray, "n"] = 2.0 * (s1 + 2.0 * s2 * squared_norm)
    temp2: Float64[ndarray, "n"] = 2.0 * (s3 + 2.0 * s4 * squared_norm)
    return (
        1.0 + 6.0 * xr * p1 + 2.0 * yr * p2 + xr * temp1,
        offdiag + yr * temp1,
        offdiag + xr * temp2,
        1.0 + 6.0 * yr * p2 + 2.0 * xr * p1 + yr * temp2,
    )


def project_no_checks(camera: Fisheye624Parameters, points_cam: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
    """Project camera-frame points with no visibility test (``projectNoChecks``); z must not be 0."""
    params: Float64[ndarray, "15"] = camera.params
    inv_z: Float64[ndarray, "n"] = 1.0 / points_cam[:, 2]
    a: Float64[ndarray, "n"] = points_cam[:, 0] * inv_z
    b: Float64[ndarray, "n"] = points_cam[:, 1] * inv_z
    r: Float64[ndarray, "n"] = np.sqrt(a * a + b * b)
    theta: Float64[ndarray, "n"] = _libm(math.atan, r)
    theta_sq: Float64[ndarray, "n"] = theta * theta
    theta_radial: Float64[ndarray, "n"] = np.ones_like(theta)
    theta_power: Float64[ndarray, "n"] = theta_sq
    for k in params[3:9]:
        theta_radial = theta_radial + theta_power * float(k)
        theta_power = theta_power * theta_sq
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_by_r: Float64[ndarray, "n"] = np.where(r < np.finfo(np.float64).eps, 1.0, theta / r)
    scale: Float64[ndarray, "n"] = theta_radial * theta_by_r
    u, v = _distort(params, scale * a, scale * b)
    f, cx, cy = (float(value) for value in params[:3])
    return np.stack([f * u + cx, f * v + cy], axis=1)


def visible_pixels(camera: Fisheye624Parameters, pixels: Float64[ndarray, "n 2"]) -> Bool[ndarray, "n"]:
    """``isVisible``: inside the image and, when the sensor has one, inside the valid radius."""
    inside: Bool[ndarray, "n"] = (
        (pixels[:, 0] >= -0.5) & (pixels[:, 0] <= camera.width - 0.5) & (pixels[:, 1] >= -0.5) & (pixels[:, 1] <= camera.height - 0.5)
    )
    if camera.valid_radius is None:
        return inside
    du: Float64[ndarray, "n"] = pixels[:, 0] - float(camera.params[1])
    dv: Float64[ndarray, "n"] = pixels[:, 1] - float(camera.params[2])
    return inside & (du * du + dv * dv <= camera.valid_radius * camera.valid_radius)


def project(camera: Fisheye624Parameters, points_cam: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
    """Project camera-frame points as ``CameraCalibration.project`` does, NaN where it returns nothing.

    A point projects when it lies inside the field-of-view cone
    (``atan2(|xy|, z) <= max_solid_angle``) and its pixel is visible.
    """
    norm_xy: Float64[ndarray, "n"] = np.sqrt(points_cam[:, 0] * points_cam[:, 0] + points_cam[:, 1] * points_cam[:, 1])
    in_cone: Bool[ndarray, "n"] = _libm(math.atan2, norm_xy, points_cam[:, 2]) <= camera.max_solid_angle
    pixels: Float64[ndarray, "n 2"] = np.full((len(points_cam), 2), np.nan)
    pixels[in_cone] = project_no_checks(camera, points_cam[in_cone])
    pixels[~visible_pixels(camera, pixels)] = np.nan
    return pixels


def unproject(camera: Fisheye624Parameters, pixels: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 3"]:
    """Back-project visible pixels to rays ``(x, y, 1)`` as ``CameraCalibration.unproject`` does, NaN elsewhere.

    Two Newton solves, as in the SDK: the tangential and thin-prism terms for
    the radially distorted point, then the radial polynomial for the angle.
    """
    params: Float64[ndarray, "15"] = camera.params
    f, cx, cy = (float(value) for value in params[:3])
    visible: Bool[ndarray, "n"] = visible_pixels(camera, pixels)
    target_u: Float64[ndarray, "m"] = (pixels[visible, 0] - cx) / f
    target_v: Float64[ndarray, "m"] = (pixels[visible, 1] - cy) / f

    xr: Float64[ndarray, "m"] = target_u.copy()
    yr: Float64[ndarray, "m"] = target_v.copy()
    active: Bool[ndarray, "m"] = np.ones(len(xr), dtype=np.bool_)
    for _ in range(NEWTON_MAX_ITERATIONS):
        if not active.any():
            break
        x, y = xr[active], yr[active]
        u, v = _distort(params, x, y)
        j00, j01, j10, j11 = _distortion_jacobian(params, x, y)
        inv_det: Float64[ndarray, "k"] = 1.0 / (j00 * j11 - j10 * j01)
        du: Float64[ndarray, "k"] = target_u[active] - u
        dv: Float64[ndarray, "k"] = target_v[active] - v
        step_u: Float64[ndarray, "k"] = (j11 * inv_det) * du + (-j01 * inv_det) * dv
        step_v: Float64[ndarray, "k"] = (-j10 * inv_det) * du + (j00 * inv_det) * dv
        xr[active] = x + step_u
        yr[active] = y + step_v
        active[np.flatnonzero(active)[step_u * step_u + step_v * step_v < NEWTON_TOLERANCE_SQ]] = False

    radial: Float64[ndarray, "m"] = np.sqrt(xr * xr + yr * yr)
    theta: Float64[ndarray, "m"] = radial.copy()
    active = np.ones(len(theta), dtype=np.bool_)
    epsilon: float = SOPHUS_EPSILON
    for _ in range(NEWTON_MAX_ITERATIONS):
        if not active.any():
            break
        th = theta[active]
        th_sq = th * th
        th_radial = np.ones_like(th)
        derivative = np.ones_like(th)
        th_power = th_sq
        for i, k in enumerate(params[3:9]):
            th_radial = th_radial + th_power * float(k)
            derivative = derivative + float(2 * i + 3) * float(k) * th_power
            th_power = th_power * th_sq
        th_radial = th_radial * th
        residual = radial[active] - th_radial
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(
                np.abs(derivative) > epsilon,
                residual / derivative,
                np.where(residual * derivative > 0.0, 10.0 * epsilon, -10.0 * epsilon),
            )
        th = th + step
        converged = np.abs(step) < NEWTON_TOLERANCE
        th = np.where(~converged & (np.abs(th) >= math.pi / 2.0), 0.999 * math.pi / 2.0, th)
        theta[active] = th
        active[np.flatnonzero(active)[converged]] = False

    rays: Float64[ndarray, "n 3"] = np.full((len(pixels), 3), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor: Float64[ndarray, "m"] = _libm(math.tan, theta) / radial
    on_axis: Bool[ndarray, "m"] = radial == 0.0
    rays[visible, 0] = np.where(on_axis, 0.0, factor * xr)
    rays[visible, 1] = np.where(on_axis, 0.0, factor * yr)
    rays[visible, 2] = 1.0
    return rays


def rotate_cw90(camera: Fisheye624Parameters) -> Fisheye624Parameters:
    """The calibration of the same camera once its images are turned a quarter turn clockwise.

    ``rotateCameraCalibCW90Deg``: width and height swap, the principal point
    moves to ``(h - cy - 1, cx)``, the tangential terms become ``(-p2, p1)`` and
    the thin-prism terms ``(-s3, -s4, s1, s2)``; the pose turns by -90° about the
    optical axis (a right-multiplied ``rotZ(-pi/2)``), its translation unchanged.
    The field of view and the valid radius do not change.
    """
    old: Float64[ndarray, "15"] = camera.params
    params: Float64[ndarray, "15"] = np.array(
        [old[0], camera.height - old[2] - 1, old[1], *old[3:9], -old[10], old[9], -old[13], -old[14], old[11], old[12]], dtype=np.float64
    )
    return Fisheye624Parameters(
        name=camera.name,
        width=camera.height,
        height=camera.width,
        params=params,
        rig_T_cam=camera.rig_T_cam @ SE3.rot_z(math.pi / -2.0),
        max_solid_angle=camera.max_solid_angle,
        valid_radius=camera.valid_radius,
    )


def rescale(camera: Fisheye624Parameters, *, width: int, height: int, scale: float, origin_offset: tuple[float, float] = (0.0, 0.0)) -> Fisheye624Parameters:
    """``CameraCalibration.rescale``: crop by ``origin_offset``, then scale focal length, principal point and valid radius.

    The principal point scales about pixel corners: ``scale * (c - offset + 0.5) - 0.5``.
    """
    params: Float64[ndarray, "15"] = camera.params.copy()
    params[1] -= origin_offset[0]
    params[2] -= origin_offset[1]
    params[0] *= scale
    params[1] = scale * (params[1] + 0.5) - 0.5
    params[2] = scale * (params[2] + 0.5) - 0.5
    return Fisheye624Parameters(
        name=camera.name,
        width=width,
        height=height,
        params=params,
        rig_T_cam=camera.rig_T_cam,
        max_solid_angle=camera.max_solid_angle,
        valid_radius=None if camera.valid_radius is None else camera.valid_radius * scale,
    )
