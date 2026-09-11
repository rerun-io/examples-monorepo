"""Convert catalog camera and IMU geometry into estimator calibration."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from slam_rs.reference import ImuParameters

CHILD_FROM_PARENT: int = 2
"""``rr.TransformRelation.ChildFromParent``; the only relation the extrinsic inversion is valid for."""
CameraModelName: TypeAlias = Literal["kb4", "radtan8"]
"""Projection models V0 supports, using the accepted calibration names."""

_MODEL_BY_DISTORTION: dict[str, tuple[CameraModelName, int]] = {"kannala_brandt": ("kb4", 4), "brown_conrady": ("radtan8", 8)}
"""``simplecv.components.DistortionModel`` string to the model and the number of coefficients it uses."""


@dataclass(slots=True, frozen=True)
class CameraStatics:
    """One camera node's static components, exactly as the catalog stores them.

    This is the untouched read side: column-major matrices, ``(width, height)``
    resolution, the raw relation code and the full fixed-width coefficient list.
    :func:`camera_calib` is the only place the conversion rules live, which is
    what lets them be tested without a catalog.
    """

    distortion_model: str
    """``simplecv.components.DistortionModel``, e.g. ``kannala_brandt``.

    The projection model comes from here and not from the camera node's own
    ``camera_model`` string, which the RoboCap conversion predates and some
    writers omit.
    """
    distortion_coefficients: Float64[ndarray, " n_slots"]
    """Fixed-width coefficient list; the unused tail is zero."""
    image_from_camera: Float64[ndarray, " 9"]
    """``Pinhole:image_from_camera``, flat and **column-major**."""
    resolution_wh: Float64[ndarray, " 2"]
    """``Pinhole:resolution``, ``(width, height)`` in pixels."""
    transform_mat3x3: Float64[ndarray, " 9"]
    """Camera ``Transform3D:mat3x3``, flat and **column-major**."""
    transform_translation: Float64[ndarray, " 3"]
    """Camera ``Transform3D:translation``."""
    transform_relation: int
    """``Transform3D:relation``; must be :data:`CHILD_FROM_PARENT`."""
    distortion_valid_radius: float | None
    """The valid radius ``rpmax``; present on msd-g2 only."""


@dataclass(slots=True, frozen=True)
class CameraCalib:
    """One camera as the estimator wants it: metric intrinsics and ``imu_T_cam``."""

    index: int
    """Camera number the estimator knows this camera by, matching the order of a frameset's images.

    On a rig fed whole that is also the rig index. Where
    :attr:`RigProfile.camera_names` feeds a subset it is the position in that
    list, and :attr:`SegmentFeed.camera_positions` maps it back to the rig — so
    the keypoints, the images and the blueprint views all speak the estimator's
    numbering and only the catalog reads speak the rig's.
    """
    width: int
    """Decoded frame width in pixels."""
    height: int
    """Decoded frame height in pixels."""
    fx: float
    """Focal length along image x, pixels."""
    fy: float
    """Focal length along image y, pixels."""
    cx: float
    """Principal point x, pixels."""
    cy: float
    """Principal point y, pixels."""
    model: CameraModelName
    """Projection model."""
    distortion: Float64[ndarray, " n_coeffs"]
    """Exactly the coefficients the model uses: 4 for kb4, ``k1 k2 p1 p2 k3 k4 k5 k6`` for radtan8."""
    distortion_valid_radius: float | None
    """The valid radius ``rpmax``, when the recording carries one."""
    imu_T_cam: Float64[ndarray, "4 4"]
    """Camera pose in the IMU frame: the inverse of the stored ``ChildFromParent`` transform."""


@dataclass(slots=True, frozen=True)
class ImuCalib:
    """The IMU as the estimator wants it: noise model plus the body transform."""

    frequency_hz: float
    """Nominal update rate, from the reference manifest."""
    gyro_noise_std: float
    """Gyroscope noise density."""
    accel_noise_std: float
    """Accelerometer noise density."""
    gyro_bias_std: float
    """Gyroscope bias random walk."""
    accel_bias_std: float
    """Accelerometer bias random walk."""
    cam_time_offset_ns: int
    """Added to a camera timestamp to reach the IMU clock."""
    imu_T_body: Float64[ndarray, "4 4"]
    """Body pose in the IMU frame; the identity whenever the rig reference is the IMU."""


def scale_principal_point(value: float, downscale: int) -> float:
    """Scale a principal-point coordinate by pixel centers.

    A pixel at c has center c + 0.5. At downscale d its index becomes
    (c + 0.5) / d - 0.5. This keeps calibration aligned with area-resampled pixels.
    """
    return (value + 0.5) / downscale - 0.5


def camera_calib(index: int, statics: CameraStatics, downscale: int = 1) -> CameraCalib:
    """Apply the catalog-to-estimator mapping rules to one camera's statics.

    The rules, each of which has cost someone a wrong trajectory: reshape
    ``image_from_camera`` column-major, read the resolution as ``(width, height)``,
    map the distortion model string and assert the coefficient tail is zero
    rather than truncating it, and invert the ``ChildFromParent`` transform to get
    ``imu_T_cam``.

    A ``downscale`` above one scales the resolution and the intrinsics to the
    frames the feed will actually decode, KB4's resolution-invariant coefficients
    untouched.

    Args:
        index: Camera index on the rig.
        statics: Raw static components of the camera node.
        downscale: Integer factor the frames are decoded at.

    Returns:
        The camera calibration in the estimator's conventions.

    Raises:
        ValueError: If the distortion model is unknown, the coefficient tail is
            non-zero, the transform relation is not ``ChildFromParent``, or
            ``downscale`` is below one.
    """
    if downscale < 1:
        raise ValueError(f"cam_{index:02d}: downscale must be at least 1; got {downscale}")
    width: int = int(statics.resolution_wh[0])
    height: int = int(statics.resolution_wh[1])
    if width // downscale < 1 or height // downscale < 1:
        raise ValueError(f"cam_{index:02d}: downscale {downscale} leaves nothing of the {width}x{height} frame")
    if statics.distortion_model not in _MODEL_BY_DISTORTION:
        raise ValueError(f"cam_{index:02d}: unsupported distortion model {statics.distortion_model!r}, known: {sorted(_MODEL_BY_DISTORTION)}")
    model: CameraModelName = _MODEL_BY_DISTORTION[statics.distortion_model][0]
    n_coeffs: int = _MODEL_BY_DISTORTION[statics.distortion_model][1]
    if statics.distortion_coefficients.shape[0] < n_coeffs:
        raise ValueError(f"cam_{index:02d}: {model} needs {n_coeffs} coefficients, got {statics.distortion_coefficients.shape[0]}")
    tail: Float64[ndarray, " n_tail"] = statics.distortion_coefficients[n_coeffs:]
    # A "kannala_brandt" string does not imply KB4 — Aria's Fisheye624 carries the
    # same string with eight live coefficients. Reject the tail, never truncate it.
    if not np.allclose(tail, 0.0):
        raise ValueError(f"cam_{index:02d}: {model} uses {n_coeffs} coefficients but the tail is non-zero: {tail.tolist()}")
    if statics.transform_relation != CHILD_FROM_PARENT:
        raise ValueError(f"cam_{index:02d}: Transform3D relation {statics.transform_relation} is not ChildFromParent({CHILD_FROM_PARENT})")
    k_matrix: Float64[ndarray, "3 3"] = statics.image_from_camera.reshape(3, 3, order="F")
    cam_R_imu: Float64[ndarray, "3 3"] = statics.transform_mat3x3.reshape(3, 3, order="F")
    imu_T_cam: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    imu_T_cam[:3, :3] = cam_R_imu.T
    imu_T_cam[:3, 3] = -cam_R_imu.T @ statics.transform_translation
    return CameraCalib(
        index=index,
        width=width // downscale,
        height=height // downscale,
        fx=float(k_matrix[0, 0]) / downscale,
        fy=float(k_matrix[1, 1]) / downscale,
        cx=scale_principal_point(float(k_matrix[0, 2]), downscale),
        cy=scale_principal_point(float(k_matrix[1, 2]), downscale),
        model=model,
        distortion=statics.distortion_coefficients[:n_coeffs].copy(),
        distortion_valid_radius=statics.distortion_valid_radius,
        imu_T_cam=imu_T_cam,
    )


def imu_calib(parameters: ImuParameters, imu_T_body: Float64[ndarray, "4 4"]) -> ImuCalib:
    """Combine the manifest's frozen noise model with the recording's IMU transform.

    Args:
        parameters: Frozen IMU parameters from the reference manifest.
        imu_T_body: Body pose in the IMU frame, the identity when the rig reference is the IMU.

    Returns:
        The IMU calibration the estimator is configured with.
    """
    return ImuCalib(
        frequency_hz=parameters.rate_hz,
        gyro_noise_std=parameters.gyro_noise_std,
        accel_noise_std=parameters.accel_noise_std,
        gyro_bias_std=parameters.gyro_bias_std,
        accel_bias_std=parameters.accel_bias_std,
        cam_time_offset_ns=parameters.cam_time_offset_ns,
        imu_T_body=imu_T_body,
    )


