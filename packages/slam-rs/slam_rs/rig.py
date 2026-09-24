"""The rig as the estimator sees it: entity paths and the calibration dataclasses.

Dependency-free on purpose (numpy + jaxtyping only): ``frontend_log`` and any Vio-only consumer import
from here, while the catalog readers in ``catalog_feed`` / ``catalog_calibration`` build these values
from a Rerun dataset and re-export the names for their callers.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

from jaxtyping import Float64
from numpy import ndarray

RIG_ENTITY: str = "/world/rig_00"
"""Rig node of the ``exoego:v2`` tree; its reference frame is the IMU."""
IMU_ENTITY: str = "/world/rig_00/imu_00"
"""IMU node, whose transform is the identity because the IMU *is* the rig frame."""
TIMELINE: str = "video_time"
"""The one index both datasets carry: nanoseconds since ``property:capture:start_time_ns``."""

CameraModelName: TypeAlias = Literal["kb4", "radtan8"]
"""Projection models V0 supports, using the accepted calibration names."""


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
    """Nominal update rate, from the catalog calibration."""
    gyro_noise_std: float
    """Gyroscope noise density."""
    accel_noise_std: float
    """Accelerometer noise density."""
    gyro_bias_std: float
    """Gyroscope bias random walk."""
    accel_bias_std: float
    """Accelerometer bias random walk."""
    cam_time_offset_ns: int
    """Common shift added to cameras, IMU and GT to restore the estimator time origin; not a relative correction."""
    imu_T_body: Float64[ndarray, "4 4"]
    """Body pose in the IMU frame; the identity whenever the rig reference is the IMU."""
