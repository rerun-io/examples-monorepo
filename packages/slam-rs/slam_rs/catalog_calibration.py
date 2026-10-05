"""Convert catalog camera and IMU geometry into estimator calibration."""

from dataclasses import dataclass

from jaxtyping import Float64
from numpy import ndarray
from simplecv.imu_calibration import ImuCalibration

from slam_rs import _core
from slam_rs.rig import CameraCalib as CameraCalib
from slam_rs.rig import CameraModelName as CameraModelName
from slam_rs.rig import ImuCalib as ImuCalib

CHILD_FROM_PARENT: int = 2
"""``rr.TransformRelation.ChildFromParent``; the only relation the extrinsic inversion is valid for."""
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
    return _core.catalog_camera_calib(index, statics, downscale)


def imu_calib(calibration: ImuCalibration, imu_T_body: Float64[ndarray, "4 4"], applied_time_shift_ns: int) -> ImuCalib:
    """Convert catalog calibration into estimator units without re-aligning samples.

    The inverse ingestion shift moves *both* cameras and IMU to the historical
    estimator time origin. It must never be applied only to the camera stream.
    Factory per-camera offsets are provenance; ingestion already aligned samples.
    """
    return _core.catalog_imu_calib(calibration, imu_T_body, applied_time_shift_ns)
