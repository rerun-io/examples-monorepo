"""SimpleCV rig-camera adapter for LAMP tracking and ray lifting."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

import cv2
import numpy as np
from jaxtyping import Float, Float32, Float64
from numpy import ndarray
from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters

CameraParameters: TypeAlias = PinholeParameters | Fisheye62Parameters


@dataclass(frozen=True, slots=True)
class PerCameraCalibration:
    """Minimal camera contract consumed by LAMP tracking and snippet building."""

    cam_idx: int
    label: str
    image_size: tuple[int, int]
    intrinsic_matrix: Float32[ndarray, "3 3"]
    T_device_camera: Float32[ndarray, "4 4"]
    distortion_model: str
    project: Callable[[Float[ndarray, "3"]], Float32[ndarray, "2"] | None] | None
    unproject: Callable[[Float[ndarray, "2"]], Float32[ndarray, "3"] | None] | None
    projection_params: Float32[ndarray, "n"] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


@dataclass(frozen=True, slots=True)
class RigCamera:
    """One fixed rig camera with native projection and LAMP lift conversion."""

    parameters: CameraParameters
    """SimpleCV pinhole or Kannala–Brandt camera parameters."""
    image_size: tuple[int, int] = field(init=False)
    """Image size as ``(width, height)`` pixels."""
    K: Float64[ndarray, "3 3"] = field(init=False)
    """Camera intrinsic matrix."""
    cam_T_rig: Float64[ndarray, "4 4"] = field(init=False)
    """Transform from rig-frame coordinates to camera-frame coordinates."""

    def __post_init__(self) -> None:
        """Cache the common calibration fields in stable NumPy dtypes."""
        intrinsics = self.parameters.intrinsics
        object.__setattr__(self, "image_size", (int(intrinsics.width), int(intrinsics.height)))
        object.__setattr__(self, "K", np.asarray(intrinsics.k_matrix, dtype=np.float64))
        object.__setattr__(self, "cam_T_rig", np.asarray(self.parameters.extrinsics.cam_T_world, dtype=np.float64))

    @property
    def name(self) -> str:
        """Human-readable sensor name."""
        return self.parameters.name

    def _kb4_coefficients(self) -> Float64[ndarray, "4"]:
        """Return OpenCV's four Kannala–Brandt radial coefficients."""
        if not isinstance(self.parameters, Fisheye62Parameters):
            return np.zeros(4, dtype=np.float64)
        distortion = self.parameters.distortion
        if distortion is None:
            return np.zeros(4, dtype=np.float64)
        return np.array([distortion.k1, distortion.k2, distortion.k3, distortion.k4], dtype=np.float64)

    def project(self, points_cam: Float[ndarray, "*batch 3"]) -> Float32[ndarray, "*batch 2"]:
        """Project camera-frame points to native image pixels.

        Args:
            points_cam: Camera-frame floating-point coordinates.

        Returns:
            Native pixels, ``Float32[ndarray, "*batch 2"]``.
        """
        points: Float64[ndarray, "*batch 3"] = np.asarray(points_cam, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError(f"points_cam must end in 3 coordinates, got {points.shape}")
        batch_shape: tuple[int, ...] = points.shape[:-1]
        flat: Float64[ndarray, "n 3"] = points.reshape(-1, 3)
        if isinstance(self.parameters, Fisheye62Parameters):
            projected, _ = cv2.fisheye.projectPoints(
                flat.reshape(-1, 1, 3),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                self.K,
                self._kb4_coefficients(),
            )
            pixels: Float64[ndarray, "n 2"] = projected.reshape(-1, 2)
        else:
            depth: Float64[ndarray, "n 1"] = flat[:, 2:3]
            if np.any(np.abs(depth) < 1e-12):
                raise ValueError("Cannot project a point with zero camera-space depth.")
            normalized: Float64[ndarray, "n 2"] = flat[:, :2] / depth
            pixels = normalized * np.diag(self.K)[:2] + self.K[:2, 2]
        return pixels.reshape(*batch_shape, 2).astype(np.float32, copy=False)

    def unproject(self, pixels: Float[ndarray, "*batch 2"]) -> Float32[ndarray, "*batch 3"]:
        """Unproject native image pixels to unit camera-frame rays.

        Args:
            pixels: Native floating-point pixel coordinates.

        Returns:
            Unit camera-frame rays, ``Float32[ndarray, "*batch 3"]``.
        """
        pixels_array: Float64[ndarray, "*batch 2"] = np.asarray(pixels, dtype=np.float64)
        if pixels_array.shape[-1] != 2:
            raise ValueError(f"pixels must end in 2 coordinates, got {pixels_array.shape}")
        batch_shape: tuple[int, ...] = pixels_array.shape[:-1]
        flat: Float64[ndarray, "n 2"] = pixels_array.reshape(-1, 2)
        if isinstance(self.parameters, Fisheye62Parameters):
            normalized: Float64[ndarray, "n 2"] = cv2.fisheye.undistortPoints(
                flat.reshape(-1, 1, 2), self.K, self._kb4_coefficients()
            ).reshape(-1, 2)
        else:
            normalized = (flat - self.K[:2, 2]) / np.diag(self.K)[:2]
        rays: Float64[ndarray, "n 3"] = np.concatenate([normalized, np.ones((len(flat), 1), dtype=np.float64)], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays.reshape(*batch_shape, 3).astype(np.float32, copy=False)

    def lifter_params(self) -> Float32[ndarray, "4"]:
        """Return LAMP's pinhole vector ``[fx, fy, cx, cy]``."""
        return np.array([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]], dtype=np.float32)

    def as_lamp_calibration(self, cam_idx: int) -> PerCameraCalibration:
        """Adapt this camera to the minimal calibration used by LAMP.

        The empty ``projection_params`` deliberately selects LAMP's four-value
        pinhole unprojector. Native KB4 callbacks remain available to the
        tracker for cross-view association.

        Args:
            cam_idx: Stable zero-based view index.

        Returns:
            Calibration with a rig-from-camera fixed transform.
        """
        return PerCameraCalibration(
            cam_idx=cam_idx,
            label=self.name,
            image_size=self.image_size,
            intrinsic_matrix=self.K.astype(np.float32),
            T_device_camera=np.linalg.inv(self.cam_T_rig).astype(np.float32),
            distortion_model="kb4" if isinstance(self.parameters, Fisheye62Parameters) else "pinhole",
            project=self.project,
            unproject=self.unproject,
        )

    def to_virtual_pinhole(self, keypoints_uv: Float[ndarray, "*batch 2"]) -> Float32[ndarray, "*batch 2"]:
        """Map native pixels to the same-``K`` virtual-pinhole image.

        Args:
            keypoints_uv: Native floating-point pixel coordinates.

        Returns:
            Virtual-pinhole pixels, ``Float32[ndarray, "*batch 2"]``. Pinhole
            inputs are returned unchanged; KB4 inputs are undistorted with
            :func:`cv2.fisheye.undistortPoints` and reprojected through ``K``.
        """
        pixels: Float64[ndarray, "*batch 2"] = np.asarray(keypoints_uv, dtype=np.float64)
        if pixels.shape[-1] != 2:
            raise ValueError(f"keypoints_uv must end in 2 coordinates, got {pixels.shape}")
        if isinstance(self.parameters, PinholeParameters):
            return pixels.astype(np.float32, copy=True)
        batch_shape: tuple[int, ...] = pixels.shape[:-1]
        virtual: Float64[ndarray, "n 2"] = cv2.fisheye.undistortPoints(
            pixels.reshape(-1, 1, 2),
            self.K,
            self._kb4_coefficients(),
            P=self.K,
        ).reshape(-1, 2)
        return virtual.reshape(*batch_shape, 2).astype(np.float32, copy=False)


def gravity_aligned_world_transform(gravity_world: Float64[ndarray, "3"]) -> Float32[ndarray, "4 4"]:
    """Build upstream's gravity-world transform from a world gravity vector.

    Args:
        gravity_world: Gravity direction and magnitude in world coordinates,
            ``Float64[ndarray, "3"]``.

    Returns:
        Transform from world to a right-handed, Z-up gravity frame,
        ``Float32[ndarray, "4 4"]``.
    """
    gravity: Float32[ndarray, "3"] = np.asarray(gravity_world, dtype=np.float32)
    norm: float = float(np.linalg.norm(gravity))
    if norm < 1e-9:
        raise ValueError("gravity_world must be non-zero.")
    z_axis: Float32[ndarray, "3"] = -gravity / norm
    helper: Float32[ndarray, "3"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(helper, z_axis))) > 0.99:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis: Float32[ndarray, "3"] = helper - np.dot(helper, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis: Float32[ndarray, "3"] = np.cross(z_axis, x_axis)
    world_R_gravity: Float32[ndarray, "3 3"] = np.column_stack([x_axis, y_axis, z_axis])
    gravity_T_world: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    gravity_T_world[:3, :3] = world_R_gravity.T
    return gravity_T_world


__all__ = ("CameraParameters", "PerCameraCalibration", "RigCamera", "gravity_aligned_world_transform")
