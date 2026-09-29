"""Native-pixel circles to perspective cameras; independent of the UmeTrack checkout."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from simplecv.umetrack_temp.perspective_cropping import make_look_at_matrix


@runtime_checkable
class RayCamera(Protocol):
    """The Fisheye62 operations needed to define a circle crop."""
    @property
    def camera_to_world_xf(self) -> Float64[ndarray, "4 4"]: ...

    def window_to_eye(self, pixels: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 3"]: ...


@dataclass(frozen=True, slots=True)
class CircleCrop:
    """Parameters passed to upstream PinholePlaneCameraModel."""
    camera_to_world: Float64[ndarray, "4 4"]
    """World from crop camera, translations in the source camera's units (upstream mm)."""
    focal: float
    """Equal horizontal and vertical focal length in pixels."""
    size: int
    """Square output width and height."""


def circle_crop(camera: RayCamera, circle: Float64[ndarray, "3"], angle: float, hand: int, circle_scale: float,
                size: int = 96, focal_multiplier: float = 0.95) -> CircleCrop:
    """Aim at the centre ray and fit 32 boundary rays, then apply the focal multipliers.

    Args:
        camera: Upstream-compatible ray camera.
        circle: Float64[ndarray, "3"] native (cx, cy, radius) pixels.
        angle: Source camera roll in degrees.
        hand: Left 0, right 1; the latter mirrors crop x.
        circle_scale: Validation-calibrated multiplier of unit-circle focal length.
        size: Square crop resolution.
        focal_multiplier: Upstream hand_ratio_in_crop.
    """
    if not np.isfinite(circle).all() or circle[2] <= 0 or not np.isfinite(circle_scale) or circle_scale <= 0:
        raise ValueError("Circle and scale must be finite with positive radius and scale")
    ray: Float64[ndarray, "3"] = camera.window_to_eye(circle[None, :2])[0]
    # Work at the origin to avoid float32 look-at cancellation at large world translations.
    origin: Float64[ndarray, "4 4"] = camera.camera_to_world_xf.copy()
    origin[:3, 3] = 0.0
    target: Float64[ndarray, "3"] = origin[:3, :3] @ ray
    crop_from_world: Float64[ndarray, "4 4"] = make_look_at_matrix(np.linalg.inv(origin).astype(np.float32), target.astype(np.float32), angle).astype(np.float64)
    if hand == 1:
        crop_from_world[0] *= -1
    theta: Float64[ndarray, "32"] = np.arange(32, dtype=np.float64) * (2 * np.pi / 32)
    pixels: Float64[ndarray, "32 2"] = circle[:2] + circle[2] * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    rays: Float64[ndarray, "32 3"] = camera.window_to_eye(pixels) @ origin[:3, :3].T @ crop_from_world[:3, :3].T
    if not np.isfinite(rays).all() or np.any(rays[:, 2] <= 0.0001):
        raise ValueError("Circle crosses the crop camera's horizon")
    extent: float = float(np.abs(rays[:, :2] / rays[:, 2:]).max())
    focal: float = ((size - 1) / 2) / extent * focal_multiplier * circle_scale
    if not np.isfinite(focal) or focal < 5:
        raise ValueError("Circle crop focal is below upstream minimum 5 px")
    world_from_crop: Float64[ndarray, "4 4"] = np.linalg.inv(crop_from_world)
    world_from_crop[:3, 3] = camera.camera_to_world_xf[:3, 3]
    return CircleCrop(world_from_crop, focal, size)


def select_views(scores: list[float], threshold: float, strict: bool = True) -> list[int]:
    """Rank eligible cameras, cap at upstream MAX_VIEW_NUM=2, then sort camera indices."""
    eligible: list[int] = [index for index, score in enumerate(scores) if np.isfinite(score) and (score > threshold if strict else score >= threshold)]
    return sorted(sorted(eligible, key=lambda index: -scores[index])[:2])


@runtime_checkable
class ProjectionCamera(Protocol):
    """Upstream camera's forward projection, also used to verify numerical inversion."""
    camera_to_world_xf: Float64[ndarray, "4 4"]
    f: tuple[float, float]
    c: tuple[float, float]
    width: int
    height: int

    def world_to_eye(self, points: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]: ...

    def eye_to_window(self, points: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]: ...


@dataclass(frozen=True, slots=True)
class FisheyeRays:
    """Inverse Fisheye62 using the unchanged upstream forward projection.

    Upstream window_to_eye asserts NoDistortion. Newton iterations here solve in
    the arctan plane, in batches, and require <1e-5 px forward residual.
    """
    camera: ProjectionCamera
    """Upstream calibrated camera."""

    @property
    def camera_to_world_xf(self) -> Float64[ndarray, "4 4"]:
        return self.camera.camera_to_world_xf

    def window_to_eye(self, pixels: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 3"]:
        """Unproject Float64[ndarray, n 2] pixels to Float64[ndarray, n 3] unit rays."""
        plane: Float64[ndarray, "n 2"] = (pixels - self.camera.c) / self.camera.f

        def rays(uv: Float64[ndarray, "n 2"]) -> Float64[ndarray, "n 3"]:
            theta: Float64[ndarray, "n 1"] = np.linalg.norm(uv, axis=1, keepdims=True)
            return np.concatenate((uv * np.sinc(theta / np.pi), np.cos(theta)), axis=1)

        for _ in range(30):
            projected: Float64[ndarray, "n 2"] = np.asarray(self.camera.eye_to_window(rays(plane)), dtype=np.float64)
            residual: Float64[ndarray, "n 2"] = projected - pixels
            if np.isfinite(residual).all() and np.max(np.abs(residual)) < 1e-5:
                return rays(plane)
            columns: list[Float64[ndarray, "n 2"]] = []
            for axis in range(2):
                perturbed: Float64[ndarray, "n 2"] = plane.copy()
                perturbed[:, axis] += 1e-6
                columns.append((self.camera.eye_to_window(rays(perturbed)) - projected) / 1e-6)
            jacobian: Float64[ndarray, "n 2 2"] = np.stack(columns, axis=-1)
            try:
                step: Float64[ndarray, "n 2"] = np.linalg.solve(jacobian, residual[..., None])[..., 0]
            except np.linalg.LinAlgError as error:
                raise ValueError("Singular Fisheye62 inverse") from error
            plane -= np.clip(step, -0.25, 0.25)
        raise ValueError("Fisheye62 inverse did not converge within 1e-5 px")
