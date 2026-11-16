"""Canonical uniform camera rings and camera serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from fdanyone.config import CAMERA, FRAMING, CameraConfig

WORLD_FRAME = {
    "name": "canonical_human_world",
    "handedness": "right",
    "axes": {
        "x": "right in the source-facing ring camera (the subject's anatomical left)",
        "y": "up",
        "z": "front; the subject initially faces +z and the source-facing camera lies on the +z side",
    },
    "origin": "initial root projected to the ground plane",
    "units": "meters",
}

CAMERA_FRAME = {
    "name": "opencv_camera",
    "handedness": "right",
    "axes": {"x": "image right", "y": "image down", "z": "forward from camera into the scene"},
    "matrix_convention": "column vectors: x_camera = world_to_camera @ x_world_homogeneous",
    "intrinsics_convention": "pixels with origin at the top-left",
}


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length direction vector.")
    return vector / norm


def look_at_pytorch3d(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the column-vector w2c used by PyTorch3D's look_at_rotation."""

    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = _normalize(target - position)
    x_axis = _normalize(np.cross(up, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)
    translation = -(rotation @ position)
    return rotation, translation


def pytorch3d_to_opencv(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation_cv = np.asarray(rotation, dtype=np.float64).copy()
    translation_cv = np.asarray(translation, dtype=np.float64).copy()
    rotation_cv[:2] *= -1.0
    translation_cv[:2] *= -1.0
    return rotation_cv, translation_cv


def homogeneous_w2c(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


@dataclass(frozen=True)
class Camera:
    camera_id: int
    layer_index: int
    yaw_degrees: float
    azimuth_degrees: float
    pitch_degrees: float
    position: tuple[float, float, float]
    K: tuple[tuple[float, float, float], ...]
    world_to_camera: tuple[tuple[float, float, float, float], ...]
    camera_to_world: tuple[tuple[float, float, float, float], ...]
    image_width: int
    image_height: int

    def to_dict(self) -> dict:
        return asdict(self)


def reference_intrinsics(
    image_height: int,
    image_width: int,
    max_render_height: int = 1280,
    *,
    focal_normalized: float = FRAMING.reference_focal_normalized,
) -> np.ndarray:
    """Reproduce the renderer's downscale-then-rescale intrinsic construction."""

    divisor = 2
    while image_height / divisor > max_render_height:
        divisor += 1
    render_height = image_height // divisor
    render_width = image_width // divisor
    scale = image_height / render_height
    if focal_normalized <= 0:
        raise ValueError("focal_normalized must be positive.")
    focal = focal_normalized * render_height
    intrinsic = np.array(
        [[focal, 0.0, render_width / 2.0], [0.0, focal, render_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsic *= scale
    intrinsic[2, 2] = 1.0
    return intrinsic


def camera_ring(
    *,
    center: np.ndarray,
    front_direction: np.ndarray,
    K: np.ndarray,
    image_height: int,
    image_width: int,
    radius: float = FRAMING.reference_radius,
    target_height: float = FRAMING.reference_target_height,
    spec: CameraConfig = CAMERA,
    start_yaw_degrees: float = 0.0,
    yaw_span_degrees: float = 360.0,
    layer_index: int = 0,
    camera_id_offset: int = 0,
) -> tuple[Camera, ...]:
    center = np.asarray(center, dtype=np.float64).copy()
    front_direction = np.asarray(front_direction, dtype=np.float64)
    front_xz = front_direction[[0, 2]]
    front_azimuth = np.arctan2(front_xz[1], front_xz[0])
    # Relative yaw zero faces the person's front; start_yaw chooses where the
    # first camera lies and IDs then advance uniformly through the span.
    azimuth_start = front_azimuth + np.pi + np.deg2rad(start_yaw_degrees)
    target = center.copy()
    if radius <= 0:
        raise ValueError("radius must be positive.")
    target[1] = target_height
    camera_height = target_height + radius * np.tan(np.deg2rad(spec.pitch_degrees))

    cameras: list[Camera] = []
    yaw_span_radians = np.deg2rad(yaw_span_degrees)
    for view_index in range(spec.count):
        camera_id = camera_id_offset + view_index
        yaw_degrees = start_yaw_degrees + view_index / spec.count * yaw_span_degrees
        azimuth = azimuth_start + view_index / spec.count * yaw_span_radians
        position = np.array(
            [
                center[0] + radius * np.cos(azimuth),
                camera_height,
                center[2] + radius * np.sin(azimuth),
            ],
            dtype=np.float64,
        )
        rotation_p3d, translation_p3d = look_at_pytorch3d(position, target)
        rotation_cv, translation_cv = pytorch3d_to_opencv(rotation_p3d, translation_p3d)
        w2c = homogeneous_w2c(rotation_cv, translation_cv)
        c2w = np.linalg.inv(w2c)
        cameras.append(
            Camera(
                camera_id=camera_id,
                layer_index=layer_index,
                yaw_degrees=float(yaw_degrees),
                azimuth_degrees=float(np.rad2deg(azimuth) % 360.0),
                pitch_degrees=spec.pitch_degrees,
                position=tuple(float(value) for value in position),
                K=tuple(tuple(float(value) for value in row) for row in K),
                world_to_camera=tuple(tuple(float(value) for value in row) for row in w2c),
                camera_to_world=tuple(tuple(float(value) for value in row) for row in c2w),
                image_width=image_width,
                image_height=image_height,
            )
        )
    return tuple(cameras)


def camera_grid(
    *,
    center: np.ndarray,
    front_direction: np.ndarray,
    K: np.ndarray,
    image_height: int,
    image_width: int,
    views_per_layer: int,
    layer_pitches: tuple[int, ...],
    start_yaw: int,
    yaw_span: int,
    radius: float = FRAMING.reference_radius,
    target_height: float = FRAMING.reference_target_height,
) -> tuple[Camera, ...]:
    """Create a layer-major target camera grid."""

    return tuple(
        camera
        for layer_index, pitch in enumerate(layer_pitches)
        for camera in camera_ring(
            center=center,
            front_direction=front_direction,
            K=K,
            image_height=image_height,
            image_width=image_width,
            radius=radius,
            target_height=target_height,
            spec=CameraConfig(count=views_per_layer, pitch_degrees=float(pitch)),
            start_yaw_degrees=float(start_yaw),
            yaw_span_degrees=float(yaw_span),
            layer_index=layer_index,
            camera_id_offset=layer_index * views_per_layer,
        )
    )


def project_points(points_world: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64)
    w2c = np.asarray(camera.world_to_camera)
    K = np.asarray(camera.K)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    points_camera = points_h @ w2c[:3].T
    homogeneous = points_camera @ K.T
    xy = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-8)
    valid = (
        (points_camera[:, 2] > 0.0)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < camera.image_width)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < camera.image_height)
    )
    return xy.astype(np.float32), points_camera[:, 2].astype(np.float32), valid
