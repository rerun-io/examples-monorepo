"""Write a synthetic bundle-adjustment benchmark model: frames on a circle around a point cloud, perturbed start.

The CASPAR -> Mojo port's benchmark problems P1 and P3-P6 come from here. With the defaults (PINHOLE, one camera,
seed 0) the draw order is exactly the earlier `ba_synth.py`'s, so `400` frames and `40000` points reproduce P1
(7.08 px before BA). Pixels carry 0.5 px noise, points start 5 cm off and every frame but the first 5 cm off; a
second rig camera starts with its `sensor_from_rig` knocked off as well, so `refine_sensor_from_rig` has work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pycolmap
from jaxtyping import Float64
from numpy import ndarray

SynthCameraModel = Literal["PINHOLE", "SIMPLE_RADIAL", "OPENCV_FISHEYE", "FULL_OPENCV"]
"""The camera models the benchmark covers: the four CASPAR projects."""

IMAGE_SIZE_PX: Final[tuple[int, int]] = (1280, 960)
FOCAL_PX: Final[float] = 900.0

DISTORTION: Final[dict[SynthCameraModel, tuple[float, ...]]] = {
    "PINHOLE": (),
    "SIMPLE_RADIAL": (-0.05,),
    "OPENCV_FISHEYE": (0.05, -0.01, 0.002, -0.0005),
    "FULL_OPENCV": (-0.1, 0.02, 1e-4, -1e-4, -0.001, 0.01, 0.002, 0.001),
}
"""Non-zero coefficients in COLMAP's order; small enough that no visible ray reaches a rational pole."""

SECOND_SENSOR_ROTATION_DEG: Final[float] = 10.0
"""Yaw of the second rig camera: a non-identity `sensor_from_rig`."""
SECOND_SENSOR_TRANSLATION_M: Final[tuple[float, float, float]] = (0.3, 0.0, 0.0)


def _camera(model: SynthCameraModel, camera_id: int) -> pycolmap.Camera:
    width, height = IMAGE_SIZE_PX
    cx, cy = width / 2.0, height / 2.0
    focals: list[float] = [FOCAL_PX] if model == "SIMPLE_RADIAL" else [FOCAL_PX, FOCAL_PX]
    params: list[float] = [*focals, cx, cy, *DISTORTION[model]]
    return pycolmap.Camera(camera_id=camera_id, model=getattr(pycolmap.CameraModelId, model), width=width, height=height, params=params)


def _yaw(degrees: float) -> Float64[ndarray, "3 3"]:
    a: float = np.deg2rad(degrees)
    return np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])


def synthetic_reconstruction(
    num_frames: int, num_points: int, model: SynthCameraModel = "PINHOLE", num_cameras: int = 1, seed: int = 0
) -> pycolmap.Reconstruction:
    """Build the perturbed benchmark model in memory.

    Args:
        num_frames: Rig frames on the circle.
        num_points: Points drawn in the box; those seen fewer than twice are dropped.
        model: Camera model of every camera.
        num_cameras: 1, or 2 for a rig whose second camera has a non-identity `sensor_from_rig`.
        seed: Seed of every draw.

    Returns:
        The model, every frame registered.

    Raises:
        ValueError: When `num_cameras` is not 1 or 2.
    """
    if num_cameras not in (1, 2):
        raise ValueError(f"num_cameras must be 1 or 2, got {num_cameras}")
    rng: np.random.Generator = np.random.default_rng(seed)
    width, height = IMAGE_SIZE_PX
    reconstruction = pycolmap.Reconstruction()
    rig = pycolmap.Rig()
    rig.rig_id = 1
    for camera_id in range(1, num_cameras + 1):
        reconstruction.add_camera(_camera(model, camera_id))
    rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 1))
    if num_cameras == 2:
        second_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(_yaw(SECOND_SENSOR_ROTATION_DEG)), np.array(SECOND_SENSOR_TRANSLATION_M))
        rig.add_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 2), second_from_rig)
    reconstruction.add_rig(rig)
    points: Float64[ndarray, "n 3"] = rng.uniform(-4.0, 4.0, (num_points, 3)) * np.array([1.0, 0.5, 1.0])
    for index in range(num_frames):
        angle: float = 2 * np.pi * index / num_frames
        center = np.array([12.0 * np.cos(angle), 0.3 * np.sin(5 * angle), 12.0 * np.sin(angle)])
        forward = -center / np.linalg.norm(center)
        right = np.cross([0.0, 1.0, 0.0], forward)
        right /= np.linalg.norm(right)
        rotation = np.stack([right, np.cross(forward, right), forward])
        frame = pycolmap.Frame()
        frame.frame_id = index + 1
        frame.rig_id = 1
        frame.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), -rotation @ center)
        for camera_id in range(1, num_cameras + 1):
            frame.add_data_id(pycolmap.data_t(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id), num_cameras * index + camera_id))
        reconstruction.add_frame(frame)
        for camera_id in range(1, num_cameras + 1):
            image_id: int = num_cameras * index + camera_id
            reconstruction.add_image(pycolmap.Image(name=f"{image_id:05d}.png", camera_id=camera_id, image_id=image_id, frame_id=index + 1))
        reconstruction.register_frame(frame.frame_id)
    tracks: dict[int, list[tuple[int, int]]] = {}
    for image_id in sorted(reconstruction.images):
        image = reconstruction.image(image_id)
        camera = reconstruction.camera(image.camera_id)
        pose = image.cam_from_world()
        in_cam = points @ np.asarray(pose.rotation.matrix()).T + np.asarray(pose.translation)
        xy = np.asarray(camera.img_from_cam(in_cam)) + rng.normal(0.0, 0.5, (num_points, 2))
        inside = (in_cam[:, 2] > 0.5) & (xy >= 0).all(1) & (xy < [width, height]).all(1)
        visible = np.flatnonzero(inside & (rng.random(num_points) < 0.15))
        image.points2D = pycolmap.Point2DList([pycolmap.Point2D(xy[i]) for i in visible])
        for k, i in enumerate(visible):
            tracks.setdefault(int(i), []).append((image_id, k))
    for i, elements in tracks.items():
        if len(elements) < 2:
            continue
        track = pycolmap.Track([pycolmap.TrackElement(a, b) for a, b in elements])
        point3D_id: int = reconstruction.add_point3D(points[i] + rng.normal(0, 0.05, 3), track, np.zeros(3, np.uint8))
        for a, b in elements:
            reconstruction.image(a).set_point3D_for_point2D(b, point3D_id)
    for frame_id in list(reconstruction.frames)[1:]:
        frame = reconstruction.frame(frame_id)
        pose: pycolmap.Rigid3d | None = frame.rig_from_world
        assert pose is not None, "every synthetic frame is posed"
        frame.rig_from_world = pycolmap.Rigid3d(pose.rotation, np.asarray(pose.translation) + rng.normal(0, 0.05, 3))
    if num_cameras == 2:
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 2)
        true_from_rig: pycolmap.Rigid3d | None = reconstruction.rig(1).sensor_from_rig(sensor)
        assert true_from_rig is not None, "the second sensor was added with a pose"
        knocked = pycolmap.Rotation3d(_yaw(0.5) @ np.asarray(true_from_rig.rotation.matrix()))
        reconstruction.rig(1).set_sensor_from_rig(sensor, pycolmap.Rigid3d(knocked, np.asarray(true_from_rig.translation) + rng.normal(0, 0.02, 3)))
    return reconstruction


@dataclass(frozen=True)
class Config:
    """Where the model goes and what it is."""

    output_dir: Path
    """COLMAP model directory to write (created)."""
    num_frames: int = 400
    num_points: int = 40000
    model: SynthCameraModel = "PINHOLE"
    num_cameras: int = 1
    """2 for a rig with a non-identity second camera."""
    seed: int = 0


def main(config: Config) -> None:
    """Build and write the model, then print its summary.

    Args:
        config: Size, camera model and destination.
    """
    reconstruction: pycolmap.Reconstruction = synthetic_reconstruction(
        config.num_frames, config.num_points, config.model, config.num_cameras, config.seed
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(config.output_dir)
    reconstruction.update_point_3d_errors()
    print(reconstruction.summary())
    print(f"mean reprojection error {reconstruction.compute_mean_reprojection_error():.4f} px")
