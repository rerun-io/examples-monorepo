"""The synthetic fisheye rig `test_calibrate` and `test_calibration_check` calibrate: three cameras with known lenses and
poses, a room of points, and a COLMAP database that holds only the seed cameras, the keypoints and the verified matches."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from colsfm.calibrate import rig_window_pairs, seed_camera

WIDTH: int = 640
HEIGHT: int = 480
NAMES: tuple[str, ...] = ("front", "left", "right")
TRUE_PARAMS: tuple[list[float], ...] = (
    [210.0, 209.0, 322.0, 238.0, 0.02, 0.005, -0.002, 0.0005],
    [206.0, 206.5, 318.0, 243.0, 0.03, -0.004, 0.001, 0.0],
    [213.0, 212.0, 325.0, 236.0, 0.01, 0.008, -0.003, 0.0008],
)
"""fx fy cx cy k1..k4 per camera; the seed (640 / pi = 203.7 px, centred, no distortion) is a few percent off, as on RoboCap."""


def _rig_T_cam(yaw_deg: float, translation: list[float]) -> np.ndarray:
    pose: np.ndarray = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    pose[:3, 3] = translation
    return pose


TRUE_RIG_T_CAM: tuple[np.ndarray, ...] = (
    _rig_T_cam(0.0, [0.0, 0.0, 0.0]),
    _rig_T_cam(-70.0, [-0.08, 0.01, -0.05]),
    _rig_T_cam(70.0, [0.09, -0.01, -0.06]),
)
"""Front, and two side cameras looking 70 degrees out (camera frame: x right, y down, z forward)."""


def synthetic_database(path: Path, num_rig_frames: int = 24, overlap: int = 3, num_points: int = 1500) -> None:
    """A room of points seen by the true rig along a turning walk; the database holds the seeds, not the truth."""
    rng: np.random.Generator = np.random.default_rng(7)
    walls: np.ndarray = rng.uniform([-4.0, -1.5, -4.0], [4.0, 1.5, 4.0], size=(num_points, 3))
    side: np.ndarray = rng.integers(0, 4, size=len(walls))
    walls[side == 0, 0], walls[side == 1, 0], walls[side == 2, 2], walls[side == 3, 2] = -4.0, 4.0, -4.0, 4.0
    true_cameras: list[pycolmap.Camera] = [
        pycolmap.Camera(model="OPENCV_FISHEYE", width=WIDTH, height=HEIGHT, params=params) for params in TRUE_PARAMS
    ]
    with pycolmap.Database.open(path) as database:
        for camera_id in range(1, len(NAMES) + 1):
            database.write_camera(seed_camera(camera_id, "OPENCV_FISHEYE", WIDTH, HEIGHT), True)
        image_ids: list[list[int | None]] = []
        observed: dict[int, dict[int, int]] = {}  # image id -> point index -> keypoint index
        for frame in range(num_rig_frames):
            world_T_rig: np.ndarray = np.eye(4)
            world_T_rig[:3, :3] = Rotation.from_euler("yx", [3.0 * frame, 8.0 * math.sin(frame / 5.0)], degrees=True).as_matrix()
            world_T_rig[:3, 3] = [0.04 * frame - 0.8, 0.0, 0.5 * math.sin(frame / 7.0)]
            row: list[int | None] = []
            for camera_index, (name, camera) in enumerate(zip(NAMES, true_cameras, strict=True)):
                cam_T_world: np.ndarray = np.linalg.inv(world_T_rig @ TRUE_RIG_T_CAM[camera_index])
                in_camera: np.ndarray = walls @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
                ahead: np.ndarray = np.arccos(np.clip(in_camera[:, 2] / np.linalg.norm(in_camera, axis=1), -1.0, 1.0)) < np.radians(85.0)
                pixels: np.ndarray = np.full((len(walls), 2), -1.0)
                pixels[ahead] = camera.img_from_cam(in_camera[ahead])
                visible: np.ndarray = np.flatnonzero(
                    ahead & (pixels[:, 0] >= 0) & (pixels[:, 0] < WIDTH) & (pixels[:, 1] >= 0) & (pixels[:, 1] < HEIGHT)
                )
                image_id: int = database.write_image(pycolmap.Image(name=f"{name}/{frame:06d}", camera_id=camera_index + 1))
                keypoints: np.ndarray = (pixels[visible] + rng.normal(0.0, 0.3, size=(len(visible), 2))).astype(np.float32)
                database.write_keypoints(image_id, keypoints)
                observed[image_id] = {int(point): index for index, point in enumerate(visible)}
                row.append(image_id)
            image_ids.append(row)
        seeds: dict[int, pycolmap.Camera] = {camera.camera_id: camera for camera in database.read_all_cameras()}
        camera_of: dict[int, int] = {image.image_id: image.camera_id for image in database.read_all_images()}
        verification: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
        for first, second in rig_window_pairs(image_ids, overlap):
            common: list[int] = sorted(observed[first].keys() & observed[second].keys())
            if len(common) < 30:
                continue
            matches: np.ndarray = np.asarray([[observed[first][point], observed[second][point]] for point in common], dtype=np.uint32)
            database.write_matches(first, second, matches)
            geometry: pycolmap.TwoViewGeometry = pycolmap.estimate_two_view_geometry(
                seeds[camera_of[first]],
                database.read_keypoints(first)[:, :2].astype(np.float64),
                seeds[camera_of[second]],
                database.read_keypoints(second)[:, :2].astype(np.float64),
                matches,
                verification,
            )
            database.write_two_view_geometry(first, second, geometry)
