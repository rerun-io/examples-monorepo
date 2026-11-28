"""Synthetic bundle-adjustment problems: frames on a circle around a point cloud (as ba_synth.py), any camera model.

Built in memory from a seed, so no data file is written or read.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from caspar_mojo.problem import MAX_INTRINSICS, BaProblem, CameraModel
from caspar_mojo.reference.cameras import project
from caspar_mojo.reference.poses import camera_point, normalize_pose, retract_pose

WIDTH: int = 1280
HEIGHT: int = 960


def default_intrinsics(model: CameraModel) -> Float64[ndarray, "12"]:
    """Realistic `camera.params` for a 1280 x 960 image, with non-zero distortion where the model has it."""
    values: dict[CameraModel, list[float]] = {
        CameraModel.PINHOLE: [900.0, 905.0, 640.0, 480.0],
        CameraModel.SIMPLE_RADIAL: [900.0, 640.0, 480.0, -0.05],
        CameraModel.OPENCV_FISHEYE: [450.0, 452.0, 640.0, 480.0, 0.02, -0.01, 0.003, -0.0005],
        CameraModel.FULL_OPENCV: [900.0, 905.0, 640.0, 480.0, -0.12, 0.05, 1e-3, -5e-4, 0.01, 0.02, 0.01, 0.005],
    }
    row: Float64[ndarray, "12"] = np.zeros(MAX_INTRINSICS)
    row[: len(values[model])] = values[model]
    return row


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The noise-free state the observations were rendered from."""

    intrinsics: Float64[ndarray, "n_cameras 12"]
    """True intrinsics."""
    sensor_from_rig: Float64[ndarray, "n_cameras 7"]
    """True rig extrinsics."""
    rig_from_world: Float64[ndarray, "n_frames 7"]
    """True frame poses."""
    points: Float64[ndarray, "n_points 3"]
    """True world points."""


def _look_at_origin(center: Float64[ndarray, "3"]) -> Float64[ndarray, "3 3"]:
    """cam_R_world of a camera at `center` looking at the origin with world +y up."""
    forward = -center / np.linalg.norm(center)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(forward, right), forward])


def _quaternion_of(matrix: Float64[ndarray, "3 3"]) -> Float64[ndarray, "4"]:
    """[qx, qy, qz, qw] of a rotation matrix (Shepperd, the branch with the largest pivot)."""
    trace = np.trace(matrix)
    pivots = [trace, matrix[0, 0], matrix[1, 1], matrix[2, 2]]
    i = int(np.argmax(pivots))
    if i == 0:
        w = np.sqrt(1.0 + trace) / 2
        q = [(matrix[2, 1] - matrix[1, 2]) / (4 * w), (matrix[0, 2] - matrix[2, 0]) / (4 * w), (matrix[1, 0] - matrix[0, 1]) / (4 * w), w]
    elif i == 1:
        x = np.sqrt(1.0 + 2 * matrix[0, 0] - trace) / 2
        q = [x, (matrix[0, 1] + matrix[1, 0]) / (4 * x), (matrix[0, 2] + matrix[2, 0]) / (4 * x), (matrix[2, 1] - matrix[1, 2]) / (4 * x)]
    elif i == 2:
        y = np.sqrt(1.0 + 2 * matrix[1, 1] - trace) / 2
        q = [(matrix[0, 1] + matrix[1, 0]) / (4 * y), y, (matrix[1, 2] + matrix[2, 1]) / (4 * y), (matrix[0, 2] - matrix[2, 0]) / (4 * y)]
    else:
        z = np.sqrt(1.0 + 2 * matrix[2, 2] - trace) / 2
        q = [(matrix[0, 2] + matrix[2, 0]) / (4 * z), (matrix[1, 2] + matrix[2, 1]) / (4 * z), z, (matrix[1, 0] - matrix[0, 1]) / (4 * z)]
    return np.asarray(q)


def _rotate_small(pose: Float64[ndarray, "n 7"], sigma: float, rng: np.random.Generator) -> Float64[ndarray, "n 7"]:
    """Right-perturb each rotation by a random rotation vector of standard deviation `sigma` (rad)."""
    tangent = np.zeros((len(pose), 6))
    tangent[:, :3] = rng.normal(0.0, sigma, (len(pose), 3))
    return normalize_pose(retract_pose(pose, tangent))


def make_problem(
    model: CameraModel,
    n_frames: int,
    n_points: int,
    seed: int,
    rig_cameras: int = 1,
    noise_px: float = 0.5,
    pose_noise: float = 0.05,
    rotation_noise: float = 0.002,
    point_noise: float = 0.05,
    focal_noise: float = 0.0,
    intrinsics_free: bool = False,
    sensor_free: bool = False,
    visible_fraction: float = 0.5,
) -> tuple[BaProblem, GroundTruth]:
    """Frames on a circle of radius 12 around a 8 x 4 x 8 point cloud, as ba_synth.py, observed with `model`.

    Args:
        model: Camera model of every camera.
        n_frames: Frames (rig poses) on the circle; frame 0 is fixed (the gauge).
        n_points: Points drawn; those seen fewer than twice are dropped.
        seed: Seed of every random draw.
        rig_cameras: Cameras per rig; camera 0 is the reference sensor, the others are yawed and shifted.
        noise_px: Standard deviation of the pixel noise.
        pose_noise: Standard deviation of the start's translation error of each free frame.
        rotation_noise: Standard deviation (rad) of the start's rotation error of each free frame.
        point_noise: Standard deviation of the start's point error.
        focal_noise: Relative standard deviation of the start's focal error when `intrinsics_free`.
        intrinsics_free: Refine every camera's focal and extra parameters.
        sensor_free: Refine every non-reference camera's sensor_from_rig.
        visible_fraction: Probability that an in-image point is observed.

    Returns:
        The perturbed problem and the state its pixels were rendered from.
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(-4.0, 4.0, (n_points, 3)) * np.array([1.0, 0.5, 1.0])
    rig_from_world = np.empty((n_frames, 7))
    for index in range(n_frames):
        angle = 2 * np.pi * index / n_frames
        center = np.array([12.0 * np.cos(angle), 0.3 * np.sin(5 * angle), 12.0 * np.sin(angle)])
        rotation = _look_at_origin(center)
        rig_from_world[index] = np.concatenate([_quaternion_of(rotation), -rotation @ center])
    sensor_from_rig = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (rig_cameras, 1))
    for camera in range(1, rig_cameras):
        yaw = 0.35 * (1 if camera % 2 else -1) * ((camera + 1) // 2)
        rotation = np.array([[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]])
        sensor_from_rig[camera] = np.concatenate([_quaternion_of(rotation), [0.15 * camera, -0.05 * camera, 0.02]])
    intrinsics = np.tile(default_intrinsics(model), (rig_cameras, 1))
    camera_models = np.full(rig_cameras, int(model), dtype=np.int32)

    frames, cameras, point_ids, pixels = [], [], [], []
    for frame in range(n_frames):
        for camera in range(rig_cameras):
            in_camera = camera_point(
                np.repeat(rig_from_world[frame : frame + 1], n_points, 0), np.repeat(sensor_from_rig[camera : camera + 1], n_points, 0), points
            ).point_cam
            pixel = project(model, in_camera, np.repeat(intrinsics[camera : camera + 1], n_points, 0)).pixel
            pixel = pixel + rng.normal(0.0, noise_px, pixel.shape)
            seen = (in_camera[:, 2] > 0.5) & (pixel >= 0).all(1) & (pixel < [WIDTH, HEIGHT]).all(1) & (rng.random(n_points) < visible_fraction)
            ids = np.flatnonzero(seen)
            frames.append(np.full(len(ids), frame))
            cameras.append(np.full(len(ids), camera))
            point_ids.append(ids)
            pixels.append(pixel[ids])
    observation_point = np.concatenate(point_ids)
    keep_point = np.bincount(observation_point, minlength=n_points) >= 2
    new_index = np.cumsum(keep_point) - 1
    keep = keep_point[observation_point]
    truth = GroundTruth(intrinsics, sensor_from_rig, rig_from_world, points[keep_point])

    pose_free = np.arange(n_frames) > 0
    start_rig = rig_from_world.copy()
    start_rig[pose_free] = _rotate_small(start_rig[pose_free], rotation_noise, rng)
    start_rig[pose_free, 4:] += rng.normal(0.0, pose_noise, (int(pose_free.sum()), 3))
    sensor_free_mask = np.zeros(rig_cameras, dtype=bool)
    sensor_free_mask[1:] = sensor_free
    start_sensor = sensor_from_rig.copy()
    start_sensor[sensor_free_mask, 4:] += rng.normal(0.0, 0.01, (int(sensor_free_mask.sum()), 3))
    start_intrinsics = intrinsics.copy()
    if intrinsics_free:
        focal_columns = [0] if model == CameraModel.SIMPLE_RADIAL else [0, 1]
        start_intrinsics[:, focal_columns] *= 1.0 + rng.normal(0.0, focal_noise, (rig_cameras, 1))
    start_points = truth.points + rng.normal(0.0, point_noise, truth.points.shape)
    problem = BaProblem(
        camera_model=camera_models,
        intrinsics=start_intrinsics,
        intrinsics_free=np.full(rig_cameras, intrinsics_free),
        sensor_from_rig=start_sensor,
        sensor_free=sensor_free_mask,
        rig_from_world=start_rig,
        pose_free=pose_free,
        points=start_points,
        point_free=np.ones(len(start_points), dtype=bool),
        pixels=np.concatenate(pixels)[keep],
        observation_frame=np.concatenate(frames)[keep].astype(np.int32),
        observation_camera=np.concatenate(cameras)[keep].astype(np.int32),
        observation_point=new_index[observation_point[keep]].astype(np.int32),
    )
    return problem, truth
