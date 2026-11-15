"""A synthetic rig sequence shared by the pycolmap probes.

The scene is deliberately tiny and fully determined: a grid of 3D points in
front of a two-camera rig that translates and yaws over a handful of frames.
Because every observation is an exact projection, any residual a probe measures
comes from pycolmap, not from the data.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Float, Int, UInt32
from numpy import ndarray
from PIL import Image as PILImage

ImageId: TypeAlias = int
"""COLMAP image identifier (1-based, unique within a database)."""

CameraId: TypeAlias = int
"""COLMAP camera identifier; here 1 = rig reference camera, 2 = second camera."""

IMAGE_WIDTH: int = 640
"""Width in pixels of every synthetic camera."""

IMAGE_HEIGHT: int = 480
"""Height in pixels of every synthetic camera."""

FOCAL_LENGTH_PX: float = 500.0
"""Focal length in pixels of every synthetic camera."""

RIG_ID: int = 1
"""Identifier of the single rig in the synthetic scene."""

REF_CAMERA_ID: CameraId = 1
"""Camera that is the rig reference sensor, i.e. `cam_from_rig` is identity."""

SECOND_CAMERA_ID: CameraId = 2
"""Second rig camera, offset along +X in rig space."""

STEREO_BASELINE_M: float = 0.25
"""Distance in metres between the two rig cameras."""


@dataclass(slots=True)
class SyntheticScene:
    """A reconstruction with known poses plus the observations that generated it."""

    reconstruction: pycolmap.Reconstruction
    """Cameras, rig, frames and images with exact `rig_from_world` poses set."""
    points_xyz: Float[ndarray, "n_points 3"]
    """Ground-truth 3D point positions in world space, in metres."""
    keypoints_px: dict[ImageId, Float[ndarray, "n_obs 2"]]
    """Exact pixel projections per image, ordered as written to the database."""
    point_indices: dict[ImageId, Int[ndarray, " n_obs"]]
    """For each image, the index into `points_xyz` of every keypoint."""
    image_names: dict[ImageId, str] = field(default_factory=dict)
    """Image file names, matching what `Reconstruction.write_text` will emit."""


def _rig_from_world_at(frame_index: int) -> pycolmap.Rigid3d:
    """Pose of the rig at a frame: a straight slide along -X with a small yaw.

    Args:
        frame_index: Zero-based index of the frame along the trajectory.

    Returns:
        The `rig_from_world` transform for that frame.
    """
    yaw_rad: float = float(np.deg2rad(4.0 * frame_index))
    rig_R_world: Float[ndarray, "3 3"] = np.array(
        [
            [np.cos(yaw_rad), 0.0, np.sin(yaw_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw_rad), 0.0, np.cos(yaw_rad)],
        ],
        dtype=np.float64,
    )
    rig_t_world: Float[ndarray, "3"] = np.array([-0.4 * frame_index, 0.0, 0.0], dtype=np.float64)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rig_R_world), rig_t_world)


def _make_points(seed: int) -> Float[ndarray, "n_points 3"]:
    """Build a jittered 6x6x2 grid of world points in front of the rig.

    Args:
        seed: PRNG seed for the jitter, so probes are reproducible.

    Returns:
        World-space point positions in metres.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    xs: Float[ndarray, "6"] = np.linspace(-2.0, 2.0, 6)
    ys: Float[ndarray, "6"] = np.linspace(-1.5, 1.5, 6)
    zs: Float[ndarray, "2"] = np.array([5.0, 7.0])
    grid: Float[ndarray, "n_points 3"] = np.array(list(itertools.product(xs, ys, zs)), dtype=np.float64)
    return grid + rng.normal(0.0, 0.05, grid.shape)


def make_pinhole_camera(camera_id: CameraId) -> pycolmap.Camera:
    """Create a PINHOLE camera with the suite's canonical intrinsics.

    Args:
        camera_id: Identifier to assign to the camera.

    Returns:
        A `pycolmap.Camera` with params `[fx, fy, cx, cy]`.
    """
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_id(
        camera_id, pycolmap.CameraModelId.PINHOLE, FOCAL_LENGTH_PX, IMAGE_WIDTH, IMAGE_HEIGHT
    )
    camera.params = [FOCAL_LENGTH_PX, FOCAL_LENGTH_PX, IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0]
    return camera


def build_scene(num_frames: int = 5, seed: int = 0) -> SyntheticScene:
    """Build the two-camera rig sequence with exact projections.

    Args:
        num_frames: Number of rig frames along the trajectory.
        seed: PRNG seed for the point-cloud jitter.

    Returns:
        The populated `SyntheticScene`.
    """
    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    for camera_id in (REF_CAMERA_ID, SECOND_CAMERA_ID):
        reconstruction.add_camera(make_pinhole_camera(camera_id))

    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, REF_CAMERA_ID))
    rig.add_sensor(
        pycolmap.sensor_t(pycolmap.SensorType.CAMERA, SECOND_CAMERA_ID),
        pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([-STEREO_BASELINE_M, 0.0, 0.0])),
    )
    reconstruction.add_rig(rig)

    names: dict[ImageId, str] = {}
    next_image_id: ImageId = 0
    for frame_index in range(num_frames):
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = frame_index + 1
        frame.rig_id = RIG_ID
        frame.rig_from_world = _rig_from_world_at(frame_index)
        pending: list[tuple[ImageId, CameraId]] = []
        for camera_id in (REF_CAMERA_ID, SECOND_CAMERA_ID):
            next_image_id += 1
            # A frame must own its data ids before Reconstruction.add_image
            # accepts the corresponding image, otherwise COLMAP aborts with
            # "Check failed: frame.HasDataId(image.DataId())".
            frame.add_data_id(
                pycolmap.data_t(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id), next_image_id)
            )
            pending.append((next_image_id, camera_id))
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)
        for image_id, camera_id in pending:
            names[image_id] = f"frame{frame_index:02d}_cam{camera_id}.png"
            image: pycolmap.Image = pycolmap.Image(
                name=names[image_id], camera_id=camera_id, image_id=image_id
            )
            image.frame_id = frame.frame_id
            reconstruction.add_image(image)

    points_xyz: Float[ndarray, "n_points 3"] = _make_points(seed)
    keypoints: dict[ImageId, Float[ndarray, "n_obs 2"]] = {}
    indices: dict[ImageId, Int[ndarray, " n_obs"]] = {}
    # Projection is batched through `Camera.img_from_cam`, not looped through
    # `Image.project_point`.  The per-point call is ~1000x slower under the
    # beartype claw and leaks ~15 objects per invocation, which turns a
    # per-test fixture into a session-long slowdown.
    for image_id in sorted(reconstruction.images):
        image = reconstruction.image(image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        points_in_cam: Float[ndarray, "n_points 3"] = (
            points_xyz @ cam_from_world.rotation.matrix().T + cam_from_world.translation
        )
        projected: Float[ndarray, "n_points 2"] = camera.img_from_cam(points_in_cam)
        in_frame: np.ndarray = (
            np.isfinite(projected).all(axis=1)
            & (points_in_cam[:, 2] > 0.0)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < IMAGE_WIDTH)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < IMAGE_HEIGHT)
        )
        keypoints[image_id] = projected[in_frame]
        indices[image_id] = np.flatnonzero(in_frame).astype(np.int64)
        image.points2D = pycolmap.Point2DList(
            [pycolmap.Point2D(xy) for xy in keypoints[image_id]]
        )

    return SyntheticScene(
        reconstruction=reconstruction,
        points_xyz=points_xyz,
        keypoints_px=keypoints,
        point_indices=indices,
        image_names=names,
    )


def matches_between(
    scene: SyntheticScene, image_id1: ImageId, image_id2: ImageId
) -> UInt32[ndarray, "n_matches 2"]:
    """Ground-truth matches between two images of the scene.

    Args:
        scene: The synthetic scene.
        image_id1: First image identifier.
        image_id2: Second image identifier.

    Returns:
        An Nx2 uint32 array of point2D indices, as `Database.write_matches` wants.
    """
    left: Int[ndarray, " n_obs"] = scene.point_indices[image_id1]
    right: Int[ndarray, " n_obs"] = scene.point_indices[image_id2]
    common: Int[ndarray, " n_common"] = np.intersect1d(left, right)
    return np.stack(
        [np.searchsorted(left, common), np.searchsorted(right, common)], axis=1
    ).astype(np.uint32)


def write_database(scene: SyntheticScene, database_path: Path, min_matches: int = 20) -> int:
    """Write the scene's cameras, rig, frames, images, keypoints and matches to SQLite.

    Args:
        scene: The synthetic scene.
        database_path: Where to create the COLMAP database.
        min_matches: Skip image pairs with fewer than this many shared points.

    Returns:
        The number of image pairs written.
    """
    database: pycolmap.Database = pycolmap.Database.open(database_path)
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    for camera_id in sorted(reconstruction.cameras):
        database.write_camera(reconstruction.camera(camera_id), use_camera_id=True)
    database.write_rig(reconstruction.rig(RIG_ID), use_rig_id=True)
    for frame_id in sorted(reconstruction.frames):
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = frame_id
        frame.rig_id = RIG_ID
        for data_id in reconstruction.frame(frame_id).data_ids:
            frame.add_data_id(data_id)
        database.write_frame(frame, use_frame_id=True)
    for image_id in sorted(reconstruction.images):
        source: pycolmap.Image = reconstruction.image(image_id)
        row: pycolmap.Image = pycolmap.Image(
            name=source.name, camera_id=source.camera_id, image_id=image_id
        )
        row.frame_id = source.frame_id
        database.write_image(row, use_image_id=True)
        database.write_keypoints(image_id, scene.keypoints_px[image_id].astype(np.float32))

    num_pairs: int = 0
    for image_id1, image_id2 in itertools.combinations(sorted(reconstruction.images), 2):
        matches: UInt32[ndarray, "n_matches 2"] = matches_between(scene, image_id1, image_id2)
        if len(matches) < min_matches:
            continue
        database.write_matches(image_id1, image_id2, matches)
        geometry: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
        geometry.config = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
        geometry.inlier_matches = matches
        database.write_two_view_geometry(image_id1, image_id2, geometry)
        num_pairs += 1
    database.close()
    return num_pairs


def write_placeholder_images(scene: SyntheticScene, image_dir: Path) -> None:
    """Write flat grey PNGs so `triangulate_points` can extract point colours.

    Args:
        scene: The synthetic scene.
        image_dir: Directory to create the placeholder images in.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    grey: np.ndarray = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 128, dtype=np.uint8)
    for name in scene.image_names.values():
        PILImage.fromarray(grey).save(image_dir / name)


def add_ground_truth_points(scene: SyntheticScene) -> list[int]:
    """Add every scene point to the reconstruction with its exact track.

    This bypasses triangulation so that a bundle-adjustment probe starts from a
    known-perfect model and only the perturbation under test moves.

    Args:
        scene: The synthetic scene, modified in place.

    Returns:
        The point3D ids in the order of `scene.points_xyz`.
    """
    reconstruction: pycolmap.Reconstruction = scene.reconstruction
    tracks: dict[int, list[pycolmap.TrackElement]] = {
        index: [] for index in range(len(scene.points_xyz))
    }
    for image_id in sorted(reconstruction.images):
        for observation_index, point_index in enumerate(scene.point_indices[image_id]):
            tracks[int(point_index)].append(pycolmap.TrackElement(image_id, observation_index))

    point3D_ids: list[int] = []
    for point_index, point_xyz in enumerate(scene.points_xyz):
        elements: list[pycolmap.TrackElement] = tracks[point_index]
        point3D_id: int = reconstruction.add_point3D(
            point_xyz, pycolmap.Track(elements), np.array([200, 100, 50], dtype=np.uint8)
        )
        for element in elements:
            reconstruction.image(element.image_id).set_point3D_for_point2D(
                element.point2D_idx, point3D_id
            )
        point3D_ids.append(point3D_id)
    return point3D_ids
