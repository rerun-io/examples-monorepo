"""The synthetic rig every mapping test scene is built from.

`test_mapping.py` was one 1 987-line file covering five unrelated concerns. The
scaffolding they all share — a two-camera rig with known poses, known points and
noisy observations, its COLMAP database, and the bookkeeping that matches a
triangulated track back to the point that generated it — lives here, so the five
test modules that split out of it hold only what each of them asserts.

What is shared with *every* synthetic-scene test — the projection and its
in-image mask — stays in `conftest.py`; this module only adds what a rig scene
needs on top of it. The `synthetic_rig` and `synthetic_database` fixtures over it
live in `conftest.py` too, declared once instead of copied into each of the four
modules that ask for them; they stay **module** scoped, because building the rig
costs 8 ms and its database 39 ms, so a per-module scene is cheaper than a shared
one is worth.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from conftest import project_and_mask
from google.protobuf.message import Message
from jaxtyping import Bool, Float64, Int, UInt32
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm.ba_backend import BaExecutionPlan, CasparCapability, resolve_ba_plan
from colsfm.cameras import colmap_cameras
from colsfm.config import VisionMappingConfig
from colsfm.frames_meta import FramesMeta, parse_message, write_rigid_transform
from colsfm.mapping import MappingOptions
from colsfm.reconstruction import build_reconstruction
from colsfm.schema import KEYFRAMES_METADATA_COLLECTION, load_schema

IMAGE_WIDTH: int = 640
"""Width in pixels of every synthetic camera."""

IMAGE_HEIGHT: int = 480
"""Height in pixels of every synthetic camera."""

FOCAL_LENGTH_PX: float = 500.0
"""Focal length in pixels of every synthetic camera."""

PERTURBATION_M: float = 0.01
"""Translation the pose-recovery test knocks every non-gauge rig frame off by."""

PERTURBATION_DEG: float = 0.5
"""Rotation the pose-recovery test knocks every non-gauge rig frame off by."""

SYNTHETIC_NOISE_PX: float = 0.3
"""Standard deviation of the Gaussian noise added to every synthetic observation."""

MULTIVIEW_OVERSAMPLING: int = 8
"""How many candidate points to draw per kept point, before the visibility filter."""

VEHICLE_R_CAM: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""Rotation from the camera's RDF axes to the vehicle's FLU axes (docs/camera.md)."""


@dataclass(slots=True)
class SyntheticRig:
    """A rig sequence with known poses, known points and noisy observations."""

    frames_meta: FramesMeta
    """Metadata carrying the true (or perturbed) poses, as the mapper would read it."""
    points_xyz: Float64[ndarray, "num_points 3"]
    """Ground-truth world points in metres."""
    keypoints_px: dict[int, Float64[ndarray, "num_observations 2"]]
    """Noisy pixel observations per keyframe id, in write order."""
    point_indices: dict[int, Int[ndarray, " num_observations"]]
    """Index into `points_xyz` of every observation, per keyframe id."""
    world_T_vehicle: dict[int, pycolmap.Rigid3d]
    """The true vehicle pose per `synced_sample_id`, before any perturbation."""


def _pinhole_projection(camera_matrix: Float64[ndarray, "3 3"]) -> list[float]:
    """A 3x4 projection matrix, row-major, from a 3x3 intrinsic matrix.

    Args:
        camera_matrix: The intrinsics.

    Returns:
        The twelve row-major entries cuSFM stores in `projection_matrix`.
    """
    projection: Float64[ndarray, "3 4"] = np.hstack([camera_matrix, np.zeros((3, 1))])
    return [float(value) for value in projection.reshape(-1)]


def _camera_matrix() -> Float64[ndarray, "3 3"]:
    """The synthetic cameras' shared intrinsic matrix.

    Returns:
        A 3x3 intrinsic matrix in pixels.
    """
    return np.array(
        [[FOCAL_LENGTH_PX, 0.0, IMAGE_WIDTH / 2.0], [0.0, FOCAL_LENGTH_PX, IMAGE_HEIGHT / 2.0], [0.0, 0.0, 1.0]]
    )


def _fill_camera(
    message: Message,
    camera_params_id: int,
    vehicle_T_cam: pycolmap.Rigid3d,
    projection_model: str,
    fisheye: bool,
) -> None:
    """Fill one `CameraSensor` entry of a synthetic collection.

    Args:
        message: The `protos.common.sensor.CameraSensor` to fill.
        camera_params_id: Its key in the map.
        vehicle_T_cam: The rig extrinsic, camera to vehicle.
        projection_model: Name of the `CameraProjectionModelType` value.
        fisheye: Write `camera_matrix` plus four distortion coefficients rather
            than a rectified `projection_matrix`.
    """
    message.sensor_meta_data.sensor_id = camera_params_id
    message.sensor_meta_data.sensor_name = f"camera_{camera_params_id}"
    message.sensor_meta_data.frequency = 30.0
    write_rigid_transform(message.sensor_meta_data.sensor_to_vehicle_transform, vehicle_T_cam)
    message.calibration_parameters.image_width = IMAGE_WIDTH
    message.calibration_parameters.image_height = IMAGE_HEIGHT
    if fisheye:
        message.calibration_parameters.camera_matrix.row_count = 3
        message.calibration_parameters.camera_matrix.column_count = 3
        message.calibration_parameters.camera_matrix.data.extend(
            [float(value) for value in _camera_matrix().reshape(-1)]
        )
        message.calibration_parameters.distortion_coefficients.row_count = 1
        message.calibration_parameters.distortion_coefficients.column_count = 4
        message.calibration_parameters.distortion_coefficients.data.extend([0.02, -0.005, 0.001, 0.0])
    else:
        message.calibration_parameters.projection_matrix.row_count = 3
        message.calibration_parameters.projection_matrix.column_count = 4
        message.calibration_parameters.projection_matrix.data.extend(_pinhole_projection(_camera_matrix()))
    message.camera_projection_model_type = message.DESCRIPTOR.fields_by_name[
        "camera_projection_model_type"
    ].enum_type.values_by_name[projection_model].number


def _world_T_vehicle_at(frame_index: int) -> pycolmap.Rigid3d:
    """The synthetic trajectory: a forward drive with a slow yaw.

    Args:
        frame_index: Zero-based rig frame index.

    Returns:
        The vehicle pose in the world frame.
    """
    yaw_degrees: float = 1.5 * frame_index
    rotation: Rotation = Rotation.from_euler("z", yaw_degrees, degrees=True)
    translation: Float64[ndarray, "3"] = np.array([0.6 * frame_index, 0.05 * frame_index, 0.0])
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rotation.as_quat()), translation)


def _project(
    reconstruction: pycolmap.Reconstruction,
    frames_meta: FramesMeta,
    points_xyz: Float64[ndarray, "num_points 3"],
) -> tuple[dict[int, Float64[ndarray, "num_points 2"]], dict[int, Bool[ndarray, " num_points"]]]:
    """Project a point cloud into every image of a reconstruction.

    Args:
        reconstruction: Poses and cameras.
        frames_meta: The collection the images came from.
        points_xyz: World points in metres.

    Returns:
        Exact pixel projections per keyframe id, and a visibility mask per keyframe
        id: in front of the camera, finite, and inside the image.
    """
    projections: dict[int, Float64[ndarray, "num_points 2"]] = {}
    visibility: dict[int, Bool[ndarray, " num_points"]] = {}
    for keyframe in frames_meta.keyframes:
        image: pycolmap.Image = reconstruction.image(keyframe.keyframe_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        projected, visible = project_and_mask(camera, image.cam_from_world(), points_xyz)
        projections[keyframe.keyframe_id] = projected
        visibility[keyframe.keyframe_id] = visible
    return projections, visibility


def build_synthetic_rig(
    num_frames: int = 20,
    num_points: int = 500,
    noise_px: float = SYNTHETIC_NOISE_PX,
    seed: int = 7,
    fisheye: bool = False,
) -> SyntheticRig:
    """Build a two-camera rig sequence with exact poses and noisy observations.

    The cameras sit half a metre apart across the vehicle and look forward, so
    every point is seen by both cameras of several rig frames — the multi-view
    geometry cuSFM's `VEHICLE_RIG` mode assumes.

    Args:
        num_frames: Rig frames along the trajectory.
        num_points: World points to scatter in front of the vehicle.
        noise_px: Standard deviation of the Gaussian pixel noise.
        seed: PRNG seed.
        fisheye: Give the cameras the OPENCV_FISHEYE model.

    Returns:
        The synthetic rig, its metadata and its observations.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    schema = load_schema()
    collection: Message = schema.message_class(KEYFRAMES_METADATA_COLLECTION)()
    projection_model: str = "OPENCV_FISHEYE" if fisheye else "PINHOLE"

    vehicle_T_cams: dict[int, pycolmap.Rigid3d] = {}
    for camera_params_id, lateral_offset_m in enumerate((0.25, -0.25)):
        vehicle_T_cams[camera_params_id] = pycolmap.Rigid3d(
            pycolmap.Rotation3d(Rotation.from_matrix(VEHICLE_R_CAM).as_quat()),
            np.array([1.2, lateral_offset_m, 1.0]),
        )
        _fill_camera(
            collection.camera_params_id_to_camera_params[camera_params_id],
            camera_params_id,
            vehicle_T_cams[camera_params_id],
            projection_model,
            fisheye,
        )

    world_T_vehicle: dict[int, pycolmap.Rigid3d] = {}
    keyframe_id: int = 0
    for frame_index in range(num_frames):
        world_T_vehicle[frame_index] = _world_T_vehicle_at(frame_index)
        for camera_params_id in sorted(vehicle_T_cams):
            entry = collection.keyframes_metadata.add()
            entry.id = keyframe_id
            entry.camera_params_id = camera_params_id
            entry.timestamp_microseconds = 1_000_000 + frame_index * 33_333
            entry.image_name = f"camera_{camera_params_id}/{frame_index:04d}.jpeg"
            entry.synced_sample_id = frame_index
            write_rigid_transform(
                entry.camera_to_world, world_T_vehicle[frame_index] * vehicle_T_cams[camera_params_id]
            )
            keyframe_id += 1

    frames_meta: FramesMeta = parse_message(collection)
    reconstruction: pycolmap.Reconstruction = build_reconstruction(frames_meta).reconstruction
    candidates: Float64[ndarray, "num_candidates 3"] = np.column_stack(
        [
            rng.uniform(2.0, 8.0 + 0.6 * num_frames, MULTIVIEW_OVERSAMPLING * num_points),
            rng.uniform(-9.0, 9.0, MULTIVIEW_OVERSAMPLING * num_points),
            rng.uniform(-1.0, 4.0, MULTIVIEW_OVERSAMPLING * num_points),
        ]
    )
    projections, visibility = _project(reconstruction, frames_meta, candidates)

    # Keep only points several images actually see: a point observed twice or
    # less cannot be triangulated under `min_correspondences: 3`, so leaving it
    # in the ground truth would measure the sampler rather than the mapper.
    seen: Int[ndarray, " num_candidates"] = np.sum(np.stack(list(visibility.values())), axis=0)
    keep: Int[ndarray, " num_kept"] = np.flatnonzero(seen >= 3)[:num_points]
    if len(keep) < num_points:
        raise ValueError(f"Only {len(keep)} of {num_points} synthetic points are seen three times")

    points_xyz: Float64[ndarray, "num_points 3"] = candidates[keep]
    keypoints: dict[int, Float64[ndarray, "num_observations 2"]] = {}
    indices: dict[int, Int[ndarray, " num_observations"]] = {}
    for keyframe in frames_meta.keyframes:
        visible: Bool[ndarray, " num_kept"] = visibility[keyframe.keyframe_id][keep]
        exact: Float64[ndarray, "num_observations 2"] = projections[keyframe.keyframe_id][keep][visible]
        keypoints[keyframe.keyframe_id] = exact + rng.normal(0.0, noise_px, exact.shape)
        indices[keyframe.keyframe_id] = np.flatnonzero(visible).astype(np.int64)
    return SyntheticRig(
        frames_meta=frames_meta,
        points_xyz=points_xyz,
        keypoints_px=keypoints,
        point_indices=indices,
        world_T_vehicle=world_T_vehicle,
    )


def write_database(
    database_path: Path,
    frames_meta: FramesMeta,
    keypoints_px: dict[int, Float64[ndarray, "num_observations 2"]],
    matches: dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]],
    image_id_offset: int = 0,
) -> int:
    """Write cameras, images, keypoints and verified two-view geometries to SQLite.

    This stands in for the feature and matching stages: the mapper only ever
    reads keypoints and `two_view_geometries` back out, so a database carrying
    those is indistinguishable from a real one.

    Args:
        database_path: Where to create the COLMAP database.
        frames_meta: The collection the images come from.
        keypoints_px: Pixel keypoints per keyframe id.
        matches: Inlier matches per ordered image pair, as point2D index columns.
        image_id_offset: Added to every keyframe id, to write a database that
            numbers the same images differently.

    Returns:
        The number of image pairs written.
    """
    cameras: dict[int, pycolmap.Camera] = colmap_cameras(frames_meta)
    with pycolmap.Database.open(database_path) as database:
        for camera_params_id in sorted(cameras):
            camera: pycolmap.Camera = cameras[camera_params_id]
            camera.has_prior_focal_length = True
            database.write_camera(camera, use_camera_id=True)
        for keyframe in frames_meta.keyframes:
            image_id: int = keyframe.keyframe_id + image_id_offset
            image: pycolmap.Image = pycolmap.Image(
                name=keyframe.image_name, camera_id=keyframe.camera_params_id, image_id=image_id
            )
            database.write_image(image, use_image_id=True)
            database.write_keypoints(image_id, keypoints_px[keyframe.keyframe_id].astype(np.float32))
        num_pairs: int = 0
        for (image_id1, image_id2), pair_matches in sorted(matches.items()):
            if len(pair_matches) == 0:
                continue
            database.write_matches(image_id1 + image_id_offset, image_id2 + image_id_offset, pair_matches)
            geometry: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
            geometry.config = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
            geometry.inlier_matches = pair_matches
            database.write_two_view_geometry(image_id1 + image_id_offset, image_id2 + image_id_offset, geometry)
            num_pairs += 1
    return num_pairs


def synthetic_matches(rig: SyntheticRig, min_matches: int = 20) -> dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]]:
    """Ground-truth matches between every pair of synthetic images.

    Args:
        rig: The synthetic rig.
        min_matches: Skip pairs sharing fewer points than this.

    Returns:
        Inlier matches per ordered image pair.
    """
    matches: dict[tuple[int, int], UInt32[ndarray, "num_matches 2"]] = {}
    keyframe_ids: list[int] = sorted(rig.point_indices)
    for image_id1, image_id2 in itertools.combinations(keyframe_ids, 2):
        left: Int[ndarray, " n"] = rig.point_indices[image_id1]
        right: Int[ndarray, " n"] = rig.point_indices[image_id2]
        common: Int[ndarray, " n"] = np.intersect1d(left, right)
        if len(common) < min_matches:
            continue
        matches[(image_id1, image_id2)] = np.stack(
            [np.searchsorted(left, common), np.searchsorted(right, common)], axis=1
        ).astype(np.uint32)
    return matches


def caspar_plan(
    *,
    caspar_option_items: tuple[str, ...] = (),
    ceres_polish: bool = True,
    optimize_extrinsics: bool = False,
    supported_camera_models: frozenset[str] | None = None,
) -> BaExecutionPlan:
    """A `--ba-backend caspar` execution plan, for a test that wants the GPU backend.

    `MappingOptions` carries the plan rather than the knobs it was made from, so a
    test states what it wants here and hands the result to `quiet_options`. Stating a
    capability is how a test says "a build whose adapters stop here" without needing
    that build: the probe is skipped and every later decision — the pre-flight and
    `BaExecutionPlan.for_camera_models` — honours what was stated.

    Args:
        caspar_option_items: `--caspar-option name=value` strings, validated as a run
            would validate them.
        ceres_polish: Whether a closing Ceres bundle adjustment finishes a CASPAR run.
        optimize_extrinsics: Whether the *bundle adjustment* frees `sensor_from_rig`,
            which CASPAR cannot honour.
        supported_camera_models: The camera models to treat as CASPAR-supported, or
            None to measure this build.

    Returns:
        The plan.
    """
    capability: CasparCapability | None = (
        None
        if supported_camera_models is None
        else CasparCapability(
            availability="available", supported_camera_models=supported_camera_models, probe_errors=()
        )
    )
    return resolve_ba_plan(
        "caspar",
        caspar_option_items,
        ceres_polish=ceres_polish,
        optimize_extrinsics=optimize_extrinsics,
        capability=capability,
    )


def quiet_options(**overrides: object) -> MappingOptions:
    """Mapping options with the per-round printing turned off.

    Args:
        **overrides: Fields to override.

    Returns:
        The options.
    """
    return dataclasses.replace(MappingOptions(verbose=False), **overrides)


def every_round(mapping_config: VisionMappingConfig) -> VisionMappingConfig:
    """The same configuration with the early exit switched off.

    `run_mapping` leaves the outer loop when the observation change falls below
    `max_observation_change`, and the statistic is never negative, so a threshold of
    0.0 runs every one of `num_ba_iterations` rounds.

    Args:
        mapping_config: The mapping configuration.

    Returns:
        A copy whose `max_observation_change` is 0.0.
    """
    return dataclasses.replace(mapping_config, max_observation_change=0.0)


def match_tracks_to_truth(
    reconstruction: pycolmap.Reconstruction, rig: SyntheticRig
) -> tuple[set[int], int, list[float]]:
    """Match every triangulated track back to the point that generated it.

    Identity comes from the observations, not from proximity: each track element
    names an image and a keypoint index, and the synthetic rig knows which world
    point that keypoint came from. A track whose elements disagree has merged two
    points and is a genuine failure, however close its position looks.

    Args:
        reconstruction: The triangulated model.
        rig: The synthetic rig the observations came from.

    Returns:
        The truth indices recovered, the number of tracks mixing two points, and
        the position error of each recovered point as a fraction of its distance
        to the first camera that saw it.
    """
    recovered: set[int] = set()
    mixed_tracks: int = 0
    relative_errors: list[float] = []
    for point3D_id in reconstruction.point3D_ids():
        point: pycolmap.Point3D = reconstruction.point3D(point3D_id)
        truth_indices: set[int] = {
            int(rig.point_indices[element.image_id][element.point2D_idx]) for element in point.track.elements
        }
        if len(truth_indices) != 1:
            mixed_tracks += 1
            continue
        truth_index: int = truth_indices.pop()
        recovered.add(truth_index)
        truth_xyz: Float64[ndarray, "3"] = rig.points_xyz[truth_index]
        first_image: pycolmap.Image = reconstruction.image(point.track.elements[0].image_id)
        range_m: float = float(np.linalg.norm(truth_xyz - first_image.projection_center()))
        relative_errors.append(float(np.linalg.norm(point.xyz - truth_xyz)) / range_m)
    return recovered, mixed_tracks, relative_errors


def random_offset(
    rng: np.random.Generator, translation_m: float, rotation_deg: float
) -> pycolmap.Rigid3d:
    """One rigid offset of an exact size, in a random direction.

    Both the pose-recovery and the extrinsic-recovery tests need the same thing: an
    error whose magnitude is known to the metre and the degree, so that what is
    measured afterwards is how much of it came back.

    Args:
        rng: The generator; the rotation is drawn before the translation.
        translation_m: Length of the translation, metres.
        rotation_deg: Angle of the rotation, degrees.

    Returns:
        The offset transform.
    """
    rotation_vector: Float64[ndarray, "3"] = rng.normal(0.0, 1.0, 3)
    rotation_vector = rotation_vector / np.linalg.norm(rotation_vector) * np.deg2rad(rotation_deg)
    translation: Float64[ndarray, "3"] = rng.normal(0.0, 1.0, 3)
    translation = translation / np.linalg.norm(translation) * translation_m
    return pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_rotvec(rotation_vector).as_quat()), translation)
