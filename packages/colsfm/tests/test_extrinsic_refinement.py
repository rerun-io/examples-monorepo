"""Tests for the regularised extrinsic refinement.

The seams under test are the module's public ones: `RigReprojectionCost` (its analytic
Jacobian against central differences), `co_observed_camera_pairs` (the blob's 777-block
arithmetic) and `solve_extrinsics` (does a wrong calibration come back, and does a tight
prior hold it in place).

The scene is a synthetic four-camera rig with exact projections, built the way
`tests/colsfm_probes/synthetic.py` builds its two-camera one, so any residual measured
here comes from the module and not from the data.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pyceres
import pycolmap
import pytest
from jaxtyping import Float
from numpy import ndarray
from scipy.spatial.transform import Rotation

from colsfm import RESOURCES_DIR
from colsfm.config import BundleAdjustmentConfig, CusfmConfig, read_config_directory
from colsfm.extrinsic_costs import RepeatedCauchyLoss, RigReprojectionCost, cauchy_square_root_scale
from colsfm.extrinsic_observations import (
    CameraObservations,
    ObservationIndex,
    build_observation_index,
    camera_observations,
    co_observed_camera_pairs,
)
from colsfm.extrinsic_refinement import (
    ExtrinsicRefinementOptions,
    apply_extrinsics,
    extrinsic_deltas,
    solve_extrinsics,
)
from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.geometry import img_from_cam
from colsfm.pose_graph import quaternion_plus_jacobian_transpose
from colsfm.reconstruction import RIG_ID, RigReference, camera_sensor_id

Matrix3: TypeAlias = Float[ndarray, "3 3"]
Vector3: TypeAlias = Float[ndarray, "3"]
QuaternionXYZW: TypeAlias = Float[ndarray, "4"]

IMAGE_WIDTH: int = 640
"""Width in pixels of every synthetic camera."""

IMAGE_HEIGHT: int = 480
"""Height in pixels of every synthetic camera."""

FOCAL_LENGTH_PX: float = 400.0
"""Focal length in pixels of every synthetic camera."""

REFERENCE_CAMERA_PARAMS_ID: int = 0
"""The rig origin; its extrinsic is pinned exactly as the pipeline pins it."""

NUM_FRAMES: int = 8
"""Rig instants along the synthetic trajectory."""


# ======================================================================================
# a synthetic four-camera rig
# ======================================================================================


@dataclass(frozen=True, slots=True)
class RigScene:
    """A posed, triangulated four-camera rig whose observations are exact projections."""

    reconstruction: pycolmap.Reconstruction
    """Cameras, rig, frames, images with keypoints, and the 3D points they observe."""
    rig_reference: RigReference
    """The camera-referenced rig description the module needs."""
    truth_vehicle_T_cam: dict[int, pycolmap.Rigid3d]
    """The extrinsics that generated every observation."""


def _look_at(position_xyz: Vector3, yaw_deg: float) -> pycolmap.Rigid3d:
    """A `vehicle_T_cam` sitting at `position_xyz` and yawed about the vehicle Z axis.

    Args:
        position_xyz: Camera origin in the vehicle frame, metres.
        yaw_deg: Rotation about the vehicle Z axis, degrees.

    Returns:
        The `vehicle_T_cam` transform.
    """
    about_z: Matrix3 = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    # Camera looks along its own +Z; the vehicle frame is FLU, so a camera pointing
    # forward has vehicle axes (x, y, z) = (right, down, forward) mapped accordingly.
    forward_looking: Matrix3 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(about_z @ forward_looking), position_xyz)


def _truth_extrinsics() -> dict[int, pycolmap.Rigid3d]:
    """Two stereo pairs, one looking forward and one looking 60 degrees to the left.

    Returns:
        `vehicle_T_cam` per `camera_params_id`, ids 0 to 3.
    """
    return {
        0: _look_at(np.array([0.30, 0.075, 0.20]), 0.0),
        1: _look_at(np.array([0.30, -0.075, 0.20]), 0.0),
        2: _look_at(np.array([0.10, 0.20, 0.20]), 60.0),
        3: _look_at(np.array([0.02, 0.075, 0.20]), 60.0),
    }


def _world_T_vehicle_at(frame_index: int) -> pycolmap.Rigid3d:
    """The rig's pose at one frame: a slide along +X with a small yaw.

    Args:
        frame_index: Zero-based frame number.

    Returns:
        The `world_T_vehicle` transform.
    """
    world_R_vehicle: Matrix3 = Rotation.from_euler("z", 6.0 * frame_index, degrees=True).as_matrix()
    return pycolmap.Rigid3d(pycolmap.Rotation3d(world_R_vehicle), np.array([0.5 * frame_index, 0.0, 0.0], dtype=np.float64))


def _world_points(seed: int = 3) -> Float[ndarray, "n_points 3"]:
    """A jittered slab of world points the rig can see from every side.

    Args:
        seed: PRNG seed for the jitter.

    Returns:
        World-space point positions in metres.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    xs: Float[ndarray, "7"] = np.linspace(-3.0, 8.0, 7)
    ys: Float[ndarray, "7"] = np.linspace(-6.0, 6.0, 7)
    zs: Float[ndarray, "3"] = np.array([-1.0, 0.5, 2.0])
    grid: Float[ndarray, "n_points 3"] = np.array(list(itertools.product(xs, ys, zs)), dtype=np.float64)
    return grid + rng.normal(0.0, 0.08, grid.shape)


def _pinhole_camera(camera_params_id: int) -> pycolmap.Camera:
    """A PINHOLE camera with the suite's canonical intrinsics.

    Args:
        camera_params_id: Identifier to assign.

    Returns:
        The camera.
    """
    camera: pycolmap.Camera = pycolmap.Camera.create_from_model_id(
        camera_params_id, pycolmap.CameraModelId.PINHOLE, FOCAL_LENGTH_PX, IMAGE_WIDTH, IMAGE_HEIGHT
    )
    camera.params = [FOCAL_LENGTH_PX, FOCAL_LENGTH_PX, IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0]
    return camera


def build_rig_scene(num_frames: int = NUM_FRAMES) -> RigScene:
    """Build the four-camera rig with exact projections and triangulated points.

    Args:
        num_frames: Rig instants along the trajectory.

    Returns:
        The populated `RigScene`.
    """
    truth: dict[int, pycolmap.Rigid3d] = _truth_extrinsics()
    reference: RigReference = RigReference(
        camera_params_id=REFERENCE_CAMERA_PARAMS_ID,
        vehicle_T_reference=truth[REFERENCE_CAMERA_PARAMS_ID],
    )

    reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction()
    for camera_params_id in sorted(truth):
        reconstruction.add_camera(_pinhole_camera(camera_params_id))
    rig: pycolmap.Rig = pycolmap.Rig()
    rig.rig_id = RIG_ID
    rig.add_ref_sensor(camera_sensor_id(REFERENCE_CAMERA_PARAMS_ID))
    for camera_params_id, vehicle_T_cam in sorted(truth.items()):
        if camera_params_id == REFERENCE_CAMERA_PARAMS_ID:
            continue
        rig.add_sensor(camera_sensor_id(camera_params_id), vehicle_T_cam.inverse() * reference.vehicle_T_reference)
    reconstruction.add_rig(rig)

    image_id: int = 0
    camera_by_image_id: dict[int, int] = {}
    for frame_index in range(num_frames):
        world_T_reference: pycolmap.Rigid3d = _world_T_vehicle_at(frame_index) * reference.vehicle_T_reference
        frame: pycolmap.Frame = pycolmap.Frame()
        frame.frame_id = frame_index + 1
        frame.rig_id = RIG_ID
        frame.rig_from_world = world_T_reference.inverse()
        for camera_params_id in sorted(truth):
            image_id += 1
            frame.add_data_id(pycolmap.data_t(camera_sensor_id(camera_params_id), image_id))
            camera_by_image_id[image_id] = camera_params_id
        reconstruction.add_frame(frame)
        reconstruction.register_frame(frame.frame_id)

    for one_image_id, camera_params_id in camera_by_image_id.items():
        image: pycolmap.Image = pycolmap.Image(
            name=f"cam{camera_params_id}/{one_image_id:04d}.png", camera_id=camera_params_id, image_id=one_image_id
        )
        image.frame_id = (one_image_id - 1) // len(truth) + 1
        reconstruction.add_image(image)

    points_xyz: Float[ndarray, "n_points 3"] = _world_points()
    point3D_ids: list[int] = [reconstruction.add_point3D(xyz, pycolmap.Track(), np.array([128, 128, 128], dtype=np.uint8)) for xyz in points_xyz]

    for one_image_id in sorted(reconstruction.images):
        image = reconstruction.image(one_image_id)
        camera: pycolmap.Camera = reconstruction.camera(image.camera_id)
        cam_from_world: pycolmap.Rigid3d = image.cam_from_world()
        points_in_cam: Float[ndarray, "n_points 3"] = points_xyz @ np.asarray(cam_from_world.rotation.matrix()).T + np.asarray(
            cam_from_world.translation
        )
        projected: Float[ndarray, "n_points 2"] = img_from_cam(camera, points_in_cam)
        visible: ndarray = (
            np.isfinite(projected).all(axis=1)
            & (points_in_cam[:, 2] > 0.3)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < IMAGE_WIDTH)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < IMAGE_HEIGHT)
        )
        image.points2D = pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in projected[visible]])
        for point2D_idx, point_index in enumerate(np.flatnonzero(visible)):
            reconstruction.add_observation(point3D_ids[int(point_index)], pycolmap.TrackElement(one_image_id, point2D_idx))

    return RigScene(reconstruction=reconstruction, rig_reference=reference, truth_vehicle_T_cam=truth)


def perturb(pose: pycolmap.Rigid3d, translation_mm: float, rotation_deg: float) -> pycolmap.Rigid3d:
    """Offset a pose by a fixed translation and rotation, deterministically.

    Args:
        pose: The pose to move.
        translation_mm: Distance to move its origin, millimetres.
        rotation_deg: Angle to rotate it by, degrees.

    Returns:
        The moved pose.
    """
    axis: Vector3 = np.array([0.4, -0.6, 0.6928203230275509], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    direction: Vector3 = np.array([0.6, 0.48, -0.64], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    delta: pycolmap.Rigid3d = pycolmap.Rigid3d(pycolmap.Rotation3d(axis * np.deg2rad(rotation_deg)), direction * translation_mm * 1e-3)
    return pose * delta


@pytest.fixture(scope="module")
def rig_scene() -> RigScene:
    """The synthetic four-camera rig, built once for the module."""
    return build_rig_scene()


# ======================================================================================
# the robust-loss algebra
# ======================================================================================


def test_cauchy_square_root_scale_reproduces_the_cauchy_cost() -> None:
    """`(g * r)^2` must equal `rho(|r|^2)` of `CauchyLoss(1.0)`, small residuals included."""
    squared: Float[ndarray, " n"] = np.array([0.0, 1e-12, 1e-9, 1e-3, 1.0, 25.0, 1e4])
    scale, _ = cauchy_square_root_scale(squared)
    assert np.allclose(scale**2 * squared, np.log1p(squared), atol=1e-14)


def test_cauchy_square_root_scale_derivative_matches_finite_differences() -> None:
    """The reported `d(rho(s)/s)/ds` must be the derivative of the reported ratio."""
    squared: Float[ndarray, " n"] = np.array([1e-3, 0.1, 1.0, 9.0, 100.0])
    step: float = 1e-7
    _, derivative = cauchy_square_root_scale(squared)
    forward, _ = cauchy_square_root_scale(squared + step)
    backward, _ = cauchy_square_root_scale(squared - step)
    numeric: Float[ndarray, " n"] = (forward**2 - backward**2) / (2.0 * step)
    assert np.allclose(derivative, numeric, rtol=1e-5)


def test_repeated_cauchy_loss_is_a_multiple_of_cauchy() -> None:
    """A multiplicity-`n` loss must be `n` times `CauchyLoss(1.0)` in all three outputs."""
    single: Float[ndarray, "3"] = np.empty(3, dtype=np.float64)
    repeated: Float[ndarray, "3"] = np.empty(3, dtype=np.float64)
    RepeatedCauchyLoss(1).Evaluate(2.5, single)
    RepeatedCauchyLoss(7).Evaluate(2.5, repeated)
    assert np.allclose(repeated, 7.0 * single)
    assert np.allclose(single, pyceres.CauchyLoss(1.0).evaluate(2.5))


def test_repeated_cauchy_loss_rejects_a_zero_multiplicity() -> None:
    """A constraint that stands for no block is a caller bug, not a free pass."""
    with pytest.raises(ValueError, match="multiplicity"):
        RepeatedCauchyLoss(0)


# ======================================================================================
# the vectorised reprojection cost
# ======================================================================================


def _quaternion_plus(quat_xyzw: QuaternionXYZW, delta: Vector3) -> QuaternionXYZW:
    """`EigenQuaternionManifold.Plus`: `q * exp(delta)` in `(x, y, z, w)` order.

    Args:
        quat_xyzw: The base quaternion.
        delta: A rotation vector in the tangent space, radians.

    Returns:
        The perturbed unit quaternion.
    """
    angle: float = float(np.linalg.norm(delta))
    if angle == 0.0:
        return quat_xyzw.copy()
    delta_quat: QuaternionXYZW = np.concatenate([np.sin(angle / 2.0) * delta / angle, [np.cos(angle / 2.0)]])
    x1, y1, z1, w1 = quat_xyzw
    x2, y2, z2, w2 = delta_quat
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def test_reprojection_cost_jacobian_matches_central_differences(rig_scene: RigScene) -> None:
    """The analytic Jacobian must agree with central differences over the local 6.

    The residual is robustified and whitened inside `Evaluate`, so the comparison is
    against the returned residual vector, not against a bare reprojection.
    """
    observations = camera_observations(rig_scene.reconstruction, rig_scene.rig_reference)[2]
    cost: RigReprojectionCost = RigReprojectionCost(observations, reprojection_sigma_px=4.0)
    # Off the true extrinsic, so the robust weights are neither 1 nor saturated.
    moved: pycolmap.Rigid3d = perturb(rig_scene.truth_vehicle_T_cam[2], translation_mm=40.0, rotation_deg=0.8)
    quat_xyzw: QuaternionXYZW = np.asarray(moved.rotation.quat, dtype=np.float64).copy()
    translation: Vector3 = np.asarray(moved.translation, dtype=np.float64).copy()

    num_residuals: int = 2 * observations.points_in_vehicle.shape[0]
    residuals: Float[ndarray, " n_residuals"] = np.empty(num_residuals, dtype=np.float64)
    quat_jacobian: Float[ndarray, " n"] = np.empty(num_residuals * 4, dtype=np.float64)
    translation_jacobian: Float[ndarray, " n"] = np.empty(num_residuals * 3, dtype=np.float64)
    assert cost.Evaluate([quat_xyzw, translation], residuals, [quat_jacobian, translation_jacobian])
    assert np.max(np.abs(residuals)) > 1e-3, "the probe point must produce a real residual"

    # `quaternion_plus_jacobian_transpose` returns `4 * P.T`, so `P` recovers the local
    # 3-column Jacobian from the ambient 4-column one Ceres is handed.
    plus_jacobian: Float[ndarray, "4 3"] = quaternion_plus_jacobian_transpose(quat_xyzw, np.empty((3, 4))).T / 4.0
    analytic: Float[ndarray, "n_residuals 6"] = np.concatenate(
        [
            quat_jacobian.reshape(num_residuals, 4) @ plus_jacobian,
            translation_jacobian.reshape(num_residuals, 3),
        ],
        axis=1,
    )

    step: float = 1e-7
    numeric: Float[ndarray, "n_residuals 6"] = np.empty((num_residuals, 6), dtype=np.float64)
    forward: Float[ndarray, " n_residuals"] = np.empty(num_residuals, dtype=np.float64)
    backward: Float[ndarray, " n_residuals"] = np.empty(num_residuals, dtype=np.float64)
    for local_index in range(6):
        delta: Float[ndarray, "6"] = np.zeros(6, dtype=np.float64)
        delta[local_index] = step
        cost.Evaluate([_quaternion_plus(quat_xyzw, delta[:3]), translation + delta[3:]], forward, None)
        cost.Evaluate([_quaternion_plus(quat_xyzw, -delta[:3]), translation - delta[3:]], backward, None)
        numeric[:, local_index] = (forward - backward) / (2.0 * step)

    assert np.allclose(analytic, numeric, rtol=2e-4, atol=2e-5)


def test_camera_observations_cover_every_camera(rig_scene: RigScene) -> None:
    """Each camera's observations must fold the frozen geometry in without losing any."""
    observations = camera_observations(rig_scene.reconstruction, rig_scene.rig_reference)
    assert sorted(observations) == [0, 1, 2, 3]
    total: int = sum(int(entry.points_in_vehicle.shape[0]) for entry in observations.values())
    assert total == rig_scene.reconstruction.compute_num_observations()
    for entry in observations.values():
        assert entry.points_in_vehicle.shape[0] == entry.observed_px.shape[0]


def test_a_prebuilt_index_gathers_exactly_what_building_one_would(rig_scene: RigScene) -> None:
    """The alternation reuses one index across its rounds; that must change no number.

    `build_observation_index` now resolves every observation to a row of the sorted
    point array and keeps the sort, because neither changes while the alternation runs
    — bundle adjustment inside it moves points but adds and removes no observation.
    The per-round pass used to redo the sort and one `searchsorted` per image.
    """
    fresh: dict[int, CameraObservations] = camera_observations(rig_scene.reconstruction, rig_scene.rig_reference)
    index: ObservationIndex = build_observation_index(rig_scene.reconstruction)
    reused: dict[int, CameraObservations] = camera_observations(rig_scene.reconstruction, rig_scene.rig_reference, index)

    assert sorted(reused) == sorted(fresh)
    for camera_params_id, entry in fresh.items():
        np.testing.assert_array_equal(reused[camera_params_id].points_in_vehicle, entry.points_in_vehicle)
        np.testing.assert_array_equal(reused[camera_params_id].observed_px, entry.observed_px)


def test_an_index_outlived_by_its_points_is_refused(rig_scene: RigScene) -> None:
    """Reusing an index across a filter would silently reproject the wrong points.

    Every row of the index names a position by its place in the sorted point array, so
    a model that has lost a point makes every later row mean a different point. The old
    per-round `searchsorted` dropped the missing observations instead, which changed
    the size of the problem without saying so.
    """
    index: ObservationIndex = build_observation_index(rig_scene.reconstruction)
    reconstruction: pycolmap.Reconstruction = rig_scene.reconstruction
    reconstruction.delete_point3D(next(iter(reconstruction.points3D)))

    with pytest.raises(ValueError, match="rebuild it with `build_observation_index`"):
        camera_observations(reconstruction, rig_scene.rig_reference, index)


def test_camera_observations_reproject_exactly_at_the_true_extrinsics(rig_scene: RigScene) -> None:
    """With the generating extrinsics the whitened residual must be zero to float noise."""
    observations = camera_observations(rig_scene.reconstruction, rig_scene.rig_reference)[3]
    truth: pycolmap.Rigid3d = rig_scene.truth_vehicle_T_cam[3]
    cost: RigReprojectionCost = RigReprojectionCost(observations, reprojection_sigma_px=4.0)
    residuals: Float[ndarray, " n_residuals"] = np.empty(2 * observations.points_in_vehicle.shape[0], dtype=np.float64)
    assert cost.Evaluate(
        [
            np.asarray(truth.rotation.quat, dtype=np.float64).copy(),
            np.asarray(truth.translation, dtype=np.float64).copy(),
        ],
        residuals,
        None,
    )
    assert np.max(np.abs(residuals)) < 1e-9


# ======================================================================================
# the blob's block arithmetic
# ======================================================================================


def test_co_observed_pairs_count_every_pair_once_per_rig_frame(rig_scene: RigScene) -> None:
    """Four cameras in every one of eight frames give `C(4, 2) * 8 = 48` blocks."""
    pair_counts: dict[tuple[int, int], int] = co_observed_camera_pairs(rig_scene.reconstruction)
    assert len(pair_counts) == 6
    assert set(pair_counts.values()) == {NUM_FRAMES}
    assert sum(pair_counts.values()) == 6 * NUM_FRAMES


def test_co_observed_pairs_reproduce_the_blobs_galileo_count(galileo_input_meta: Path) -> None:
    """Galileo's 29 rig frames must give the blob's logged 777 relative-extrinsic blocks.

    `bundle_adjustment_solver.cc:1138` logs `Added 777 camera rig extrinsic constraints`
    for a run whose rig frames hold 8, 8, ..., 6 and 4 cameras. This is the arithmetic
    that identifies the constraint as one per co-observed camera pair per rig frame,
    rather than one per declared stereo pair.
    """
    frames_meta: FramesMeta = read_frames_meta(galileo_input_meta)
    cameras_by_rig_frame: dict[int, set[int]] = {}
    for keyframe in frames_meta.keyframes:
        cameras_by_rig_frame.setdefault(keyframe.synced_sample_id, set()).add(keyframe.camera_params_id)
    total: int = sum(len(cameras) * (len(cameras) - 1) // 2 for cameras in cameras_by_rig_frame.values())
    assert total == 777


# ======================================================================================
# the extrinsics-only solve
# ======================================================================================


def _shift(before: pycolmap.Rigid3d, after: pycolmap.Rigid3d) -> tuple[float, float]:
    """Translation in millimetres and rotation in degrees between two poses.

    Args:
        before: The first pose.
        after: The second pose.

    Returns:
        `(translation mm, rotation deg)`.
    """
    return extrinsic_deltas({0: before}, {0: after})


def _scene_with_miscalibrated_camera(
    translation_mm: float, rotation_deg: float, camera_params_id: int = 2
) -> tuple[RigScene, dict[int, pycolmap.Rigid3d], int]:
    """Build the scene and move one camera's calibration away from the truth.

    The observations still come from the true extrinsics, so the imagery disagrees with
    the calibration by exactly the perturbation.

    Args:
        translation_mm: How far to move the camera's origin.
        rotation_deg: How far to rotate it.
        camera_params_id: Which camera to miscalibrate.

    Returns:
        The scene (with the perturbed extrinsic already written into its rig), the
        calibration both priors pull towards, and the miscalibrated camera's id.
    """
    scene: RigScene = build_rig_scene()
    calibration: dict[int, pycolmap.Rigid3d] = dict(scene.truth_vehicle_T_cam)
    calibration[camera_params_id] = perturb(scene.truth_vehicle_T_cam[camera_params_id], translation_mm, rotation_deg)
    apply_extrinsics(scene.reconstruction, scene.rig_reference, calibration)
    return scene, calibration, camera_params_id


def test_weak_priors_recover_a_miscalibrated_camera() -> None:
    """A 20 mm / 0.5 deg calibration error must come back when the priors barely pull."""
    scene, calibration, camera_params_id = _scene_with_miscalibrated_camera(20.0, 0.5)
    started_mm, started_deg = _shift(scene.truth_vehicle_T_cam[camera_params_id], calibration[camera_params_id])
    assert started_mm == pytest.approx(20.0, abs=1e-6)
    assert started_deg == pytest.approx(0.5, abs=1e-6)

    refined, stats = solve_extrinsics(
        scene.reconstruction,
        scene.rig_reference,
        calibration,
        ExtrinsicRefinementOptions(extrinsic_translation_sigma_m=10.0, extrinsic_rotation_sigma_deg=180.0, verbose=False),
    )
    recovered_mm, recovered_deg = _shift(scene.truth_vehicle_T_cam[camera_params_id], refined[camera_params_id])
    assert stats.termination == "CONVERGENCE"
    assert recovered_mm < 1.0
    assert recovered_deg < 0.05


def test_strong_priors_hold_a_miscalibrated_camera_at_its_calibration() -> None:
    """With sigmas far below the error, the solve must stay inside the prior."""
    scene, calibration, camera_params_id = _scene_with_miscalibrated_camera(20.0, 0.5)
    refined, _ = solve_extrinsics(
        scene.reconstruction,
        scene.rig_reference,
        calibration,
        ExtrinsicRefinementOptions(extrinsic_translation_sigma_m=1e-6, extrinsic_rotation_sigma_deg=1e-4, verbose=False),
    )
    moved_mm, moved_deg = _shift(calibration[camera_params_id], refined[camera_params_id])
    assert moved_mm < 1.0
    assert moved_deg < 0.05


def test_the_reference_camera_never_moves() -> None:
    """The rig origin is pinned, so its extrinsic must come out bit-identical."""
    scene, calibration, _ = _scene_with_miscalibrated_camera(20.0, 0.5)
    refined, _ = solve_extrinsics(
        scene.reconstruction,
        scene.rig_reference,
        calibration,
        ExtrinsicRefinementOptions(extrinsic_translation_sigma_m=10.0, extrinsic_rotation_sigma_deg=180.0, verbose=False),
    )
    moved_mm, moved_deg = _shift(calibration[REFERENCE_CAMERA_PARAMS_ID], refined[REFERENCE_CAMERA_PARAMS_ID])
    assert moved_mm == 0.0
    assert moved_deg == 0.0


def test_the_solve_reports_the_blocks_it_built() -> None:
    """The reported counts are the parity evidence against the blob's log; check them."""
    scene, calibration, _ = _scene_with_miscalibrated_camera(20.0, 0.5)
    _, stats = solve_extrinsics(scene.reconstruction, scene.rig_reference, calibration, ExtrinsicRefinementOptions(verbose=False))
    assert stats.num_cameras == 4
    assert stats.num_reprojection_blocks == 3, "one per free camera; the reference has none"
    assert stats.num_absolute_prior_blocks == 3, "one per free camera; the pinned one is inert"
    assert stats.num_relative_prior_blocks == 6
    assert stats.num_relative_prior_frames == 6 * NUM_FRAMES
    assert stats.num_reprojection_residuals == 2 * (
        scene.reconstruction.compute_num_observations()
        - sum(
            scene.reconstruction.image(image_id).num_points3D
            for image_id in scene.reconstruction.reg_image_ids()
            if scene.reconstruction.image(image_id).camera_id == REFERENCE_CAMERA_PARAMS_ID
        )
    )


def test_the_relative_prior_alone_still_pulls_towards_the_calibration() -> None:
    """Turning the absolute prior off must leave the inter-camera constraint in charge."""
    scene, calibration, camera_params_id = _scene_with_miscalibrated_camera(20.0, 0.5)
    refined, stats = solve_extrinsics(
        scene.reconstruction,
        scene.rig_reference,
        calibration,
        ExtrinsicRefinementOptions(
            use_absolute_prior=False,
            extrinsic_translation_sigma_m=1e-6,
            extrinsic_rotation_sigma_deg=1e-4,
            verbose=False,
        ),
    )
    assert stats.num_absolute_prior_blocks == 0
    moved_mm, _ = _shift(calibration[camera_params_id], refined[camera_params_id])
    assert moved_mm < 1.0


def test_apply_extrinsics_round_trips_through_the_rig() -> None:
    """Writing extrinsics into the rig and reading them back must be the identity."""
    scene: RigScene = build_rig_scene(num_frames=2)
    wanted: dict[int, pycolmap.Rigid3d] = {
        camera_params_id: perturb(pose, 12.0, 0.3) if camera_params_id else pose for camera_params_id, pose in scene.truth_vehicle_T_cam.items()
    }
    apply_extrinsics(scene.reconstruction, scene.rig_reference, wanted)
    read_back: dict[int, pycolmap.Rigid3d] = scene.rig_reference.vehicle_T_cam_by_camera_params_id(scene.reconstruction)
    translation_mm, rotation_deg = extrinsic_deltas(wanted, read_back)
    assert translation_mm < 1e-9
    assert rotation_deg < 1e-9


def test_solve_extrinsics_rejects_a_vehicle_referenced_rig(rig_scene: RigScene) -> None:
    """A rig whose reference owns no images is the gotcha-13 trap; refuse it loudly."""
    with pytest.raises(ValueError, match="camera-referenced"):
        solve_extrinsics(
            rig_scene.reconstruction,
            RigReference(camera_params_id=None, vehicle_T_reference=pycolmap.Rigid3d()),
            rig_scene.truth_vehicle_T_cam,
        )


def test_default_sigmas_are_the_shipped_configs_own(repo_root: Path) -> None:
    """The prior sigmas are `extrinsic_error_*`, identified by reproducing the blob's costs.

    Recomputing the blob's logged `absolute_extrinsic` group cost from its exported
    extrinsics gives 0.7230 against its logged 0.722949 with these two values and
    `CauchyLoss(1.0)`; `relative_pose_translation_error_meters` / `_rotation_error_degrees`
    (0.1 m / 5 deg) gives 0.09, eight times too small. Pin the defaults to the config so a
    silent drift cannot undo that identification.
    """
    config: CusfmConfig = read_config_directory(RESOURCES_DIR / "configs" / "loop-closure-fixed")
    ba_config: BundleAdjustmentConfig = config.vision_mapping.bundle_adjustment
    defaults: ExtrinsicRefinementOptions = ExtrinsicRefinementOptions()
    assert defaults.extrinsic_translation_sigma_m == ba_config.extrinsic_error_meters
    assert defaults.extrinsic_rotation_sigma_deg == ba_config.extrinsic_error_degrees
    assert defaults.reprojection_sigma_px == ba_config.reprojection_error_standard_deviation
    assert ba_config.loss_type == "CAUCHY"
    assert ba_config.loss_function_scale == 1.0
