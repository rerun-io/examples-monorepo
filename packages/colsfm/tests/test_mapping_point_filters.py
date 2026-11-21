"""The two guards that run before every solve, on the points Ceres cannot survive.

`filter_degenerate_points` and `filter_projection_failures` are the reason a
RoboCap round no longer reports 1e152 px of mean reprojection error. Both are
pure functions over a triangulated model and its observation manager, so they are
tested by corrupting one point of a healthy synthetic map and naming which point
had to go — no solver, no backend, no dataset.

They are their own file because that is a different subject from the mapper's own
behaviour: these tests never call `run_mapping`, they build the triangulation
themselves so the observation manager stays reachable afterwards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float64
from mapping_helpers import SyntheticRig, synthetic_matches, write_database
from numpy import ndarray

from colsfm import point_filters
from colsfm.config import CusfmConfig
from colsfm.mapping import (
    BundleAdjustmentSolve,
    Correspondences,
    filter_degenerate_points,
    filter_projection_failures,
    load_correspondences,
    projection_sanity_bound_px,
    solve_bundle_adjustment,
)
from colsfm.reconstruction import build_reconstruction
from colsfm.solver_report import SolverFailure


@pytest.fixture(scope="module")
def synthetic_database(synthetic_rig: SyntheticRig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic rig's observations as a COLMAP database."""
    database_path: Path = tmp_path_factory.mktemp("synthetic") / "database.db"
    write_database(database_path, synthetic_rig.frames_meta, synthetic_rig.keypoints_px, synthetic_matches(synthetic_rig))
    return database_path


DEPTH_THRESHOLD_M: float = 300.0
"""`vision_mapping_config.depth_threshold`, the isaac profile's bound in metres on a point's distance to its nearest
observing camera."""


def _triangulated(rig: SyntheticRig, database_path: Path, config: CusfmConfig) -> tuple[pycolmap.Reconstruction, Correspondences]:
    """Triangulate a rig and hand back the model with its live observation manager.

    `run_mapping` loads its own correspondences, so a caller that wants the
    manager afterwards has to build the model itself; a manager without the
    correspondence graph silently declines to delete anything.

    Args:
        rig: The synthetic rig.
        database_path: Its COLMAP database.
        config: The isaac profile.

    Returns:
        The triangulated reconstruction and the correspondences that own its manager.
    """
    reconstruction: pycolmap.Reconstruction = build_reconstruction(rig.frames_meta).reconstruction
    correspondences: Correspondences = load_correspondences(reconstruction, database_path, config.vision_mapping.min_num_matches_per_pair)
    triangulator: pycolmap.IncrementalTriangulator = pycolmap.IncrementalTriangulator(
        correspondences.graph, reconstruction, correspondences.observation_manager
    )
    options: pycolmap.IncrementalTriangulatorOptions = pycolmap.IncrementalTriangulatorOptions()
    for image_id in sorted(reconstruction.images):
        triangulator.triangulate_image(options, image_id)
    reconstruction.update_point_3d_errors()
    return reconstruction, correspondences


def _corrupt_point(reconstruction: pycolmap.Reconstruction, point3D_id: int, xyz: Float64[ndarray, "3"]) -> None:
    """Overwrite one point's world position in place.

    Args:
        reconstruction: The model to edit.
        point3D_id: The point to move.
        xyz: Its new world position.
    """
    reconstruction.point3D(point3D_id).xyz = xyz


def test_degeneracy_guard_drops_points_ceres_cannot_survive(synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig) -> None:
    """Non-finite, astronomically distant and beyond-cap points all go; sane ones stay.

    Ceres reported ~1e152 px mean reprojection on the first RoboCap rounds, which
    is what a point at 1e30 m or a NaN does to a normal equation. This is the
    guard that keeps such a point out of the solver.
    """
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager

    point3D_ids: list[int] = sorted(reconstruction.points3D)
    assert len(point3D_ids) >= 4
    doomed: list[int] = point3D_ids[:3]
    survivor: int = point3D_ids[3]
    survivor_xyz: Float64[ndarray, "3"] = np.asarray(reconstruction.point3D(survivor).xyz, dtype=np.float64).copy()
    _corrupt_point(reconstruction, doomed[0], np.array([np.nan, 0.0, 0.0]))
    _corrupt_point(reconstruction, doomed[1], np.array([1e30, 0.0, 0.0]))
    _corrupt_point(reconstruction, doomed[2], np.array([0.0, 0.0, 2.0 * DEPTH_THRESHOLD_M]))

    dropped: int = filter_degenerate_points(observation_manager, reconstruction, DEPTH_THRESHOLD_M)

    assert dropped > 0
    for point3D_id in doomed:
        assert point3D_id not in reconstruction.points3D
    assert survivor in reconstruction.points3D
    assert np.array_equal(np.asarray(reconstruction.point3D(survivor).xyz, dtype=np.float64), survivor_xyz)


def test_degeneracy_guard_is_a_no_op_on_a_healthy_model(synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig) -> None:
    """A model whose points all sit inside the cap loses nothing to the guard."""
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager
    num_points_before: int = reconstruction.num_points3D()

    assert filter_degenerate_points(observation_manager, reconstruction, DEPTH_THRESHOLD_M) == 0
    assert reconstruction.num_points3D() == num_points_before


def test_degeneracy_guard_measures_from_the_cameras_not_the_world_origin(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A walk far from the world origin keeps its points; one point pushed far from its own cameras goes.

    The bound used to be a box on the world coordinates, which deleted every point of Casa Loma 006 once the walk was
    300 m from where it started.
    """
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager
    reconstruction.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(), np.array([5.0 * DEPTH_THRESHOLD_M, -3.0 * DEPTH_THRESHOLD_M, 0.0])))
    num_points_before: int = reconstruction.num_points3D()
    assert filter_degenerate_points(observation_manager, reconstruction, DEPTH_THRESHOLD_M) == 0
    assert reconstruction.num_points3D() == num_points_before

    doomed: int = sorted(reconstruction.points3D)[0]
    xyz: Float64[ndarray, "3"] = np.asarray(reconstruction.point3D(doomed).xyz, dtype=np.float64)
    _corrupt_point(reconstruction, doomed, xyz + np.array([0.0, 0.0, 2.0 * DEPTH_THRESHOLD_M]))
    assert filter_degenerate_points(observation_manager, reconstruction, DEPTH_THRESHOLD_M) > 0
    assert doomed not in reconstruction.points3D
    assert reconstruction.num_points3D() == num_points_before - 1


def test_projection_failure_guard_drops_a_point_pushed_into_a_camera(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A point moved a micron in front of a camera loses its observations.

    This is RoboCap's failure mode exactly: 4 points of 209 905 ended up 5-90 mm
    from a camera centre, where the projection diverges and the reprojection error
    reaches 4.5e153 px, while the world position stays perfectly ordinary and
    every coordinate bound is satisfied.
    """
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager
    bound_px: float = projection_sanity_bound_px(reconstruction)
    assert bound_px > 0.0

    point3D_ids: list[int] = sorted(reconstruction.points3D)
    doomed: int = point3D_ids[0]
    survivor: int = point3D_ids[1]
    survivor_track_length: int = reconstruction.point3D(survivor).track.length()
    observer: pycolmap.Image = reconstruction.image(reconstruction.point3D(doomed).track.elements[0].image_id)
    _corrupt_point(reconstruction, doomed, observer.cam_from_world().inverse() * np.array([0.5, 0.5, 1e-6]))
    reconstruction.update_point_3d_errors()
    assert reconstruction.point3D(doomed).error > bound_px

    removed: int = filter_projection_failures(observation_manager, reconstruction)

    assert removed > 0
    assert doomed not in reconstruction.points3D
    assert reconstruction.point3D(survivor).track.length() == survivor_track_length
    reconstruction.update_point_3d_errors()
    assert all(point.error <= bound_px for point in reconstruction.points3D.values())


def test_projection_failure_guard_is_a_no_op_on_a_healthy_model(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """Sub-pixel observations are nowhere near an image width, so nothing is dropped."""
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager
    num_observations_before: int = reconstruction.compute_num_observations()

    assert filter_projection_failures(observation_manager, reconstruction) == 0
    assert reconstruction.compute_num_observations() == num_observations_before


def _solve(reconstruction: pycolmap.Reconstruction, correspondences: Correspondences, depth_threshold: float) -> BundleAdjustmentSolve:
    gauge_frame_id: int = reconstruction.image(min(reconstruction.reg_image_ids())).frame_id
    return solve_bundle_adjustment(
        reconstruction, pycolmap.BundleAdjustmentOptions(), gauge_frame_id, None, correspondences.observation_manager, depth_threshold
    )


def test_the_solve_boundary_refuses_an_unusable_bundle_adjustment(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A NaN point fails Ceres' first evaluation: the boundary raises rather than hand the caller that model."""
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    _corrupt_point(reconstruction, sorted(reconstruction.points3D)[0], np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(SolverFailure, match="bundle adjustment"):
        _solve(reconstruction, correspondences, DEPTH_THRESHOLD_M)


def test_the_solve_boundary_guards_the_points_it_returns(synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig) -> None:
    """Every caller (rounds, polish, extrinsic refinement) gets the post-solve guards from the boundary itself.

    A bound inside the scene stands in for a solve that walked points out of the world: the
    boundary removes them after the solve and reports how many observations went.
    """
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    extent: float = float(np.median(list(_nearest_camera_m(reconstruction).values())))
    observations_before: int = reconstruction.compute_num_observations()
    solve: BundleAdjustmentSolve = _solve(reconstruction, correspondences, extent)
    assert solve.summary.is_solution_usable()
    assert solve.num_diverged > 0
    assert reconstruction.compute_num_observations() == observations_before - solve.num_diverged
    assert max(_nearest_camera_m(reconstruction).values()) <= extent


def _nearest_camera_m(reconstruction: pycolmap.Reconstruction) -> dict[int, float]:
    """Each point's distance to its nearest observing camera centre, metres (the degeneracy guard's measure)."""
    nearest: dict[int, float] = {}
    for point3D_id, point in reconstruction.points3D.items():
        centres = [np.asarray(reconstruction.image(element.image_id).projection_center()) for element in point.track.elements]
        nearest[int(point3D_id)] = float(min(np.linalg.norm(np.asarray(point.xyz) - centre) for centre in centres))
    return nearest


def _off_axis_deg(reconstruction: pycolmap.Reconstruction, image_ids: list[int]) -> list[float]:
    """Every observation's angle off its camera's optical axis, degrees, over the given images."""
    angles: list[float] = []
    for image_id in image_ids:
        image = reconstruction.image(image_id)
        cam_from_world = image.cam_from_world()
        for point2D in image.points2D:
            if point2D.has_point3D():
                xyz = cam_from_world * reconstruction.point3D(point2D.point3D_id).xyz
                angles.append(float(np.degrees(np.arctan2(np.hypot(xyz[0], xyz[1]), xyz[2]))))
    return angles


def test_the_pole_guard_survives_tracks_it_deletes_and_counts_every_observation_removed(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a pole close enough to cut most tracks, deleting one observation also deletes whole points.

    The first version deleted the rows it had listed up front and failed on a point an earlier deletion had already
    removed (`Point3D with ID -1`); every listed row is now checked again before its deletion. The count is the
    model's observation count before minus after: a deletion that takes a two-observation track removes both.
    """
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    monkeypatch.setattr(point_filters, "rational_pole_angle", lambda camera: np.radians(5.0))
    before: int = reconstruction.compute_num_observations()
    removed: int = point_filters.filter_past_rational_pole(correspondences.observation_manager, reconstruction)
    assert removed > 0
    assert removed == before - reconstruction.compute_num_observations()
    assert max(_off_axis_deg(reconstruction, list(reconstruction.reg_image_ids()))) < 5.0


def test_new_extrinsics_that_move_observations_past_the_pole_never_reach_the_adjuster(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extrinsic refinement writes new extrinsics, then solves: the boundary guards before it builds the adjuster.

    One camera's `sensor_from_rig` is turned so that part of its observations land past a pole every observation was
    short of before. The adjuster is handed none of them, and the solve reports every observation its guards removed.
    """
    from colsfm.reconstruction import RIG_ID, camera_sensor_id

    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    camera_id: int = reconstruction.image(min(reconstruction.reg_image_ids())).camera_id
    image_ids: list[int] = [image_id for image_id in reconstruction.reg_image_ids() if reconstruction.image(image_id).camera_id == camera_id]
    pole_deg: float = max(_off_axis_deg(reconstruction, image_ids)) + 1.0
    monkeypatch.setattr(point_filters, "rational_pole_angle", lambda camera: np.radians(pole_deg) if camera.camera_id == camera_id else 0.5 * np.pi)
    rig: pycolmap.Rig = reconstruction.rig(RIG_ID)
    sensor: pycolmap.sensor_t = camera_sensor_id(camera_id)
    turn = pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([0.0, np.radians(0.5 * pole_deg), 0.0])), np.zeros(3))
    sensor_from_rig: pycolmap.Rigid3d | None = rig.sensor_from_rig(sensor)
    assert sensor_from_rig is not None
    rig.set_sensor_from_rig(sensor, turn * sensor_from_rig)
    assert max(_off_axis_deg(reconstruction, image_ids)) > pole_deg
    handed: list[float] = []
    build = pycolmap.create_default_bundle_adjuster

    def adjuster(
        options: pycolmap.BundleAdjustmentOptions, config: pycolmap.BundleAdjustmentConfig, model: pycolmap.Reconstruction
    ) -> pycolmap.BundleAdjuster:
        handed.extend(_off_axis_deg(model, image_ids))
        return build(options, config, model)

    monkeypatch.setattr(pycolmap, "create_default_bundle_adjuster", adjuster)
    before: int = reconstruction.compute_num_observations()
    solve: BundleAdjustmentSolve = _solve(reconstruction, correspondences, DEPTH_THRESHOLD_M)
    assert handed and max(handed) < pole_deg
    assert solve.num_past_pole_before > 0
    assert solve.num_removed == before - reconstruction.compute_num_observations()
