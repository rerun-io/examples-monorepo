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

from colsfm.config import CusfmConfig
from colsfm.mapping import (
    Correspondences,
    filter_degenerate_points,
    filter_projection_failures,
    load_correspondences,
    projection_sanity_bound_px,
)
from colsfm.reconstruction import build_reconstruction


@pytest.fixture(scope="module")
def synthetic_database(synthetic_rig: SyntheticRig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic rig's observations as a COLMAP database."""
    database_path: Path = tmp_path_factory.mktemp("synthetic") / "database.db"
    write_database(
        database_path, synthetic_rig.frames_meta, synthetic_rig.keypoints_px, synthetic_matches(synthetic_rig)
    )
    return database_path

DEPTH_THRESHOLD_M: float = 300.0
"""`vision_mapping_config.depth_threshold`, the isaac profile's world-Z cap in metres."""


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
    correspondences: Correspondences = load_correspondences(
        reconstruction, database_path, config.vision_mapping.min_num_matches_per_pair
    )
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


def test_degeneracy_guard_drops_points_ceres_cannot_survive(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
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


def test_degeneracy_guard_is_a_no_op_on_a_healthy_model(
    synthetic_rig: SyntheticRig, synthetic_database: Path, isaac_config: CusfmConfig
) -> None:
    """A model whose points all sit inside the cap loses nothing to the guard."""
    reconstruction, correspondences = _triangulated(synthetic_rig, synthetic_database, isaac_config)
    observation_manager: pycolmap.ObservationManager = correspondences.observation_manager
    num_points_before: int = reconstruction.num_points3D()

    assert filter_degenerate_points(observation_manager, reconstruction, DEPTH_THRESHOLD_M) == 0
    assert reconstruction.num_points3D() == num_points_before


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
