"""Scoring and reporting keep going on trajectories that determine no similarity: they do not move, or barely overlap.

Before simplecv's Umeyama took over, colsfm's own fit held the scale at 1.0 for a source that did not move. These tests
pin that the reports still score such a run (stationary, rotating in place), that the overview reports one or two
overlapping poses as unscored instead of aborting, and that a small but moving trajectory keeps its scale.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from colsfm.alignment import RigTrack
from colsfm.benchmark import compute_pose_delta
from colsfm.catalog import CatalogTrajectory
from colsfm.overview import trajectory_error
from colsfm.rigid_fit import align_rigid
from colsfm.trajectory import score_positions


def _targets() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]])


def test_a_stationary_trajectory_is_scored_with_a_unit_scale() -> None:
    """Five identical positions: the rigid fit puts them on the targets' centroid and reports the spread around it."""
    source = np.full((5, 3), 4.0)
    targets = _targets()
    spread = float(np.sqrt((np.linalg.norm(targets - targets.mean(axis=0), axis=1) ** 2).mean()))
    for estimate_scale in (False, True):
        fit = align_rigid(source, targets, estimate_scale=estimate_scale)
        assert fit.scale == 1.0 and fit.would_be_scale == 1.0
        assert fit.rmse_meters == pytest.approx(spread)
    score = score_positions(source, targets)
    assert score.num_matched == 5
    assert score.rigid_rmse_millimeters == pytest.approx(1000.0 * spread)


def test_a_rig_rotating_in_place_still_gets_a_pose_delta() -> None:
    """Same position throughout, turning about z: the comparison reports the rotation difference instead of aborting."""
    times = np.arange(5, dtype=np.int64) * 1000
    positions = np.zeros((5, 3))
    turning = Rotation.from_euler("z", np.linspace(0.0, 90.0, 5)[:, None], degrees=True).as_matrix()
    offset = Rotation.from_euler("z", 2.0, degrees=True).as_matrix()
    track_a = RigTrack(timestamps_microseconds=times, world_t_rig=positions, world_R_rig=turning)
    track_b = RigTrack(timestamps_microseconds=times, world_t_rig=positions.copy(), world_R_rig=np.einsum("ij,njk->nik", offset, turning))
    delta = compute_pose_delta(track_a, track_b)
    assert delta.num_matched == 5
    assert delta.position_rmse_millimeters == pytest.approx(0.0, abs=1e-9)
    assert delta.rotation_rmse_degrees == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("overlap", [1, 2])
def test_one_or_two_poses_inside_the_ground_truth_are_unscored(overlap: int) -> None:
    """The overview's error is unscored with no fit, and its table says so, when fewer than three poses overlap."""
    ground_truth = CatalogTrajectory(
        t_ns=np.array([0, 10, 20], dtype=np.int64), rotations=Rotation.identity(3), translations=_targets()[:3].copy()
    )
    times = np.array({1: [5, 25, 35], 2: [5, 15, 25]}[overlap], dtype=np.int64)
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    error, fit = trajectory_error(times, poses, ground_truth)
    assert fit is None
    assert error.num_matched == overlap
    assert error.se3_rmse_m is None and error.sim3_rmse_m is None and error.sim3_scale is None
    assert error.table_cells() == f"unscored | unscored | – | {overlap}"


def test_a_small_moving_trajectory_keeps_its_scale() -> None:
    """Micrometre motion is still motion: the similarity fit estimates its scale rather than holding it at 1.0."""
    rng = np.random.default_rng(3)
    source = 1e-6 * rng.normal(size=(8, 3))
    fit = align_rigid(source, 3.0 * source + 1.0, estimate_scale=True)
    assert fit.scale == pytest.approx(3.0)
    assert fit.rmse_meters == pytest.approx(0.0, abs=1e-12)
