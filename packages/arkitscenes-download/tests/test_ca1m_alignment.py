"""Behavior checks for CA-1M clock mapping and rigid alignment."""

import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from arkitscenes_download.ca1m.alignment import diagnose_clock, rigid_umeyama


def test_rigid_umeyama_recovers_known_transform_with_noise() -> None:
    """A noisy proper rigid transform is recovered without estimating scale."""
    rng: np.random.Generator = np.random.default_rng(42898570)
    faro_xyz: np.ndarray = rng.normal(size=(200, 3)).astype(np.float64)
    expected_rotation_33: np.ndarray = Rotation.from_euler("xyz", [17.0, -8.0, 31.0], degrees=True).as_matrix()
    expected_translation_xyz: np.ndarray = np.asarray([1.2, -0.7, 0.35], dtype=np.float64)
    arkit_xyz: np.ndarray = (expected_rotation_33 @ faro_xyz.T).T + expected_translation_xyz
    arkit_xyz += rng.normal(scale=1e-4, size=arkit_xyz.shape)

    alignment = rigid_umeyama(faro_xyz, arkit_xyz)

    np.testing.assert_allclose(alignment.rotation_33, expected_rotation_33, atol=3e-5)
    np.testing.assert_allclose(alignment.translation_xyz, expected_translation_xyz, atol=3e-5)
    assert alignment.rms_m < 2e-4


def test_rigid_umeyama_rejects_reflection_solution() -> None:
    """Reflected correspondences still produce a proper rotation."""
    faro_xyz: np.ndarray = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64
    )
    reflection_33: np.ndarray = np.diag(np.asarray([-1.0, 1.0, 1.0], dtype=np.float64))
    arkit_xyz: np.ndarray = (reflection_33 @ faro_xyz.T).T + np.asarray([4.0, -2.0, 1.0], dtype=np.float64)

    alignment = rigid_umeyama(faro_xyz, arkit_xyz)

    assert np.linalg.det(alignment.rotation_33) == pytest.approx(1.0, abs=1e-12)
    assert alignment.rms_m > 0.0


def test_clock_diagnostic_warns_when_threshold_is_exceeded() -> None:
    """A broken clock mapping sets the should_warn flag for the ingest boundary."""
    gt_times_s: np.ndarray = np.asarray([0.100, 0.110, 0.120], dtype=np.float64)
    calibration_times_s: np.ndarray = np.asarray([0.0, 0.010, 0.020], dtype=np.float64)

    diagnostics = diagnose_clock(gt_times_s, calibration_times_s)

    assert diagnostics.median_delta_s > 0.002
    assert diagnostics.fraction_over_10ms > 0.05
    assert diagnostics.should_warn


def test_clock_diagnostic_accepts_nearby_samples() -> None:
    """Sub-millisecond matches remain below every warning threshold."""
    gt_times_s: np.ndarray = np.asarray([0.0004, 0.0103, 0.0198], dtype=np.float64)
    calibration_times_s: np.ndarray = np.asarray([0.0, 0.010, 0.020], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        diagnostics = diagnose_clock(gt_times_s, calibration_times_s)

    assert diagnostics.median_delta_s < 0.002
    assert diagnostics.fraction_over_10ms == 0.0
    assert not diagnostics.should_warn


def test_rigid_umeyama_flags_collinear_trajectories() -> None:
    """Rank-1 motion reports a tiny second extent even when the residual is zero."""
    line_xyz: np.ndarray = np.stack([np.linspace(0.0, 2.0, 8), np.zeros(8), np.zeros(8)], axis=1)

    alignment = rigid_umeyama(line_xyz, line_xyz.copy())

    assert alignment.rms_m == pytest.approx(0.0, abs=1e-12)
    assert alignment.source_extent2_m == pytest.approx(0.0, abs=1e-12)


def test_rigid_umeyama_reports_planar_extent() -> None:
    """Well-spread planar motion keeps a healthy second extent."""
    rng = np.random.default_rng(7)
    plane_xyz: np.ndarray = np.column_stack([rng.uniform(-1.0, 1.0, 64), rng.uniform(-1.0, 1.0, 64), np.zeros(64)])

    alignment = rigid_umeyama(plane_xyz, plane_xyz.copy())

    assert alignment.source_extent2_m > 0.3
