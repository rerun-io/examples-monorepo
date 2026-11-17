"""The .insv converter: the KB4 fit of the factory lens curve, and the recording the feed reads."""

import os
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

pytest.importorskip("dataforge", reason="the converter writes through dataforge's exoego:v2 helpers (linux-only)")

from slam_rs.apis.insv import X4_AIR, Config, Kb4Fit, angle_to_nearest_permutation_deg, kb4_fit, main, orthonormal

CLIP: Path | None = Path(clip) if (clip := os.environ.get("SLAM_RS_INSV")) else None
"""The X4 Air clip VID_20261116_125200_00_003.insv, from ``SLAM_RS_INSV``; integration tests skip without it."""


def test_kb4_fit_recovers_a_kb4_curve_exactly() -> None:
    theta: Float64[ndarray, " n"] = np.linspace(0.0, np.deg2rad(97.0), 500)
    k: Float64[ndarray, " 4"] = np.array([0.07, -0.03, 0.003, -0.0005])
    radius: Float64[ndarray, " n"] = 550.0 * (theta + k[0] * theta**3 + k[1] * theta**5 + k[2] * theta**7 + k[3] * theta**9)
    fit: Kb4Fit = kb4_fit(theta, radius)
    assert fit.focal_px == pytest.approx(550.0, abs=1e-6)
    np.testing.assert_allclose(fit.k, k, atol=1e-9)
    assert fit.max_residual_px < 1e-6


def test_kb4_fit_of_the_factory_v3_curve_of_lens_0_is_within_a_few_hundredths_of_a_pixel() -> None:
    """Lens 0 of the X4 Air at 1920 px (recon/03 section 5.3): Mei xi 2.31494, mean focal 1834.85, V3 k1..k3."""
    theta: Float64[ndarray, " n"] = np.linspace(0.0, np.deg2rad(97.0), 2000)
    m: Float64[ndarray, " n"] = np.sin(theta) / (np.cos(theta) + 2.31494)
    radius: Float64[ndarray, " n"] = 0.5 * (7095.54 + 7093.97) * 1920 / 7424 * m * (1 + 0.96273077 * m**2 - 2.06431413 * m**4 + 4.31214333 * m**6)
    fit: Kb4Fit = kb4_fit(theta, radius)
    assert fit.focal_px == pytest.approx(553.5, abs=0.05)
    assert fit.max_residual_px < 0.05


def test_the_measured_imu_rotation_orthonormalises_to_0_13_deg_from_its_permutation() -> None:
    """recon/09: R_P_imu is 0.13 deg from x_P = -z_imu, y_P = +x_imu, z_P = -y_imu."""
    rotation: Float64[ndarray, "3 3"] = orthonormal(X4_AIR.pano_from_imu_measured)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.abs(rotation - X4_AIR.pano_from_imu_measured).max() < 1e-4
    assert angle_to_nearest_permutation_deg(rotation) == pytest.approx(0.13, abs=0.02)
    permutation: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert angle_to_nearest_permutation_deg(permutation) == pytest.approx(0.0, abs=1e-6)


def test_orthonormal_returns_a_proper_rotation_for_a_reflection_free_input() -> None:
    noisy: Float64[ndarray, "3 3"] = np.eye(3) + np.array([[0.0, 0.01, 0.0], [0.0, 0.0, 0.02], [0.005, 0.0, 0.0]])
    rotation: Float64[ndarray, "3 3"] = orthonormal(noisy)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.integration
def test_the_first_three_seconds_convert_into_a_recording_the_feed_reads(tmp_path: Path) -> None:
    if CLIP is None or not CLIP.is_file():
        pytest.skip(f"asset VID_20261116_125200_00_003.insv is absent: set SLAM_RS_INSV to the clip (now {CLIP})")
    from rerun.chunk import RrdReader

    from slam_rs.catalog_feed import LocalSegment, open_segment

    output: Path = tmp_path / "clip.rrd"
    main(Config(insv=CLIP, output=output, max_frames=90))
    columns: set[str] = set()
    for chunk in RrdReader(output).stream():
        columns.update(f"{chunk.entity_path}:{name}" for name in chunk.to_record_batch().schema.names)
    for column in (
        "/world/rig_00:reference",
        "/world/rig_00/imu_00:applied_time_shift_ns",
        "/world/rig_00/imu_00:simplecv.ImuCalibration:source",
        "/world/rig_00/imu_00/gyro:Scalars:scalars",
        "/world/rig_00/imu_00/accel:Scalars:scalars",
        "/world/rig_00/cam_00:Transform3D:mat3x3",
        "/world/rig_00/cam_01/pinhole:Pinhole:image_from_camera",
        "/world/rig_00/cam_01/pinhole:simplecv.components.DistortionModel",
        "/world/rig_00/cam_00/pinhole/video:VideoStream:sample",
        "/world/rig_00/cam_01/pinhole/video:VideoStream:codec",
    ):
        assert column in columns, column

    from slam_rs import _core

    lenses: list[_core.InsvLens] = _core.InsvClip(str(CLIP)).lenses_v6
    with open_segment(LocalSegment(output), include_ground_truth=False) as feed:
        lens0, lens1 = feed.cameras
        assert (lens0.model, lens0.width, lens0.height) == ("kb4", 1920, 1920)
        assert (lens0.cx, lens0.cy) == pytest.approx((lenses[0].cx, lenses[0].cy), abs=1e-3)
        assert (lens1.cx, lens1.cy) == pytest.approx((lenses[1].cx, lenses[1].cy), abs=1e-3)
        assert (lens0.fx, lens1.fx) == pytest.approx((553.5, 557.7), abs=0.1)
        # The rig frame is the IMU: the lens centres sit 33.3 mm apart (V6 string; V3 says 33.1 mm) along the lens
        # axis, z_P = -y_imu (recon/09).
        baseline: Float64[ndarray, " 3"] = lens1.imu_T_cam[:3, 3] - lens0.imu_T_cam[:3, 3]
        assert np.linalg.norm(baseline) == pytest.approx(0.03329, abs=1e-4)
        assert baseline[1] > 0.99 * np.linalg.norm(baseline), "lens 1 sits behind lens 0: +y_imu"
        # Lens 0 looks along z_P = -y_imu.
        assert lens0.imu_T_cam[1, 2] == pytest.approx(-1.0, abs=1e-3)
        assert feed.imu.cam_time_offset_ns == 0, "IMU times are the trailer's, no gyro_timestamp shift"
        assert feed.insta360 is not None
        assert feed.insta360.readout_s == pytest.approx(0.015883)
        assert feed.insta360.window_crop == (7680, 7680, 7424, 7424, 0, 0)
        assert len(feed.frame_t_ns) == 90
