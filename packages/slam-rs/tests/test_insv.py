"""The .insv converter: the KB4 fit of the factory lens curve, and the recording the feed reads."""

import os
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

pytest.importorskip("dataforge", reason="the converter writes through dataforge's exoego:v2 helpers (linux-only)")

from slam_rs.apis.insv import Config, Kb4Fit, kb4_fit, main

CLIP: Path = Path(os.environ.get("SLAM_RS_INSV", "/home/pablo/data/insta360/VID_20261116_125200_00_003.insv"))
"""The X4 Air clip VID_20261116_125200_00_003.insv; integration tests skip when it is absent."""


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


@pytest.mark.integration
def test_the_first_three_seconds_convert_into_a_recording_the_feed_reads(tmp_path: Path) -> None:
    if not CLIP.is_file():
        pytest.skip(f"asset {CLIP} (VID_20261116_125200_00_003.insv, set SLAM_RS_INSV) is absent")
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

    with open_segment(LocalSegment(output), include_ground_truth=False) as feed:
        lens0, lens1 = feed.cameras
        assert (lens0.model, lens0.width, lens0.height) == ("kb4", 1920, 1920)
        assert (lens0.cx, lens0.cy) == pytest.approx((960.242, 965.681), abs=1e-3)
        assert (lens1.cx, lens1.cy) == pytest.approx((966.509, 965.847), abs=1e-3)
        assert (lens0.fx, lens1.fx) == pytest.approx((553.5, 557.7), abs=0.05)
        # The rig origin is the midpoint: the lens centres sit 33.1 mm apart, along the optical axis.
        baseline: Float64[ndarray, " 3"] = lens1.imu_T_cam[:3, 3] - lens0.imu_T_cam[:3, 3]
        assert np.linalg.norm(baseline) == pytest.approx(0.0331, abs=1e-4)
        assert abs(baseline[2]) > 0.99 * np.linalg.norm(baseline)
        assert feed.imu.cam_time_offset_ns == 1_600_000
        assert len(feed.frame_t_ns) == 90
