"""The Insta360 calibration tool: the epipolar pair fit, the IMU-to-rig rotation and both frame-time estimators.

The synthetic rig rotates and translates through a static cloud of points seen on the whole sphere. Its gyro
carries a bias and white noise; its images are taken ``EXTRA_OFFSET_MS`` after their stamps, the error the
tool must find. The bearings are expressed in the panorama frame P, the gyro in the IMU frame.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs.apis.insv_calibrate import (
    BootstrapAttempt,
    Config,
    TrackedFrame,
    WindowScan,
    epipolar_residuals,
    first_attempt,
    fit_hand_eye,
    main,
    offset_summary,
    parabola_minimum,
    rate_offset_curve,
    relative_pose,
    rotation_angle_deg,
    rotation_exp,
    visual_pairs,
)
from slam_rs.catalog_timing import ImuStream

PERMUTATION: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""``R_P_imu`` of the X4 Air up to a fraction of a degree."""
TRUE_PANO_R_IMU: Float64[ndarray, "3 3"] = rotation_exp(np.radians([0.6, -0.4, 0.8])) @ PERMUTATION
BIAS: Float64[ndarray, " 3"] = np.array([0.004, -0.003, 0.002])
EXTRA_OFFSET_MS: float = 3.0
PIXEL_RAD: float = np.pi / 960.0
"""One panorama pixel, radians."""


def angular_rate(t_s: Float64[ndarray, " n"]) -> Float64[ndarray, "n 3"]:
    """Body rate in the IMU frame, rad/s: the 1-2 rad/s of a hand-held walk."""
    return np.stack([1.5 * np.sin(2 * np.pi * 0.7 * t_s), 1.0 * np.sin(2 * np.pi * 0.43 * t_s + 1.0), 0.8 * np.sin(2 * np.pi * 1.1 * t_s + 2.0)], axis=1)


def synthetic_rig(seconds: float, seed: int) -> tuple[list[TrackedFrame], ImuStream]:
    """Frames of tracked bearings in P and the IMU stream of one synthetic walk."""
    rng: np.random.Generator = np.random.default_rng(seed)
    imu_t_s: Float64[ndarray, " n_imu"] = np.arange(0.0, seconds + 0.5, 1e-3)
    rates: Float64[ndarray, "n_imu 3"] = angular_rate(imu_t_s)
    # Midpoint rule on the analytic rate: the truth must not carry the half-step lag of a rectangle rule.
    world_R_imu: list[Float64[ndarray, "3 3"]] = [np.eye(3)]
    for rate in angular_rate(imu_t_s[:-1] + 0.5e-3):
        world_R_imu.append(world_R_imu[-1] @ rotation_exp(rate * 1e-3))
    points: Float64[ndarray, "n_points 3"] = rng.normal(size=(500, 3))
    points *= rng.uniform(2.0, 8.0, size=(500, 1)) / np.linalg.norm(points, axis=1, keepdims=True)
    frames: list[TrackedFrame] = []
    for stamp_s in np.arange(0.2, seconds, 1 / 30):
        true_s: float = stamp_s + EXTRA_OFFSET_MS * 1e-3
        index: int = int(true_s * 1e3)
        rest_s: float = true_s - index * 1e-3
        world_R_frame: Float64[ndarray, "3 3"] = world_R_imu[index] @ rotation_exp(angular_rate(np.array([index * 1e-3 + rest_s / 2]))[0] * rest_s)
        centre: Float64[ndarray, " 3"] = np.array([np.sin(0.5 * true_s), 0.5 * np.sin(0.3 * true_s), 0.2 * true_s])
        rays: Float64[ndarray, "n_points 3"] = (points - centre) @ world_R_frame @ TRUE_PANO_R_IMU.T
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True) + rng.normal(scale=0.5 * PIXEL_RAD, size=rays.shape)
        frames.append(TrackedFrame(t_ns=int(round(stamp_s * 1e9)), ids=np.arange(500), bearings=rays / np.linalg.norm(rays, axis=1, keepdims=True)))
    gyro: Float64[ndarray, "n_imu 3"] = rates + BIAS + rng.normal(scale=0.0028 * np.sqrt(1e3), size=rates.shape)
    imu_t_ns: Int64[ndarray, " n_imu"] = np.round(imu_t_s * 1e9).astype(np.int64)
    return frames, ImuStream(t_ns=imu_t_ns, gyro_rad_s=gyro, accel_m_s2=np.zeros_like(gyro))


def test_the_epipolar_residual_is_the_sine_of_the_angle_off_the_epipolar_plane() -> None:
    # Translation along x, b0 on the z axis: the epipolar plane is y = 0, and t x b0 = (0, -1, 0).
    angle: float = 0.01
    residual: Float64[ndarray, " 1"] = epipolar_residuals(
        np.eye(3), np.array([1.0, 0.0, 0.0]), np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, np.sin(angle), np.cos(angle)]])
    )
    assert residual[0] == pytest.approx(-np.sin(angle), abs=1e-15)


def test_a_hand_placed_pose_is_recovered_from_a_wrong_seed() -> None:
    # The rig turns 90 degrees about z and moves 1 m along x; frame-1 rays are R (p - c).
    rotation: Float64[ndarray, "3 3"] = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    centre: Float64[ndarray, " 3"] = np.array([1.0, 0.0, 0.0])
    points: Float64[ndarray, "8 3"] = np.array(
        [[3.0, 1.0, 2.0], [-2.0, 4.0, 1.0], [0.5, -3.0, -2.0], [-4.0, -1.0, 3.0], [2.0, 2.0, -4.0], [-1.0, 5.0, -1.0], [4.0, -2.0, 1.0], [0.0, 1.0, 6.0]]
    )
    rays0: Float64[ndarray, "8 3"] = points / np.linalg.norm(points, axis=1, keepdims=True)
    rays1: Float64[ndarray, "8 3"] = (points - centre) @ rotation.T
    rays1 /= np.linalg.norm(rays1, axis=1, keepdims=True)
    fit = relative_pose(rays0, rays1, rotation_exp(np.radians([2.0, -3.0, 4.0])) @ rotation)
    assert rotation_angle_deg(fit.rotation @ rotation.T) < 1e-6
    # t = -R c = (0, 1, 0) up to sign.
    assert abs(fit.direction @ np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0, abs=1e-9)


def test_the_rotation_and_the_rate_offset_are_recovered_on_a_synthetic_rig() -> None:
    frames, imu = synthetic_rig(seconds=10.0, seed=3)
    current: Float64[ndarray, "3 3"] = rotation_exp(np.radians([1.0, 0.0, -1.0])) @ TRUE_PANO_R_IMU
    pairs = visual_pairs(frames, imu, current)
    offsets_ms: tuple[float, ...] = tuple(float(value) for value in range(-8, 9))
    curve: Float64[ndarray, " n_offsets"] = rate_offset_curve(pairs, imu, fit_hand_eye(pairs, imu, 0.0), offsets_ms)
    offset_ms: float = parabola_minimum(np.asarray(offsets_ms), curve)
    assert offset_ms == pytest.approx(EXTRA_OFFSET_MS, abs=0.5)
    fit = fit_hand_eye(pairs, imu, offset_ms)
    assert rotation_angle_deg(fit.pano_R_imu @ TRUE_PANO_R_IMU.T) < 0.1
    np.testing.assert_allclose(fit.bias_rad_s, BIAS, atol=3e-3)


def test_the_parabola_finds_a_minimum_between_grid_points_and_stops_at_an_edge() -> None:
    offsets: Float64[ndarray, " 5"] = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert parabola_minimum(offsets, 1.0 + (offsets - 0.3) ** 2) == pytest.approx(0.3, abs=1e-9)
    assert parabola_minimum(offsets, offsets) == -2.0


def test_the_bootstrap_scan_recovers_the_offset_over_noisy_windows() -> None:
    rng: np.random.Generator = np.random.default_rng(7)
    offsets_ms: tuple[float, ...] = tuple(float(value) for value in range(-8, 9))
    grid: Float64[ndarray, " n_offsets"] = np.asarray(offsets_ms)
    scans: list[WindowScan] = []
    for start_s in range(6):
        # The solver residual doubles 5 ms from its minimum (DESIGN 13.1); a rejected window reports no residual.
        residual: Float64[ndarray, " n_offsets"] = 2e-3 * (1 + ((grid + 2.0) / 5.0) ** 2) * (1 + rng.normal(scale=0.03, size=grid.size))
        values: list[float | None] = [None if index == 16 else float(value) for index, value in enumerate(residual)]
        scans.append(WindowScan(start_s=float(start_s), offsets_ms=offsets_ms, residual_rad=tuple(values), verdicts=("Accepted",) * grid.size))
    mean_ms, spread_ms = offset_summary(scans)
    assert mean_ms == pytest.approx(-2.0, abs=0.5)
    assert spread_ms < 1.0


def test_the_first_attempt_is_read_from_the_report_text() -> None:
    text: str = (
        '{"attempts":[{"first_t_ns":1,"last_t_ns":2,"frames":61,"tracks":3653,"verdict":"Accepted","accepted":true,'
        '"excitation_mps2":2.2,"tracks_used":913,"residual_rms_rad":0.0021,"condition":14.5,"gravity_norm_unconstrained":9.79,'
        '"scale_sigma":0.0028,"velocity_world":[0,0,0],"gravity_body0":[0,0,9.8],"bias_gyro":[0,0,0],"bias_accel":[0,0,0]}],'
        '"start_t_ns":1,"accepted_at_t_ns":2,"replayed":61}'
    )
    assert first_attempt(text) == BootstrapAttempt(verdict="Accepted", accepted=True, frames=61, residual_rms_rad=0.0021, scale_sigma=0.0028)
    assert first_attempt('{"attempts":[],"start_t_ns":null,"accepted_at_t_ns":null,"replayed":0}') is None


CLIP_RRD: Path = Path(os.environ.get("SLAM_RS_INSTA360_RRD", "/home/pablo/data/insta360/VID_20261116_125200_00_003.rrd"))


@pytest.mark.integration
def test_the_real_clip_gives_the_converters_rotation() -> None:
    if not CLIP_RRD.is_file():
        pytest.skip(f"the converted Insta360 clip {CLIP_RRD} (set SLAM_RS_INSTA360_RRD) is absent")
    report = main(Config(rrd=CLIP_RRD, rotation_spans_s=(40.0,), span_s=20.0, scan_windows_s=()))
    assert report.rotation.spans[0].angle_to_current_deg < 0.5
