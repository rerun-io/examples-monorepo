"""The Insta360 calibration tool: the epipolar pair fit, the IMU-to-rig rotation and both frame-time estimators.

The synthetic rig rotates and translates through a static cloud of points seen on the whole sphere. Its gyro
carries a bias and white noise; its images are taken ``EXTRA_OFFSET_MS`` after their stamps, the error the
tool must find. The bearings are expressed in the rig frame P (camera 0's axes), the gyro in the IMU frame.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fixture_types import FRAME, RigFactory
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from slam_rs import _core
from slam_rs.apis.insv_calibrate import (
    Config,
    TrackedFrame,
    WindowScan,
    fit_hand_eye,
    main,
    nearest_axis_permutation,
    offset_summary,
    parabola_minimum,
    rate_offset_curve,
    relative_pose,
    rotation_angle_deg,
    scan_window,
    visual_pairs,
)
from slam_rs.catalog_feed import Frameset, LocalSegment, RigProfile, open_segment
from slam_rs.catalog_timing import ImuStream
from slam_rs.config import load_slam_config

PERMUTATION: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""``R_P_imu`` of the X4 Air up to a fraction of a degree."""
TRUE_PANO_R_IMU: Float64[ndarray, "3 3"] = Rotation.from_rotvec(np.radians([0.6, -0.4, 0.8])).as_matrix() @ PERMUTATION
BIAS: Float64[ndarray, " 3"] = np.array([0.004, -0.003, 0.002])
EXTRA_OFFSET_MS: float = 3.0
PIXEL_RAD: float = np.pi / 960.0
"""About two lens pixels at the centre of a 1920 px X4 Air lens, radians."""


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
        world_R_imu.append(world_R_imu[-1] @ Rotation.from_rotvec(rate * 1e-3).as_matrix())
    points: Float64[ndarray, "n_points 3"] = rng.normal(size=(500, 3))
    points *= rng.uniform(2.0, 8.0, size=(500, 1)) / np.linalg.norm(points, axis=1, keepdims=True)
    frames: list[TrackedFrame] = []
    for stamp_s in np.arange(0.2, seconds, 1 / 30):
        true_s: float = stamp_s + EXTRA_OFFSET_MS * 1e-3
        index: int = int(true_s * 1e3)
        rest_s: float = true_s - index * 1e-3
        world_R_frame: Float64[ndarray, "3 3"] = world_R_imu[index] @ Rotation.from_rotvec(angular_rate(np.array([index * 1e-3 + rest_s / 2]))[0] * rest_s).as_matrix()
        centre: Float64[ndarray, " 3"] = np.array([np.sin(0.5 * true_s), 0.5 * np.sin(0.3 * true_s), 0.2 * true_s])
        rays: Float64[ndarray, "n_points 3"] = (points - centre) @ world_R_frame @ TRUE_PANO_R_IMU.T
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True) + rng.normal(scale=0.5 * PIXEL_RAD, size=rays.shape)
        frames.append(TrackedFrame(t_ns=int(round(stamp_s * 1e9)), ids=np.arange(500), bearings=rays / np.linalg.norm(rays, axis=1, keepdims=True)))
    gyro: Float64[ndarray, "n_imu 3"] = rates + BIAS + rng.normal(scale=0.0028 * np.sqrt(1e3), size=rates.shape)
    imu_t_ns: Int64[ndarray, " n_imu"] = np.round(imu_t_s * 1e9).astype(np.int64)
    return frames, ImuStream(t_ns=imu_t_ns, gyro_rad_s=gyro, accel_m_s2=np.zeros_like(gyro))


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
    fit = relative_pose(rays0, rays1, Rotation.from_rotvec(np.radians([2.0, -3.0, 4.0])).as_matrix() @ rotation)
    assert rotation_angle_deg(fit.rotation @ rotation.T) < 1e-6
    # t = -R c = (0, 1, 0) up to sign.
    assert abs(fit.direction @ np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0, abs=1e-9)


def test_the_rotation_and_the_rate_offset_are_recovered_on_a_synthetic_rig() -> None:
    frames, imu = synthetic_rig(seconds=10.0, seed=3)
    current: Float64[ndarray, "3 3"] = Rotation.from_rotvec(np.radians([1.0, 0.0, -1.0])).as_matrix() @ TRUE_PANO_R_IMU
    pairs = visual_pairs(frames, imu, current)
    offsets_ms: tuple[float, ...] = tuple(float(value) for value in range(-8, 9))
    curve: Float64[ndarray, " n_offsets"] = rate_offset_curve(pairs, imu, fit_hand_eye(pairs, imu, 0.0), offsets_ms)
    offset_ms: float = parabola_minimum(np.asarray(offsets_ms), curve)
    assert offset_ms == pytest.approx(EXTRA_OFFSET_MS, abs=0.5)
    fit = fit_hand_eye(pairs, imu, offset_ms)
    assert rotation_angle_deg(fit.pano_R_imu @ TRUE_PANO_R_IMU.T) < 0.1
    np.testing.assert_allclose(fit.bias_rad_s, BIAS, atol=3e-3)
    np.testing.assert_array_equal(nearest_axis_permutation(fit.pano_R_imu), PERMUTATION)


def test_bearings_come_from_the_frontend_cameras(rig: RigFactory) -> None:
    # kb4 with zero coefficients, f = 100 px: 100 px right of the centre is 1 rad off the axis.
    vio: _core.Vio = _core.Vio(rig(2), _core.VioConfig(), backend="cpu")
    pixels: Float32[ndarray, "3 2"] = np.array([[FRAME / 2, FRAME / 2], [FRAME / 2 + 100.0, FRAME / 2], [FRAME / 2, FRAME / 2 - 100.0]], dtype=np.float32)
    expected: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, 1.0], [np.sin(1.0), 0.0, np.cos(1.0)], [0.0, -np.sin(1.0), np.cos(1.0)]])
    for camera in range(2):
        np.testing.assert_allclose(vio.bearings(camera, pixels), expected, atol=1e-6)
    with pytest.raises(IndexError):
        vio.bearings(2, pixels)


def test_the_parabola_finds_a_minimum_between_grid_points_and_stops_at_an_edge() -> None:
    offsets: Float64[ndarray, " 5"] = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert parabola_minimum(offsets, 1.0 + (offsets - 0.3) ** 2) == pytest.approx(0.3, abs=1e-9)
    assert parabola_minimum(offsets, offsets) == -2.0


def test_the_offset_summary_averages_the_window_minima_and_skips_missing_residuals() -> None:
    rng: np.random.Generator = np.random.default_rng(7)
    offsets_ms: tuple[float, ...] = tuple(float(value) for value in range(-8, 9))
    grid: Float64[ndarray, " n_offsets"] = np.asarray(offsets_ms)
    scans: list[WindowScan] = []
    for start_s in range(6):
        # The solver residual doubles 5 ms from its minimum; a rejected window reports no residual.
        residual: Float64[ndarray, " n_offsets"] = 2e-3 * (1 + ((grid + 2.0) / 5.0) ** 2) * (1 + rng.normal(scale=0.03, size=grid.size))
        values: list[float | None] = [None if index == 16 else float(value) for index, value in enumerate(residual)]
        scans.append(WindowScan(start_s=float(start_s), offsets_ms=offsets_ms, residual_rad=tuple(values), verdicts=("Accepted",) * grid.size))
    mean_ms, spread_ms = offset_summary(scans)
    assert mean_ms == pytest.approx(-2.0, abs=0.5)
    assert spread_ms < 1.0


CLIP_RRD: Path | None = Path(clip) if (clip := os.environ.get("SLAM_RS_INSTA360_RRD")) else None
"""The converted Insta360 clip VID_20261116_125200_00_003.rrd, from ``SLAM_RS_INSTA360_RRD``; the integration test skips without it."""


@pytest.mark.integration
def test_the_real_clip_gives_the_converters_rotation() -> None:
    if CLIP_RRD is None or not CLIP_RRD.is_file():
        pytest.skip(f"the converted Insta360 clip VID_20261116_125200_00_003.rrd is absent: set SLAM_RS_INSTA360_RRD (now {CLIP_RRD})")
    report = main(Config(rrd=CLIP_RRD, rotation_spans_s=(40.0,), span_s=20.0, scan_windows_s=()))
    assert report.rotation.spans[0].angle_to_current_deg < 0.5


@pytest.mark.golden
def test_the_bootstrap_scan_finds_a_known_shift_of_the_frame_times_on_the_real_clip() -> None:
    """Estimator (b) through ``Vio`` on one 2.3 s window of the clip: frame times shifted by a known offset move the
    scan's minimum by minus that offset, with the recording's readout, none and the sign flipped. There is no synthetic
    version: the scan runs the native two-lens rig, whose inertial bootstrap needs two lens images of a moving 3D scene
    with matching IMU, and the unit fixtures give only a still rig with flat textures, on which no offset is observable.
    """
    if CLIP_RRD is None or not CLIP_RRD.is_file():
        pytest.skip(f"the converted Insta360 clip VID_20261116_125200_00_003.rrd is absent: set SLAM_RS_INSTA360_RRD (now {CLIP_RRD})")
    shift_ns: int = 3_000_000
    offsets_ms: tuple[float, ...] = tuple(float(value) for value in range(-8, 9, 2))
    vio_config: _core.VioConfig = _core.VioConfig.from_json(load_slam_config().vio_config_text("insta360", profile="fast"))
    with open_segment(LocalSegment(CLIP_RRD), profile=RigProfile(decode_threads=4), window_s=1.0, include_ground_truth=False) as feed:
        start_ns: int = int(feed.frame_t_ns[0]) + 60_000_000_000
        stop_ns: int = start_ns + 2_300_000_000
        framesets: list[Frameset] = [frameset for frameset in feed.framesets(stop_ns, start_ns) if frameset.t_ns <= stop_ns]
        imu: ImuStream = feed.imu_between(start_ns - 500_000_000, stop_ns + 500_000_000)
        shifted: list[Frameset] = [replace(frameset, t_ns=frameset.t_ns + shift_ns) for frameset in framesets]
        readout_s: float | None = feed.cameras[0].readout_s
        assert readout_s is not None
        minima: dict[float, tuple[float | None, float | None]] = {}
        for readout in (readout_s, 0.0, -readout_s):
            base, _ = scan_window(feed, vio_config, "cpu", framesets, imu, 60.0, offsets_ms, readout)
            moved, _ = scan_window(feed, vio_config, "cpu", shifted, imu, 60.0, offsets_ms, readout)
            minima[readout] = (base.minimum_ms(), moved.minimum_ms())
    print(f"readout -> (minimum, minimum with the frames {shift_ns * 1e-6:+.1f} ms later), ms: {minima}")
    for readout, (base_ms, moved_ms) in minima.items():
        assert base_ms is not None and moved_ms is not None, f"readout {readout}: the scan found no minimum"
        assert moved_ms - base_ms == pytest.approx(-shift_ns * 1e-6, abs=1.0), f"readout {readout}"
