"""The Aria VRS readers against a real LaMAria sequence.

Skipped unless R_01_easy's 897 MB VRS has been fetched into the package's
gitignored ``data/`` tree:

    packages/dataforge/data/raw/lamaria/training/R_01_easy/raw_data/R_01_easy.vrs

The load-bearing assertion is the transform chain: ``rig_T_cam`` derived from
the VRS device calibration must reproduce the published ``T_b_s``. Those two
paths share no code — one is ``imuR_T_device @ device_T_cam`` out of the
factory calibration, the other is a quaternion in a JSON file — so agreement
pins both the chain and the quaternion order.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray
from projectaria_tools.core.data_provider import VrsDataProvider
from simplecv.camera_parameters import Fisheye62Parameters

from dataforge import aria
from dataforge.logging_toolkit import ImuChannel

PACKAGE_DIR: Path = Path(__file__).parents[1]
SEQUENCE_DIR: Path = PACKAGE_DIR / "data" / "raw" / "lamaria" / "training" / "R_01_easy"
VRS_PATH: Path = SEQUENCE_DIR / "raw_data" / "R_01_easy.vrs"
PUBLISHED_CALIBRATION_PATH: Path = SEQUENCE_DIR / "aria_calibrations" / "R_01_easy.json"

pytestmark = pytest.mark.skipif(not VRS_PATH.is_file(), reason=f"no development VRS at {VRS_PATH}")

# Stream id → (frames, nominal rate in Hz), read out of R_01_easy itself.
EXPECTED_STREAMS: dict[aria.AriaStreamId, tuple[int, float]] = {
    aria.SLAM_LEFT_STREAM_ID: (2898, 20.0),
    aria.SLAM_RIGHT_STREAM_ID: (2898, 20.0),
    aria.RGB_STREAM_ID: (1449, 10.0),
    aria.IMU_RIGHT_STREAM_ID: (145162, 1000.0),
    aria.IMU_LEFT_STREAM_ID: (115081, 800.0),
}


@pytest.fixture(scope="module")
def provider() -> VrsDataProvider:
    return aria.open_vrs(VRS_PATH)


@pytest.fixture(scope="module")
def rig(provider: VrsDataProvider) -> aria.AriaRig:
    return aria.AriaRig.from_provider(provider)


def test_open_vrs_names_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no readable VRS at"):
        aria.open_vrs(tmp_path / "absent.vrs")


def test_every_expected_stream_is_present_at_its_nominal_rate(provider: VrsDataProvider) -> None:
    for stream_id, (frames, nominal_rate_hz) in EXPECTED_STREAMS.items():
        times_ns: Int64[ndarray, "n_frames"] = aria.frame_timestamps_ns(provider, stream_id)
        assert times_ns.size == frames, stream_id
        # Ascending is asserted inside frame_timestamps_ns; here the measured rate
        # has to match the nominal one within a percent, or the clock is wrong.
        measured_rate_hz: float = 1e9 * (times_ns.size - 1) / float(times_ns[-1] - times_ns[0])
        assert measured_rate_hz == pytest.approx(nominal_rate_hz, rel=0.01), stream_id


def test_slam_frames_are_gray_and_rgb_frames_are_three_channel(provider: VrsDataProvider) -> None:
    slam: list[aria.TimedImage] = list(islice(aria.iter_frames(provider, aria.SLAM_LEFT_STREAM_ID), 3))
    assert [image.shape for _, image in slam] == [(480, 640)] * 3
    assert {image.dtype for _, image in slam} == {np.dtype(np.uint8)}
    rgb: list[aria.TimedImage] = list(islice(aria.iter_frames(provider, aria.RGB_STREAM_ID), 3))
    assert [image.shape for _, image in rgb] == [(1408, 1408, 3)] * 3
    assert {image.dtype for _, image in rgb} == {np.dtype(np.uint8)}


def test_iter_frames_and_frame_timestamps_agree_frame_for_frame(provider: VrsDataProvider) -> None:
    """The encoder pairs the mp4's Nth sample with times_ns[N], so record order must match."""
    times_ns: Int64[ndarray, "n_frames"] = aria.frame_timestamps_ns(provider, aria.SLAM_RIGHT_STREAM_ID)
    walked: list[int] = [timestamp for timestamp, _ in islice(aria.iter_frames(provider, aria.SLAM_RIGHT_STREAM_ID), 200)]
    assert walked == times_ns[:200].tolist()


def test_rig_T_cam_reproduces_the_published_calibration(rig: aria.AriaRig) -> None:
    published: dict[str, aria.PublishedCamera] = aria.read_calibration_json(PUBLISHED_CALIBRATION_PATH)
    for name, stream_id in (("cam0", aria.SLAM_LEFT_STREAM_ID), ("cam1", aria.SLAM_RIGHT_STREAM_ID)):
        from_vrs: Float64[ndarray, "4 4"] = rig.cameras[stream_id].extrinsics.world_T_cam
        assert from_vrs == pytest.approx(published[name].rig_T_cam.to_matrix(), abs=1e-3), name


def test_published_intrinsics_match_the_vrs_ones(rig: aria.AriaRig) -> None:
    published: dict[str, aria.PublishedCamera] = aria.read_calibration_json(PUBLISHED_CALIBRATION_PATH)
    for name, stream_id in (("cam0", aria.SLAM_LEFT_STREAM_ID), ("cam1", aria.SLAM_RIGHT_STREAM_ID)):
        camera: Fisheye62Parameters = rig.cameras[stream_id]
        fx, fy, cx, cy = published[name].params[:4]
        assert (camera.intrinsics.fl_x, camera.intrinsics.fl_y) == pytest.approx((fx, fy), abs=1e-6), name
        assert (camera.intrinsics.cx, camera.intrinsics.cy) == pytest.approx((cx, cy), abs=1e-6), name
        assert (camera.intrinsics.width, camera.intrinsics.height) == (640, 480), name


def test_the_rig_frame_is_imu_right(rig: aria.AriaRig) -> None:
    assert rig.rig_T_imu[aria.IMU_RIGHT_STREAM_ID] == pytest.approx(np.eye(4), abs=1e-12)
    # imu-left sits a few millimetres away and is rotated: exoego:v2 needs that
    # pose on imu_01, which is why log_imu grew a rig_T_imu argument.
    rig_T_imu_left: Float64[ndarray, "4 4"] = rig.rig_T_imu[aria.IMU_LEFT_STREAM_ID]
    assert np.abs(rig_T_imu_left - np.eye(4)).max() > 0.1
    # 129.3 mm apart, which is the distance between the two IMU origins in the
    # device frame ((0.005065, -0.102204, -0.086374) vs (0.000773, -0.000515, -0.006673)).
    assert np.linalg.norm(rig_T_imu_left[:3, 3]) == pytest.approx(0.129272, abs=1e-5)
    assert rig_T_imu_left[:3, :3] @ rig_T_imu_left[:3, :3].T == pytest.approx(np.eye(3), abs=1e-9)


def test_the_rgb_camera_is_only_in_the_vrs(rig: aria.AriaRig) -> None:
    """The published calibration has no RGB entry, so the VRS is the only source for cam_02."""
    assert set(rig.cameras) == set(aria.CAMERA_STREAM_IDS)
    assert "cam2" not in aria.read_calibration_json(PUBLISHED_CALIBRATION_PATH)
    rgb: Fisheye62Parameters = rig.cameras[aria.RGB_STREAM_ID]
    assert (rgb.intrinsics.width, rgb.intrinsics.height) == (1408, 1408)


def test_imu_channels_are_raw_and_share_one_clock(provider: VrsDataProvider) -> None:
    channels: aria.ImuSamples = aria.read_imu(provider, aria.IMU_RIGHT_STREAM_ID)
    gyro, accel = channels
    frames, nominal_rate_hz = EXPECTED_STREAMS[aria.IMU_RIGHT_STREAM_ID]
    assert gyro.times_ns.size == frames, "no sample of R_01_easy is flagged invalid"
    assert np.array_equal(gyro.times_ns, accel.times_ns)
    assert gyro.values_xyz.shape == (frames, 3)
    measured_rate_hz: float = 1e9 * (frames - 1) / float(gyro.times_ns[-1] - gyro.times_ns[0])
    assert measured_rate_hz == pytest.approx(nominal_rate_hz, rel=0.01)
    # Units, not counts: the device is stationary for the first second of R_01_easy,
    # so raw accel reads one gravity and raw gyro reads nearly nothing.
    assert np.linalg.norm(accel.values_xyz[:100], axis=1) == pytest.approx(9.7, abs=0.5)
    assert np.abs(gyro.values_xyz[:100]).max() < 0.1


def test_imu_left_runs_at_its_own_rate(provider: VrsDataProvider) -> None:
    gyro: ImuChannel = aria.read_imu(provider, aria.IMU_LEFT_STREAM_ID)[0]
    frames, nominal_rate_hz = EXPECTED_STREAMS[aria.IMU_LEFT_STREAM_ID]
    assert gyro.times_ns.size == frames
    measured_rate_hz: float = 1e9 * (frames - 1) / float(gyro.times_ns[-1] - gyro.times_ns[0])
    assert measured_rate_hz == pytest.approx(nominal_rate_hz, rel=0.01)


def test_the_pseudo_gt_is_stamped_on_the_slam_left_frame_clock(provider: VrsDataProvider) -> None:
    """A gt layer can therefore be logged on ``video_time`` with no shift at all."""
    pgt: aria.PseudoGt = aria.read_pseudo_gt(SEQUENCE_DIR / "ground_truth" / "pGT" / "R_01_easy.txt")
    assert np.array_equal(pgt.times_ns, aria.frame_timestamps_ns(provider, aria.SLAM_LEFT_STREAM_ID))
