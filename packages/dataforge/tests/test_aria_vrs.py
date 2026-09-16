"""The Aria VRS readers against a real LaMAria sequence.

Skipped unless R_01_easy's 897 MB VRS has been fetched into the package's
gitignored ``data/`` tree:

    packages/dataforge/data/raw/lamaria/training/R_01_easy/raw_data/R_01_easy.vrs

The load-bearing assertion is the transform chain: ``rig_T_cam`` derived from
the VRS device calibration must reproduce the published ``T_b_s``. Those two
paths share no code — one is ``imuR_T_device @ device_T_cam`` out of the
factory calibration, the other is a quaternion in a JSON file — so agreement
pins both the chain and the quaternion order. That cross-check reads the
**unrotated** rig, because the published file is unrotated; the upright rig
converters log is checked against it here too.

``lamaria.open_streams`` is exercised here as well: it is the only place where
the rotated pixels and the rotated calibration meet a real frame.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np
import pytest
from conftest import PublishedCamera, read_calibration_json  # pyrefly: ignore[missing-import]
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from projectaria_tools.core.data_provider import VrsDataProvider
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Fisheye62Parameters

from dataforge import aria
from dataforge.datasets import lamaria
from dataforge.logging_toolkit import ImuChannel

PACKAGE_DIR: Path = Path(__file__).parents[1]
SEQUENCE_DIR: Path = PACKAGE_DIR / "data" / "raw" / "lamaria" / "training" / "R_01_easy"
VRS_PATH: Path = SEQUENCE_DIR / "raw_data" / "R_01_easy.vrs"
PUBLISHED_CALIBRATION_PATH: Path = SEQUENCE_DIR / "aria_calibrations" / "R_01_easy.json"

pytestmark = pytest.mark.skipif(not VRS_PATH.is_file(), reason=f"no development VRS at {VRS_PATH}")

CW90_ABOUT_OPTICAL_AXIS: Float64[ndarray, "3 3"] = Rotation.from_euler("z", -90.0, degrees=True).as_matrix()
"""The quarter turn about a camera's own z that rotating its pixels clockwise is.

Turning the image content clockwise turns the camera frame the other way, so
``rig_R_cam`` of the upright camera is the native one times this."""

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
    """The rig as the VRS publishes it, which is what the published JSON describes."""
    return aria.AriaRig.from_provider(provider, rotate_cw90=False)


@pytest.fixture(scope="module")
def upright_rig(provider: VrsDataProvider) -> aria.AriaRig:
    """The rig LaMAria converts log: every camera turned a quarter turn clockwise."""
    return aria.AriaRig.from_provider(provider, rotate_cw90=True)


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
    published: dict[str, PublishedCamera] = read_calibration_json(PUBLISHED_CALIBRATION_PATH)
    for name, stream_id in (("cam0", aria.SLAM_LEFT_STREAM_ID), ("cam1", aria.SLAM_RIGHT_STREAM_ID)):
        from_vrs: Float64[ndarray, "4 4"] = rig.cameras[stream_id].extrinsics.world_T_cam
        assert from_vrs == pytest.approx(published[name].rig_T_cam.to_matrix(), abs=1e-3), name


def test_published_intrinsics_match_the_vrs_ones(rig: aria.AriaRig) -> None:
    published: dict[str, PublishedCamera] = read_calibration_json(PUBLISHED_CALIBRATION_PATH)
    for name, stream_id in (("cam0", aria.SLAM_LEFT_STREAM_ID), ("cam1", aria.SLAM_RIGHT_STREAM_ID)):
        camera: Fisheye62Parameters = rig.cameras[stream_id]
        fx, fy, cx, cy = published[name].params[:4]
        assert (camera.intrinsics.fl_x, camera.intrinsics.fl_y) == pytest.approx((fx, fy), abs=1e-6), name
        assert (camera.intrinsics.cx, camera.intrinsics.cy) == pytest.approx((cx, cy), abs=1e-6), name
        assert (camera.intrinsics.width, camera.intrinsics.height) == (640, 480), name


def test_the_upright_cameras_swap_their_axes_and_move_their_principal_point(rig: aria.AriaRig, upright_rig: aria.AriaRig) -> None:
    """A clockwise quarter turn of the pixels: ``w`` and ``h`` swap, ``(cx, cy)`` becomes ``(h - 1 - cy, cx)``."""
    for stream_id in aria.CAMERA_STREAM_IDS:
        native: Fisheye62Parameters = rig.cameras[stream_id]
        upright: Fisheye62Parameters = upright_rig.cameras[stream_id]
        assert (upright.intrinsics.width, upright.intrinsics.height) == (native.intrinsics.height, native.intrinsics.width), stream_id
        # One focal for both axes, so turning the image cannot change it.
        assert (upright.intrinsics.fl_x, upright.intrinsics.fl_y) == (native.intrinsics.fl_x, native.intrinsics.fl_y), stream_id
        native_pp_px: Float64[ndarray, "2"] = np.array([native.intrinsics.cx, native.intrinsics.cy], dtype=np.float64)
        upright_pp_px: Float64[ndarray, "2"] = np.array([upright.intrinsics.cx, upright.intrinsics.cy], dtype=np.float64)
        turned_pp_px: Float64[ndarray, "2"] = aria.rotate_uv_cw90(native_pp_px.reshape(1, 2), native_height_px=native.intrinsics.height)[0]
        assert upright_pp_px == pytest.approx(turned_pp_px, abs=1e-9), stream_id
    assert (upright_rig.cameras[aria.SLAM_LEFT_STREAM_ID].intrinsics.width, upright_rig.cameras[aria.SLAM_LEFT_STREAM_ID].intrinsics.height) == (
        480,
        640,
    )
    # 322.895555771704, 240.4445599560633 → 479 - cy, cx.
    assert upright_rig.cameras[aria.SLAM_LEFT_STREAM_ID].intrinsics.cx == pytest.approx(238.5554400439367, abs=1e-9)
    assert upright_rig.cameras[aria.SLAM_LEFT_STREAM_ID].intrinsics.cy == pytest.approx(322.895555771704, abs=1e-9)
    # camera-rgb is square, so only its principal point moves.
    assert (upright_rig.cameras[aria.RGB_STREAM_ID].intrinsics.width, upright_rig.cameras[aria.RGB_STREAM_ID].intrinsics.height) == (1408, 1408)


def test_the_upright_rig_turns_each_camera_about_its_own_optical_axis(rig: aria.AriaRig, upright_rig: aria.AriaRig) -> None:
    """The calibration follows the pixels: the pose turns, the camera stays where it is."""
    for stream_id in aria.CAMERA_STREAM_IDS:
        native: Float64[ndarray, "4 4"] = rig.cameras[stream_id].extrinsics.world_T_cam
        upright: Float64[ndarray, "4 4"] = upright_rig.cameras[stream_id].extrinsics.world_T_cam
        assert upright[:3, :3] == pytest.approx(native[:3, :3] @ CW90_ABOUT_OPTICAL_AXIS, abs=1e-12), stream_id
        assert upright[:3, 3] == pytest.approx(native[:3, 3], abs=1e-12), stream_id
    # And the IMUs are not cameras: rotating the images leaves them alone.
    for stream_id in aria.IMU_STREAM_IDS:
        assert upright_rig.rig_T_imu[stream_id] == pytest.approx(rig.rig_T_imu[stream_id], abs=1e-15), stream_id


def test_the_published_rig_T_cam0_is_the_unrotated_one(rig: aria.AriaRig, upright_rig: aria.AriaRig) -> None:
    """What the gt layer composes the pGT with must be the native transform.

    The pGT poses camera-slam-left as the archive published it, so the layer that
    moves it onto the rig reads ``cam0.T_b_s`` — the native pose — and never the
    upright one the base layer's pixels ride.
    """
    published: Float64[ndarray, "4 4"] = aria.read_rig_T_cam0(PUBLISHED_CALIBRATION_PATH)
    assert published == pytest.approx(rig.cameras[aria.SLAM_LEFT_STREAM_ID].extrinsics.world_T_cam, abs=1e-3)
    upright: Float64[ndarray, "4 4"] = upright_rig.cameras[aria.SLAM_LEFT_STREAM_ID].extrinsics.world_T_cam
    assert np.abs(published[:3, :3] - upright[:3, :3]).max() > 0.5, "a quarter turn is not a rounding difference"


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
    assert "cam2" not in read_calibration_json(PUBLISHED_CALIBRATION_PATH)
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


def test_open_streams_hands_the_encoder_upright_contiguous_frames() -> None:
    """The converter's frame stream turns the pixels the way it turned the calibration.

    ``encode_frames_to_mp4`` writes the buffer it is given straight into ffmpeg's
    stdin, so a rotated frame has to arrive contiguous and at the rotated
    camera's own size, or the video comes out sheared.
    """
    streams: lamaria.SequenceStreams = lamaria.open_streams(VRS_PATH)
    left: lamaria.CameraStream = streams.cameras[0]
    assert (left.camera.intrinsics.width, left.camera.intrinsics.height) == (480, 640)

    plane: bytes | memoryview = next(left.frames)
    assert isinstance(plane, memoryview)
    assert plane.c_contiguous, "ffmpeg reads the buffer as it stands"
    assert plane.nbytes == 480 * 640

    native: aria.AriaImage = next(aria.iter_frames(aria.open_vrs(VRS_PATH), aria.SLAM_LEFT_STREAM_ID))[1]
    upright: UInt8[ndarray, "h w"] = np.frombuffer(plane, dtype=np.uint8).reshape(640, 480)
    np.testing.assert_array_equal(upright, np.rot90(native, -1))


def test_the_pseudo_gt_is_stamped_on_the_slam_left_frame_clock(provider: VrsDataProvider) -> None:
    """A gt layer can therefore be logged on ``video_time`` with no shift at all."""
    pgt: aria.PseudoGt = aria.read_pseudo_gt(SEQUENCE_DIR / "ground_truth" / "pGT" / "R_01_easy.txt")
    assert np.array_equal(pgt.times_ns, aria.frame_timestamps_ns(provider, aria.SLAM_LEFT_STREAM_ID))
