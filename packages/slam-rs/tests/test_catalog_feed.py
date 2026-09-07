"""The catalog-to-estimator mapping rules, on synthetic statics, plus one real smoke segment."""

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

from slam_rs.catalog_feed import (
    CameraCalib,
    CameraStatics,
    Frameset,
    LocalSegment,
    SegmentFeed,
    camera_calib,
    imu_calib,
    open_segment,
    rotate_pinhole_clockwise,
)
from slam_rs.reference import ReferenceManifest, ReferenceSegment, load_manifest

SMOKE_SEGMENT: str = "msd-index__MIO_others__MIO10_short_2_panorama"
"""The 7.6 s two-camera segment the smoke tier runs on."""


def _kb4_statics(
    distortion_model: str = "kannala_brandt",
    distortion_coefficients: Float64[ndarray, " n_slots"] | None = None,
    transform_relation: int = 2,
    distortion_valid_radius: float | None = None,
    image_rotation_cw_deg: int = 0,
) -> CameraStatics:
    """msd-index cam0's statics, exactly as the catalog stores them.

    Args:
        distortion_model: ``simplecv.components.DistortionModel`` string to store.
        distortion_coefficients: Fixed-width coefficient list; defaults to msd-index cam0's KB4 values with a zero tail.
        transform_relation: ``Transform3D:relation`` code to store.
        distortion_valid_radius: basalt's ``rpmax``, when the recording carries one.
        image_rotation_cw_deg: Clockwise rotation baked into the stored images.

    Returns:
        Synthetic statics with the storage conventions the catalog really uses.
    """
    coefficients: Float64[ndarray, " n_slots"] = (
        np.array([0.192938, 0.042115, -0.233115, 0.095410, 0.0, 0.0, 0.0, 0.0]) if distortion_coefficients is None else distortion_coefficients
    )
    return CameraStatics(
        camera_model="kb4",
        distortion_model=distortion_model,
        distortion_coefficients=coefficients,
        # Column-major: reading it row-major would swap (fx, fy) with (cx, cy).
        image_from_camera=np.array([420.5274, 0.0, 0.0, 0.0, 420.6685, 0.0, 469.4826, 479.1369, 1.0]),
        resolution_wh=np.array([960.0, 960.0]),
        transform_mat3x3=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        transform_translation=np.array([1.0, 2.0, 3.0]),
        transform_relation=transform_relation,
        distortion_valid_radius=distortion_valid_radius,
        image_rotation_cw_deg=image_rotation_cw_deg,
    )


def test_the_intrinsics_are_read_column_major() -> None:
    calib: CameraCalib = camera_calib(0, _kb4_statics(), frequency_hz=54.0)
    assert (calib.fx, calib.fy) == pytest.approx((420.5274, 420.6685))
    assert (calib.cx, calib.cy) == pytest.approx((469.4826, 479.1369))
    assert (calib.width, calib.height) == (960, 960)


def test_kb4_keeps_four_coefficients_and_rejects_a_live_tail() -> None:
    calib: CameraCalib = camera_calib(0, _kb4_statics(), frequency_hz=54.0)
    assert calib.model == "kb4"
    assert calib.distortion.shape == (4,)
    # Aria's Fisheye624 carries the same "kannala_brandt" string with eight live
    # coefficients, so a non-zero tail must fail loudly instead of truncating.
    aria_like: Float64[ndarray, " 8"] = np.array([-0.0248, 0.0963, -0.0633, 0.0062, 0.00349, -0.00073, -0.00036, 0.000895])
    with pytest.raises(ValueError, match="tail is non-zero"):
        camera_calib(0, _kb4_statics(distortion_coefficients=aria_like), frequency_hz=54.0)


def test_radtan8_keeps_eight_coefficients() -> None:
    coefficients: Float64[ndarray, " 14"] = np.zeros(14, dtype=np.float64)
    coefficients[:8] = np.array([0.3022, -0.0215, 6e-05, 0.00025, 0.01582, 0.57506, -0.06264, 0.03385])
    statics: CameraStatics = _kb4_statics(distortion_model="brown_conrady", distortion_coefficients=coefficients, distortion_valid_radius=2.72764)
    calib: CameraCalib = camera_calib(0, statics, frequency_hz=30.0)
    assert calib.model == "radtan8"
    np.testing.assert_allclose(calib.distortion, coefficients[:8])
    assert calib.distortion_valid_radius == pytest.approx(2.72764)


def test_an_unknown_distortion_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported distortion model"):
        camera_calib(0, _kb4_statics(distortion_model="double_sphere"), frequency_hz=54.0)


def test_the_extrinsic_is_inverted_only_for_child_from_parent() -> None:
    """The stored transform is ``cam_T_imu``; the feed hands the estimator ``imu_T_cam``."""
    statics: CameraStatics = _kb4_statics()
    calib: CameraCalib = camera_calib(0, statics, frequency_hz=54.0)
    cam_R_imu: Float64[ndarray, "3 3"] = statics.transform_mat3x3.reshape(3, 3, order="F")
    cam_T_imu: Float64[ndarray, "4 4"] = np.eye(4)
    cam_T_imu[:3, :3] = cam_R_imu
    cam_T_imu[:3, 3] = statics.transform_translation
    np.testing.assert_allclose(calib.imu_T_cam @ cam_T_imu, np.eye(4), atol=1e-12)
    with pytest.raises(ValueError, match="is not ChildFromParent"):
        camera_calib(0, _kb4_statics(transform_relation=0), frequency_hz=54.0)


def test_the_msd_g2_rotation_arithmetic() -> None:
    """basalt's landscape msd-g2 calibration, rotated, is the catalog's portrait calibration."""
    landscape_width: int = 640
    landscape_height: int = 480
    # cam0 and cam1 are stored 90 degrees clockwise, cam2 and cam3 270.
    cam0: tuple[float, float, float, float] = rotate_pinhole_clockwise(269.6842, 269.7883, 322.5579, 228.8732, landscape_width, landscape_height, 90)
    assert cam0 == pytest.approx((269.7883, 269.6842, 250.1268, 322.5579), abs=1e-4)
    # The rule itself: cx' = (H-1) - cy and cy' = cx at 90 clockwise.
    assert cam0[2] == pytest.approx(landscape_height - 1 - 228.8732, abs=1e-9)
    assert cam0[3] == pytest.approx(322.5579, abs=1e-9)

    rotated_270: tuple[float, float, float, float] = rotate_pinhole_clockwise(100.0, 200.0, 300.0, 150.0, landscape_width, landscape_height, 270)
    assert rotated_270 == pytest.approx((200.0, 100.0, 150.0, landscape_width - 1 - 300.0))

    # Two turns of 90 clockwise are one turn of 180, in the rotated frame's size.
    once: tuple[float, float, float, float] = rotate_pinhole_clockwise(100.0, 200.0, 300.0, 150.0, landscape_width, landscape_height, 90)
    twice: tuple[float, float, float, float] = rotate_pinhole_clockwise(*once, landscape_height, landscape_width, 90)
    assert twice == pytest.approx(rotate_pinhole_clockwise(100.0, 200.0, 300.0, 150.0, landscape_width, landscape_height, 180))

    assert rotate_pinhole_clockwise(1.0, 2.0, 3.0, 4.0, 8, 6, 0) == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError, match="must be 0, 90, 180 or 270"):
        rotate_pinhole_clockwise(1.0, 2.0, 3.0, 4.0, 8, 6, 45)


def test_the_imu_calibration_carries_the_manifests_frozen_numbers() -> None:
    manifest: ReferenceManifest = load_manifest()
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    calib = imu_calib(segment.imu, np.eye(4))
    assert calib.frequency_hz == 1000.0
    assert calib.gyro_noise_std == 0.000282
    assert calib.accel_noise_std == 0.016
    assert calib.cam_time_offset_ns == 0
    np.testing.assert_array_equal(calib.imu_T_body, np.eye(4))


@pytest.mark.slow
def test_the_smoke_segment_decodes_from_the_nas() -> None:
    """One real segment end to end: frame count, shape, dtype and paired IMU timestamps."""
    manifest: ReferenceManifest = load_manifest()
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    if not segment.base_path.is_file():
        pytest.skip(f"{segment.base_path} is not mounted on this host")
    gt_path = segment.gt_path if segment.gt_path.is_file() else None

    with open_segment(LocalSegment(base_rrd=segment.base_path, gt_rrd=gt_path), segment.imu) as feed:
        assert isinstance(feed, SegmentFeed)
        assert len(feed.cameras) == segment.capture.num_cameras
        assert len(feed.frame_t_ns) == segment.capture.num_frames
        for camera in feed.cameras:
            assert (camera.width, camera.height) == (960, 960)
            assert camera.model == "kb4"
        # Gyroscope and accelerometer share timestamps on MSD; the feed refuses to
        # build an ImuStream otherwise, so reaching here is the assertion.
        assert len(feed.imu_stream) > 7_000
        assert feed.imu_stream.t_ns.dtype == np.int64
        assert feed.imu_stream.gyro_rad_s.shape == (len(feed.imu_stream), 3)
        assert feed.imu_stream.accel_m_s2.shape == (len(feed.imu_stream), 3)
        assert feed.ground_truth is not None
        assert len(feed.ground_truth) == segment.gt.num_poses

        digests: list[str] = []
        frameset: Frameset
        for frameset in feed.framesets():
            assert len(frameset.images) == segment.capture.num_cameras
            for image in frameset.images:
                assert image.shape == (960, 960)
                assert image.dtype == np.uint8
                assert image.flags["C_CONTIGUOUS"]
            digests.append(frameset.sha256)
        assert len(digests) == segment.capture.num_frames


@pytest.mark.slow
def test_the_window_size_does_not_change_a_single_pixel() -> None:
    """Cutting the segment into 2 s windows must reproduce the one-window digests exactly."""
    manifest: ReferenceManifest = load_manifest()
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    if not segment.base_path.is_file():
        pytest.skip(f"{segment.base_path} is not mounted on this host")
    digests: dict[float, list[tuple[int, str]]] = {}
    for window_s in (60.0, 2.0):
        with open_segment(LocalSegment(base_rrd=segment.base_path), segment.imu, window_s=window_s) as feed:
            digests[window_s] = [(frameset.t_ns, frameset.sha256) for frameset in feed.framesets()]
    assert digests[60.0] == digests[2.0]
    assert len(digests[60.0]) == segment.capture.num_frames
