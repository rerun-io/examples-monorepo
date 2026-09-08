"""The catalog-to-estimator mapping rules, on synthetic statics, plus one real smoke segment."""

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import _core
from slam_rs.apis.replay import Config, VioStage, _cpp_trajectory, _replay
from slam_rs.catalog_feed import (
    CameraCalib,
    CameraStatics,
    Frameset,
    ImuStream,
    LocalSegment,
    SegmentFeed,
    camera_calib,
    imu_calib,
    open_segment,
    rotate_pinhole_clockwise,
)
from slam_rs.reference import ReferenceManifest, ReferenceSegment, flow_config, load_manifest
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import AteResult, Trajectory, associate, ate, read_trajectory, shift_clock, write_trajectory
from slam_rs.vio_log import VioLogger

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
        assert feed.capture_start_time_ns == segment.capture.start_time_ns
        for camera in feed.cameras:
            assert (camera.width, camera.height) == (960, 960)
            assert camera.model == "kb4"
        # Gyroscope and accelerometer share timestamps on MSD; the feed refuses to
        # build an ImuStream otherwise, so reaching here is the assertion.
        whole_imu: ImuStream = feed.imu_between(-(2**62), 2**62)
        assert len(whole_imu) > 7_000
        assert whole_imu.t_ns.dtype == np.int64
        assert whole_imu.gyro_rad_s.shape == (len(whole_imu), 3)
        assert whole_imu.accel_m_s2.shape == (len(whole_imu), 3)
        whole_gt: Trajectory | None = feed.ground_truth_between(-(2**62), 2**62)
        assert whole_gt is not None
        assert len(whole_gt) == segment.gt.num_poses

        digests: list[str] = []
        with_truth: int = 0
        frameset: Frameset
        for frameset in feed.framesets():
            assert len(frameset.images) == segment.capture.num_cameras
            for image in frameset.images:
                assert image.shape == (960, 960)
                assert image.dtype == np.uint8
                assert image.flags["C_CONTIGUOUS"]
            if frameset.ground_truth is not None:
                assert frameset.ground_truth.shape == (7,)
                with_truth += 1
            digests.append(frameset.sha256)
        assert len(digests) == segment.capture.num_frames
        # Ground truth starts 17.5 ms into the segment, so only the first frameset
        # is without it; a frameset outside the truth's span reports None rather
        # than borrowing a stale pose.
        assert with_truth == segment.capture.num_frames - 1


@pytest.mark.slow
def test_the_window_size_does_not_change_a_single_pixel_or_an_imu_sample() -> None:
    """Cutting the segment into 2 s windows must reproduce the pixels and the inertial stream exactly."""
    manifest: ReferenceManifest = load_manifest()
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    if not segment.base_path.is_file():
        pytest.skip(f"{segment.base_path} is not mounted on this host")
    digests: dict[float, list[tuple[int, str]]] = {}
    imu_t_ns: dict[float, Int64[ndarray, " n_samples"]] = {}
    for window_s in (60.0, 2.0):
        with open_segment(LocalSegment(base_rrd=segment.base_path), segment.imu, window_s=window_s) as feed:
            per_frameset: list[tuple[int, str]] = []
            emitted: list[Int64[ndarray, " n"]] = []
            for frameset in feed.framesets():
                per_frameset.append((frameset.t_ns, frameset.sha256))
                emitted.append(frameset.imu.t_ns)
            digests[window_s] = per_frameset
            imu_t_ns[window_s] = np.concatenate(emitted)
    assert digests[60.0] == digests[2.0]
    assert len(digests[60.0]) == segment.capture.num_frames

    # The inertial samples handed out across window boundaries must form one
    # stream: strictly increasing, no sample delivered twice, and the same set
    # whatever the window size. A window that failed to reach back would drop
    # samples here instead of silently under-integrating.
    for window_s, times in imu_t_ns.items():
        assert bool(np.all(np.diff(times) > 0)), f"window {window_s}s emitted non-monotonic IMU timestamps"
    np.testing.assert_array_equal(imu_t_ns[60.0], imu_t_ns[2.0])


@pytest.mark.slow
def test_the_absolute_clock_matches_the_ground_truth_sidecar() -> None:
    """The feed's ground truth, shifted by the capture start time, is the ``gt.csv`` sidecar."""
    manifest: ReferenceManifest = load_manifest()
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    if not segment.base_path.is_file() or not segment.gt_csv.is_file():
        pytest.skip(f"{segment.base_path} or {segment.gt_csv} is not mounted on this host")
    sidecar: Trajectory = read_trajectory(segment.gt_csv)
    assert len(sidecar) == segment.gt.num_poses

    with open_segment(LocalSegment(base_rrd=segment.base_path, gt_rrd=segment.gt_path), segment.imu) as feed:
        relative: Trajectory | None = feed.ground_truth_between(-(2**62), 2**62)
        assert relative is not None
        absolute: Trajectory = shift_clock(relative, feed.capture_start_time_ns)

    # Exact, not approximate: video_time + capture.start_time_ns is the sidecar's
    # clock by construction, so every one of the 6,998 timestamps must land.
    np.testing.assert_array_equal(absolute.t_ns, sidecar.t_ns)
    assert associate(sidecar, absolute).count == len(sidecar)
    # Without the shift the two share no instant at all.
    assert associate(sidecar, relative).count == 0
    # The rrd stores float32 positions; the sidecar keeps more digits.
    result: AteResult = ate(absolute, sidecar)
    assert result.n_associated == len(absolute)
    assert result.rmse_m < 1e-5


@pytest.mark.slow
def test_a_replay_export_associates_with_the_ground_truth_sidecar(tmp_path: Path) -> None:
    """The replay's own export path, end to end, lands on the sidecar's clock.

    Forty framesets of the smoke segment through the whole pipeline: enough for
    the estimator to initialise and produce poses, which is what gets exported.
    Written with ``video_time`` the file associates with **nothing**, which is
    the regression being pinned.
    """
    manifest: ReferenceManifest = load_manifest()
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    if not segment.base_path.is_file() or not segment.gt_csv.is_file():
        pytest.skip(f"{segment.base_path} or {segment.gt_csv} is not mounted on this host")

    config: Config = Config(rr_config=RerunTyroConfig(headless=True), segment=SMOKE_SEGMENT, stage="vio", max_framesets=40)
    with open_segment(LocalSegment(base_rrd=segment.base_path, gt_rrd=segment.gt_path), segment.imu) as feed:
        truth: Trajectory | None = feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1]))
        assert truth is not None
        stage: VioStage = VioStage(
            lockstep=Lockstep(vio=_core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), flow_config(manifest, segment))),
            logger=VioLogger(
                cameras=feed.cameras,
                ground_truth=truth,
                cpp=_cpp_trajectory(manifest, segment, feed.capture_start_time_ns),
                frame_t_ns=feed.frame_t_ns,
            ),
        )
        replayed: int = _replay(feed, config, stage)
        estimate: Trajectory = stage.logger.estimated()
        exported: Path = tmp_path / "slam_rs.csv"
        write_trajectory(exported, shift_clock(estimate, feed.capture_start_time_ns))
        relative_export: Path = tmp_path / "relative.csv"
        write_trajectory(relative_export, estimate)

    assert replayed == 40
    assert stage.lockstep.imu_samples > 0
    # One pose per frameset: every frameset's own batch runs past its frame time,
    # and one that did not would be held and tracked again rather than lost (D17).
    assert len(estimate) == replayed
    assert not stage.pending
    sidecar: Trajectory = read_trajectory(segment.gt_csv)
    # All but the first pose, which sits on the capture's start time — 17 ms
    # before the sidecar's first row, so it has nothing to associate with. The
    # ground truth does not cover the whole segment (D36).
    assert associate(read_trajectory(exported), sidecar).count == len(estimate) - 1
    assert associate(read_trajectory(relative_export), sidecar).count == 0
