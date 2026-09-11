"""The catalog-to-estimator mapping rules, on synthetic statics, plus one real smoke segment."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs import _core
from slam_rs.apis.replay import Config, _replay
from slam_rs.catalog_feed import (
    CHILD_FROM_PARENT,
    RIG_ENTITY,
    TIMELINE,
    CameraCalib,
    CameraStatics,
    CatalogSegment,
    Frameset,
    ImuStream,
    SegmentFeed,
    _rig_trajectory,
    _shared_codec,
    _static_int,
    _static_string,
    _static_values,
    _video_codec,
    camera_calib,
    imu_calib,
    open_segment,
)
from slam_rs.reference import SMOKE_SEGMENTS, ReferenceManifest, ReferenceSegment, resolved_flow_config
from slam_rs.tracking import Lockstep
from slam_rs.trajectory import Trajectory, associate, read_trajectory, shift_clock, write_trajectory
from slam_rs.vio_log import VioLogger, VioStage

SMOKE_SEGMENT: str = SMOKE_SEGMENTS[1]
"""The 7.6 s two-camera segment the smoke tier runs on."""


def _kb4_statics(
    distortion_model: str = "kannala_brandt",
    distortion_coefficients: Float64[ndarray, " n_slots"] | None = None,
    transform_relation: int = CHILD_FROM_PARENT,
    distortion_valid_radius: float | None = None,
) -> CameraStatics:
    """msd-index cam0's statics, exactly as the catalog stores them.

    Args:
        distortion_model: ``simplecv.components.DistortionModel`` string to store.
        distortion_coefficients: Fixed-width coefficient list; defaults to msd-index cam0's KB4 values with a zero tail.
        transform_relation: ``Transform3D:relation`` code to store.
        distortion_valid_radius: The valid radius ``rpmax``, when the recording carries one.

    Returns:
        Synthetic statics with the storage conventions the catalog really uses.
    """
    coefficients: Float64[ndarray, " n_slots"] = (
        np.array([0.192938, 0.042115, -0.233115, 0.095410, 0.0, 0.0, 0.0, 0.0]) if distortion_coefficients is None else distortion_coefficients
    )
    return CameraStatics(
        distortion_model=distortion_model,
        distortion_coefficients=coefficients,
        # Column-major: reading it row-major would swap (fx, fy) with (cx, cy).
        image_from_camera=np.array([420.5274, 0.0, 0.0, 0.0, 420.6685, 0.0, 469.4826, 479.1369, 1.0]),
        resolution_wh=np.array([960.0, 960.0]),
        transform_mat3x3=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        transform_translation=np.array([1.0, 2.0, 3.0]),
        transform_relation=transform_relation,
        distortion_valid_radius=distortion_valid_radius,
    )


def test_the_intrinsics_are_read_column_major() -> None:
    calib: CameraCalib = camera_calib(0, _kb4_statics())
    assert (calib.fx, calib.fy) == pytest.approx((420.5274, 420.6685))
    assert (calib.cx, calib.cy) == pytest.approx((469.4826, 479.1369))
    assert (calib.width, calib.height) == (960, 960)


def test_kb4_keeps_four_coefficients_and_rejects_a_live_tail() -> None:
    calib: CameraCalib = camera_calib(0, _kb4_statics())
    assert calib.model == "kb4"
    assert calib.distortion.shape == (4,)
    # Aria's Fisheye624 carries the same "kannala_brandt" string with eight live
    # coefficients, so a non-zero tail must fail loudly instead of truncating.
    aria_like: Float64[ndarray, " 8"] = np.array([-0.0248, 0.0963, -0.0633, 0.0062, 0.00349, -0.00073, -0.00036, 0.000895])
    with pytest.raises(ValueError, match="tail is non-zero"):
        camera_calib(0, _kb4_statics(distortion_coefficients=aria_like))


def test_radtan8_keeps_eight_coefficients() -> None:
    coefficients: Float64[ndarray, " 14"] = np.zeros(14, dtype=np.float64)
    coefficients[:8] = np.array([0.3022, -0.0215, 6e-05, 0.00025, 0.01582, 0.57506, -0.06264, 0.03385])
    statics: CameraStatics = _kb4_statics(distortion_model="brown_conrady", distortion_coefficients=coefficients, distortion_valid_radius=2.72764)
    calib: CameraCalib = camera_calib(0, statics)
    assert calib.model == "radtan8"
    np.testing.assert_allclose(calib.distortion, coefficients[:8])
    assert calib.distortion_valid_radius == pytest.approx(2.72764)


def test_an_unknown_distortion_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported distortion model"):
        camera_calib(0, _kb4_statics(distortion_model="double_sphere"))


def test_the_extrinsic_is_inverted_only_for_child_from_parent() -> None:
    """The stored transform is ``cam_T_imu``; the feed hands the estimator ``imu_T_cam``."""
    statics: CameraStatics = _kb4_statics()
    calib: CameraCalib = camera_calib(0, statics)
    cam_R_imu: Float64[ndarray, "3 3"] = statics.transform_mat3x3.reshape(3, 3, order="F")
    cam_T_imu: Float64[ndarray, "4 4"] = np.eye(4)
    cam_T_imu[:3, :3] = cam_R_imu
    cam_T_imu[:3, 3] = statics.transform_translation
    np.testing.assert_allclose(calib.imu_T_cam @ cam_T_imu, np.eye(4), atol=1e-12)
    with pytest.raises(ValueError, match="is not ChildFromParent"):
        camera_calib(0, _kb4_statics(transform_relation=0))


def _rotate_pinhole_clockwise(
    fx: float, fy: float, cx: float, cy: float, width: int, height: int, rotation_cw_deg: int
) -> tuple[float, float, float, float]:
    """Rotate landscape intrinsics into the stored image orientation.

    The catalog already stores rotated calibration, so this helper is test-only.
    It checks the relationship between landscape calibration and portrait images.

    Args:
        fx: Focal length along x before rotation.
        fy: Focal length along y before rotation.
        cx: Principal point x before rotation.
        cy: Principal point y before rotation.
        width: Image width before rotation.
        height: Image height before rotation.
        rotation_cw_deg: Clockwise rotation applied to the image, 0, 90, 180 or 270.

    Returns:
        ``(fx, fy, cx, cy)`` in the rotated frame.

    Raises:
        ValueError: If the rotation is not a multiple of 90 degrees.
    """
    if rotation_cw_deg == 0:
        return fx, fy, cx, cy
    if rotation_cw_deg == 90:
        return fy, fx, (height - 1) - cy, cx
    if rotation_cw_deg == 180:
        return fx, fy, (width - 1) - cx, (height - 1) - cy
    if rotation_cw_deg == 270:
        return fy, fx, cy, (width - 1) - cx
    raise ValueError(f"image rotation must be 0, 90, 180 or 270 degrees clockwise; got {rotation_cw_deg}")


def test_the_msd_g2_rotation_arithmetic() -> None:
    """The landscape msd-g2 calibration, rotated, is the catalog's portrait calibration."""
    landscape_width: int = 640
    landscape_height: int = 480
    # cam0 and cam1 are stored 90 degrees clockwise, cam2 and cam3 270.
    cam0: tuple[float, float, float, float] = _rotate_pinhole_clockwise(269.6842, 269.7883, 322.5579, 228.8732, landscape_width, landscape_height, 90)
    assert cam0 == pytest.approx((269.7883, 269.6842, 250.1268, 322.5579), abs=1e-4)
    # The rule itself: cx' = (H-1) - cy and cy' = cx at 90 clockwise.
    assert cam0[2] == pytest.approx(landscape_height - 1 - 228.8732, abs=1e-9)
    assert cam0[3] == pytest.approx(322.5579, abs=1e-9)

    rotated_270: tuple[float, float, float, float] = _rotate_pinhole_clockwise(100.0, 200.0, 300.0, 150.0, landscape_width, landscape_height, 270)
    assert rotated_270 == pytest.approx((200.0, 100.0, 150.0, landscape_width - 1 - 300.0))

    # Two turns of 90 clockwise are one turn of 180, in the rotated frame's size.
    once: tuple[float, float, float, float] = _rotate_pinhole_clockwise(100.0, 200.0, 300.0, 150.0, landscape_width, landscape_height, 90)
    twice: tuple[float, float, float, float] = _rotate_pinhole_clockwise(*once, landscape_height, landscape_width, 90)
    assert twice == pytest.approx(_rotate_pinhole_clockwise(100.0, 200.0, 300.0, 150.0, landscape_width, landscape_height, 180))

    assert _rotate_pinhole_clockwise(1.0, 2.0, 3.0, 4.0, 8, 6, 0) == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError, match="must be 0, 90, 180 or 270"):
        _rotate_pinhole_clockwise(1.0, 2.0, 3.0, 4.0, 8, 6, 45)


def test_the_imu_calibration_carries_the_manifests_frozen_numbers(manifest: ReferenceManifest) -> None:
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    calib = imu_calib(manifest.dataset(segment.dataset_name).imu, np.eye(4))
    assert calib.frequency_hz == 1000.0
    assert calib.gyro_noise_std == 0.000282
    assert calib.accel_noise_std == 0.016
    assert calib.cam_time_offset_ns == 0
    np.testing.assert_array_equal(calib.imu_T_body, np.eye(4))


def test_a_static_table_with_no_rows_names_the_entity() -> None:
    """A recording whose rig node carries no statics reached `statics[column][0]`.

    Row zero of an empty table is an `IndexError` out of pyarrow with nothing in
    it that says which entity was being read.
    """
    empty: pa.Table = pa.table({"/rig:num_cameras": pa.array([], type=pa.list_(pa.float64()))})
    with pytest.raises(ValueError, match="/rig:num_cameras has no rows"):
        _static_values(empty, "/rig:num_cameras")
    with pytest.raises(ValueError, match="/rig:reference has no rows"):
        _static_string(pa.table({"/rig:reference": pa.array([], type=pa.string())}), "/rig:reference")
    with pytest.raises(ValueError, match="property:capture:start_time_ns has no rows"):
        _static_int(pa.table({"property:capture:start_time_ns": pa.array([], type=pa.int64())}), "property:capture:start_time_ns")


def test_a_video_stream_with_no_codec_names_the_entity() -> None:
    """The codec is read from row zero of this camera's non-null codecs; both can be empty."""
    entity: str = "/rig/cam_00/pinhole/video"
    columns: list[str] = ["video_time", f"{entity}:VideoStream:is_keyframe", f"{entity}:VideoStream:codec"]
    no_rows: pa.Table = pa.table({name: pa.array([], type=pa.int64()) for name in columns})
    with pytest.raises(ValueError, match=f"{entity}: the recording carries no video samples"):
        _video_codec(no_rows, entity)

    no_codec: pa.Table = pa.table(
        {
            columns[0]: pa.array([1, 2], type=pa.int64()),
            columns[1]: pa.array([None, None], type=pa.int64()),
            columns[2]: pa.array([None, None], type=pa.list_(pa.uint32())),
        }
    )
    with pytest.raises(ValueError, match=f"{entity}: the video stream carries no codec"):
        _video_codec(no_codec, entity)


def test_a_rig_whose_cameras_disagree_on_the_codec_names_both() -> None:
    """One index carries one codec, so a mixed rig must say which two cameras differ.

    Every current rig is uniform (MSD is AV1, RoboCap H.264), so the muxer would
    otherwise write the last camera's codec over every camera's samples.
    """
    assert _shared_codec([(0, "av1"), (2, "av1")], SMOKE_SEGMENT) == "av1"
    with pytest.raises(ValueError, match="cam_02 is h264 where cam_00 is av1"):
        _shared_codec([(0, "av1"), (2, "h264")], SMOKE_SEGMENT)


def _transform_rows(t_ns: list[int], translations: list[list[float] | None], quaternions: list[list[float] | None]) -> pa.Table:
    """One window of the ``gt`` layer, with each component present on the rows the caller names.

    Rerun nests a component's instances one list deep, so a row carrying one
    translation is ``[[x, y, z]]`` and a row carrying none is null — which is what
    component-level logging or a clear produces, and the two components need not
    be null on the same rows.

    Args:
        t_ns: ``video_time`` of each row.
        translations: Per row, the rig position in metres, or None where the row carries no translation.
        quaternions: Per row, the XYZW rotation, or None where the row carries no quaternion.

    Returns:
        The window in the shape the reader hands :func:`_rig_trajectory`.
    """
    return pa.table(
        {
            TIMELINE: pa.array(t_ns, type=pa.int64()),
            f"{RIG_ENTITY}:Transform3D:translation": pa.array(
                [None if value is None else [value] for value in translations], type=pa.list_(pa.list_(pa.float64(), 3))
            ),
            f"{RIG_ENTITY}:Transform3D:quaternion": pa.array(
                [None if value is None else [value] for value in quaternions], type=pa.list_(pa.list_(pa.float64(), 4))
            ),
        }
    )


def test_a_ground_truth_window_reads_both_components_off_the_same_rows() -> None:
    """A pose is a translation and a rotation from one row, and the reader used to pair them by position.

    The timestamps came from the translation column's validity mask while the
    quaternion column was flattened with its own null removal, so two components
    valid on *different* rows still flattened to the same length and the pose at
    ``t0`` was handed ``t1``'s rotation: a plausible layout — component-level
    logging or a clear writes it, and the public ``--gt-rrd`` path reads it —
    silently attached a rotation from the wrong time to every pose (S25 review).
    """
    aligned: Trajectory = _rig_trajectory(
        _transform_rows(
            [10, 20, 30],
            [[1.0, 0.0, 0.0], None, [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0], None, [0.0, 0.0, 1.0, 0.0]],
        ),
        SMOKE_SEGMENT,
    )
    np.testing.assert_array_equal(aligned.t_ns, np.array([10, 30], dtype=np.int64))
    np.testing.assert_array_equal(aligned.position_m[:, 0], np.array([1.0, 3.0]))
    # w-first in memory, from Rerun's XYZW on the wire.
    np.testing.assert_array_equal(aligned.quaternion_wxyz[:, 0], np.array([1.0, 0.0]))

    # Two masks, two valid rows each, one row in common: the counts match, so
    # nothing downstream could notice the swap.
    with pytest.raises(ValueError, match=f"{SMOKE_SEGMENT}: 2 of 3 rig rows.*translation.*quaternion.*first at 10 ns"):
        _rig_trajectory(
            _transform_rows(
                [10, 20, 30],
                [[1.0, 0.0, 0.0], None, [3.0, 0.0, 0.0]],
                [None, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
            ),
            SMOKE_SEGMENT,
        )


def test_a_ground_truth_window_with_no_pose_in_it_is_empty() -> None:
    """A window the layer does not cover is the same answer as a layer that is not there."""
    assert len(_rig_trajectory(_transform_rows([10, 20], [None, None], [None, None]), SMOKE_SEGMENT)) == 0
    assert len(_rig_trajectory(pa.table({TIMELINE: pa.array([10], type=pa.int64())}), SMOKE_SEGMENT)) == 0


@pytest.mark.slow
def test_the_smoke_segment_decodes_from_the_catalog(manifest: ReferenceManifest) -> None:
    """One real segment end to end: frame count, shape, dtype and paired IMU timestamps."""
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)

    with open_segment(CatalogSegment(manifest.catalog_url, segment.dataset_name, segment.segment_id), manifest.dataset(segment.dataset_name).imu) as feed:
        assert isinstance(feed, SegmentFeed)
        assert len(feed.cameras) == 2
        assert len(feed.frame_t_ns) > 0
        assert feed.capture_start_time_ns > 0
        # MSD's video_time is relative to that, so an export has to add it back.
        assert feed.export_offset_ns == feed.capture_start_time_ns
        for camera in feed.cameras:
            assert (camera.width, camera.height) == (960, 960)
            assert camera.model == "kb4"
        # Gyroscope and accelerometer share timestamps on MSD; the feed refuses to
        # build an ImuStream otherwise, so reaching here is the assertion.
        whole_imu: ImuStream = feed.imu_between(-(2**62), 2**62)
        assert len(whole_imu) > 7_000
        assert whole_imu.t_ns.dtype == np.int64

        # A count of framesets is a time to the feed, or it is told to run to
        # the end of the segment and fetches a window nothing decodes. The
        # hundredth frameset with no stride, and the segment's last when the
        # count runs past its end.
        assert feed.stop_ns_after(None) is None
        assert feed.stop_ns_after(100) == int(feed.frame_t_ns[99])
        assert feed.stop_ns_after(len(feed.frame_t_ns) + 1) == int(feed.frame_t_ns[-1])
        assert feed.stop_ns_after(1) == int(feed.frame_t_ns[0])
        assert whole_imu.gyro_rad_s.shape == (len(whole_imu), 3)
        assert whole_imu.accel_m_s2.shape == (len(whole_imu), 3)
        whole_gt: Trajectory = feed.ground_truth_between(-(2**62), 2**62)
        assert len(whole_gt)

        digests: list[str] = []
        with_truth: int = 0
        frameset: Frameset
        for frameset in feed.framesets():
            assert len(frameset.images) == 2
            for image in frameset.images:
                assert image.shape == (960, 960)
                assert image.dtype == np.uint8
                assert image.flags["C_CONTIGUOUS"]
            if frameset.ground_truth is not None:
                assert frameset.ground_truth.shape == (7,)
                with_truth += 1
            digests.append(frameset.digest())
        assert len(digests) == len(feed.frame_t_ns)
        # Ground truth starts 17.5 ms into the segment, so only the first frameset
        # is without it; a frameset outside the truth's span reports None rather
        # than borrowing a stale pose.
        assert with_truth == len(feed.frame_t_ns) - 1


@pytest.mark.slow
def test_the_window_size_does_not_change_a_single_pixel_or_an_imu_sample(manifest: ReferenceManifest) -> None:
    """Cutting the segment into 2 s windows must reproduce the pixels and the inertial stream exactly."""
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)
    digests: dict[float, list[tuple[int, str]]] = {}
    imu_t_ns: dict[float, Int64[ndarray, " n_samples"]] = {}
    for window_s in (60.0, 2.0):
        with open_segment(CatalogSegment(manifest.catalog_url, segment.dataset_name, segment.segment_id), manifest.dataset(segment.dataset_name).imu, window_s=window_s) as feed:
            per_frameset: list[tuple[int, str]] = []
            emitted: list[Int64[ndarray, " n"]] = []
            for frameset in feed.framesets():
                per_frameset.append((frameset.t_ns, frameset.digest()))
                emitted.append(frameset.imu.t_ns)
            digests[window_s] = per_frameset
            imu_t_ns[window_s] = np.concatenate(emitted)
    assert digests[60.0] == digests[2.0]
    assert len(digests[60.0]) == len(feed.frame_t_ns)

    # A bounded read is the same read, stopped: the same rows in the same order,
    # and no window opening past the bound is fetched at all — with 2 s windows
    # over a 7.6 s segment, a bound at 2.5 s leaves the last two windows unread.
    first_ns: int = digests[2.0][0][0]
    bounded_ns: int = first_ns + 2_500_000_000
    with open_segment(CatalogSegment(manifest.catalog_url, segment.dataset_name, segment.segment_id), manifest.dataset(segment.dataset_name).imu, window_s=2.0) as feed:
        bounded: list[tuple[int, str]] = [(frameset.t_ns, frameset.digest()) for frameset in feed.framesets(bounded_ns)]
    assert bounded == digests[2.0][: len(bounded)]
    assert bounded_ns <= bounded[-1][0] < bounded_ns + 2_000_000_000
    assert len(bounded) < len(digests[2.0])

    # The inertial samples handed out across window boundaries must form one
    # stream: strictly increasing, no sample delivered twice, and the same set
    # whatever the window size. A window that failed to reach back would drop
    # samples here instead of silently under-integrating.
    for window_s, times in imu_t_ns.items():
        assert bool(np.all(np.diff(times) > 0)), f"window {window_s}s emitted non-monotonic IMU timestamps"
    np.testing.assert_array_equal(imu_t_ns[60.0], imu_t_ns[2.0])




@pytest.mark.slow
def test_a_replay_export_associates_with_the_catalog_ground_truth(manifest: ReferenceManifest, tmp_path: Path) -> None:
    """The replay's own export path, end to end, lands on the sidecar's clock.

    Forty framesets of the smoke segment through the whole pipeline: enough for
    the estimator to initialise and produce poses, which is what gets exported.
    Written with ``video_time`` the file associates with **nothing**, which is
    the regression being pinned.
    """
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENT)

    config: Config = Config(rr_config=RerunTyroConfig(headless=True), segment=SMOKE_SEGMENT, stage="vio", max_framesets=40)
    with open_segment(CatalogSegment(manifest.catalog_url, segment.dataset_name, segment.segment_id), manifest.dataset(segment.dataset_name).imu) as feed:
        truth: Trajectory = feed.ground_truth_between(int(feed.frame_t_ns[0]), int(feed.frame_t_ns[-1]))
        assert len(truth)
        stage: VioStage = VioStage(
            lockstep=Lockstep(vio=_core.Vio(_core.Calibration.from_catalog(feed.cameras, feed.imu), resolved_flow_config(manifest, segment)[0])),
            logger=VioLogger(
                cameras=feed.cameras,
                ground_truth=truth,
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
    sidecar: Trajectory = shift_clock(truth, feed.capture_start_time_ns)
    # All but the first pose, which sits on the capture's start time — 17 ms
    # before the sidecar's first row, so it has nothing to associate with. The
    # ground truth does not cover the whole segment.
    assert associate(read_trajectory(exported), sidecar).count == len(estimate) - 1
    assert associate(read_trajectory(relative_export), sidecar).count == 0


@pytest.mark.parametrize("require_truth", [False, True])
def test_catalog_resolution_batches_segments_by_dataset(monkeypatch: pytest.MonkeyPatch, require_truth: bool) -> None:
    from unittest.mock import MagicMock

    from slam_rs import catalog_feed
    from slam_rs.catalog_feed import resolve_catalog_segments

    client: MagicMock = MagicMock()
    client.get_dataset.return_value = MagicMock(spec=DatasetEntry)
    dataset: MagicMock = client.get_dataset.return_value
    dataset.manifest.return_value.to_arrow_table.return_value = pa.table(
        {"rerun_segment_id": ["first", "first", "second", "second"], "rerun_layer_name": ["base", "gt", "base", "gt"]}
    )
    monkeypatch.setattr(catalog_feed, "CatalogClient", lambda _url: client)
    resolved: tuple[CatalogSegment, ...] = resolve_catalog_segments(
        (CatalogSegment("test", "dataset", "first"), CatalogSegment("test", "dataset", "second")), require_ground_truth=require_truth
    )
    assert [source.segment_id for source in resolved] == ["first", "second"]
    assert all(source.dataset is dataset and source.has_ground_truth for source in resolved)
    client.get_dataset.assert_called_once_with("dataset")
    dataset.manifest.assert_called_once_with()


@pytest.mark.parametrize(("segment", "require_truth", "error"), [("missing", False, "absent from catalog"), ("present", True, "ground-truth layer absent")])
def test_catalog_resolution_refuses_missing_inputs(monkeypatch: pytest.MonkeyPatch, segment: str, require_truth: bool, error: str) -> None:
    from unittest.mock import MagicMock

    from slam_rs import catalog_feed
    from slam_rs.catalog_feed import resolve_catalog_segments

    client: MagicMock = MagicMock()
    client.get_dataset.return_value = MagicMock(spec=DatasetEntry)
    client.get_dataset.return_value.manifest.return_value.to_arrow_table.return_value = pa.table(
        {"rerun_segment_id": ["present"], "rerun_layer_name": ["base"]}
    )
    monkeypatch.setattr(catalog_feed, "CatalogClient", lambda _url: client)
    with pytest.raises(ValueError, match=error):
        resolve_catalog_segments((CatalogSegment("test", "dataset", segment),), require_ground_truth=require_truth)
    available: CatalogSegment = resolve_catalog_segments((CatalogSegment("test", "dataset", "present"),))[0]
    assert available.dataset is not None
    assert not available.has_ground_truth
