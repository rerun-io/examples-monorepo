"""Camera-aligned catalog samples keep their relative timing at the estimator boundary."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr

pytest.importorskip("rerun.catalog", reason="requires the rerun.catalog dependency in this environment")

from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer
from simplecv.imu_calibration import ImuCalibration

from slam_rs.catalog_feed import CatalogSegment, ImuStream, LocalSegment, open_segment
from slam_rs.trajectory import Trajectory

pytestmark = pytest.mark.integration

@pytest.fixture
def offset_ns() -> int:
    return 0


@pytest.fixture
def clock_recording(tmp_path: Path, offset_ns: int) -> Path:
    """One camera and nearby IMU samples; encoded pixels are unused by timing tests."""
    path: Path = tmp_path / "clock.rrd"
    with rr.RecordingStream("clock-test", recording_id="clock") as rec:
        rec.save(path)
        rec.send_property("capture", rr.AnyValues(start_time_ns=0))
        rec.log("/world/rig_00", rr.AnyValues(reference="imu_00", num_cameras=1), static=True)
        rec.log("/world/rig_00/imu_00", ImuCalibration(gyro_noise_density=0.0007, accel_noise_density=0.006, gyro_bias_random_walk=0.00003, accel_bias_random_walk=0.0002, rate_hz=200.0, source="synthetic"), rr.AnyValues(applied_time_shift_ns=-offset_ns), static=True)
        rec.log("/world/rig_00/imu_00", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)), static=True)
        rec.log(
            "/world/rig_00/cam_00",
            rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3), relation=rr.TransformRelation.ChildFromParent),
            # Factory calibration is provenance. Applying it again would move
            # the camera away from the already-aligned IMU sample below.
            rr.AnyValues(camera_imu_time_offset_ns=99_000_000, time_offset_reference="/world/rig_00/imu_00"),
            static=True,
        )
        rec.log(
            "/world/rig_00/cam_00/pinhole",
            rr.Pinhole(focal_length=100, resolution=[200, 200]),
            rr.AnyValues(**{"simplecv.components.DistortionModel": "kannala_brandt", "simplecv.components.DistortionCoefficients": [0.0] * 4}),
            static=True,
        )
        for stamp in (990_000_000, 1_000_000_000, 1_010_000_000):
            rec.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
            rec.log("/world/rig_00/imu_00/gyro", rr.Scalars([1.0, 2.0, 3.0]))
            rec.log("/world/rig_00/imu_00/accel", rr.Scalars([4.0, 5.0, 6.0]))
        for stamp in (1_000_000_000, 1_033_333_333):
            rec.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
            rec.log("/world/rig_00/cam_00/pinhole/video", rr.VideoStream(sample=b"unused", codec=rr.VideoCodec.H264, is_keyframe=True))
    return path


@pytest.mark.parametrize("offset_ns", [0, 14_902_432])
def test_camera_and_imu_share_one_clock_and_window(clock_recording: Path, offset_ns: int) -> None:
    """Select the sample at the frame's estimator time, excluding both neighbours."""
    with open_segment(LocalSegment(clock_recording)) as feed:
        expected_ns: int = 1_000_000_000 if offset_ns == 0 else 1_014_902_432
        assert feed.imu.gyro_noise_std == 0.0007
        assert feed.imu.frequency_hz == 200.0
        assert feed.frame_t_ns[0] == expected_ns
        samples: ImuStream = feed.imu_between(expected_ns - 1_000_000, expected_ns + 1_000_000)
        np.testing.assert_array_equal(samples.t_ns, [expected_ns])
        np.testing.assert_array_equal(samples.gyro_rad_s, [[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(samples.accel_m_s2, [[4.0, 5.0, 6.0]])


def test_publishing_and_replacing_estimates_cannot_change_ground_truth(clock_recording: Path, tmp_path: Path) -> None:
    with rr.server.Server(datasets={"test": [clock_recording]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("test")
        for layer, stamps, positions in [("gt", [10, 30], [1.0, 3.0]), ("slam_rs", [10, 20], [99.0, 88.0]), ("slam_rs", [10, 20], [77.0, 66.0])]:
            path: Path = tmp_path / f"{layer}-{positions[0]}.rrd"
            with rr.RecordingStream("clock-test", recording_id="clock") as rec:
                rec.save(path)
                for stamp, position in zip(stamps, positions, strict=True):
                    rec.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
                    rec.log("/world/rig_00", rr.Transform3D(translation=[position, 0.0, 0.0], quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0])))
            dataset.register([path.as_uri()], layer_name=layer, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
            with open_segment(CatalogSegment(server.url(), "test", "clock")) as feed:
                truth: Trajectory = feed.ground_truth_between(0, 100)
                np.testing.assert_array_equal(truth.t_ns, [10, 30])
                np.testing.assert_array_equal(truth.position_m[:, 0], [1.0, 3.0])


def test_a_published_result_leaves_the_merged_rig_to_ground_truth(clock_recording: Path, tmp_path: Path) -> None:
    """With base, gt and a slam_rs layer merged, ``/world/rig_00`` holds the ground truth alone and the run sits under its own path."""
    from slam_rs.catalog_layer import POSE_SOURCE, write_layer
    from slam_rs.tracking import SegmentRun
    from slam_rs.trajectory import empty_trajectory

    truth_path: Path = tmp_path / "gt.rrd"
    with rr.RecordingStream("clock-test", recording_id="clock") as rec:
        rec.save(truth_path)
        for stamp, position in ((10, 1.0), (30, 3.0)):
            rec.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
            rec.log("/world/rig_00", rr.Transform3D(translation=[position, 0.0, 0.0], quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0])))
    estimate: Trajectory = Trajectory(
        t_ns=np.array([10, 20], dtype=np.int64), position_m=np.array([[99.0, 0.0, 0.0], [88.0, 0.0, 0.0]]), quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
    )
    result_path: Path = tmp_path / "slam_rs.rrd"
    write_layer(result_path, "clock", SegmentRun(estimate=estimate, framesets=2, lost=0, wall_s=0.1, ground_truth=empty_trajectory(),
                                                 median_tracker_ms=1.0, config_sha256="test", calibration_sha256="test-calibration"), clock_offset_ns=0, profile="fast", backend="cpu")
    with rr.server.Server(datasets={"test": [clock_recording]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("test")
        dataset.register([truth_path.as_uri()], layer_name="gt").wait()
        dataset.register([result_path.as_uri()], layer_name=POSE_SOURCE).wait()
        merged: pa.Table = (
            dataset.filter_segments(["clock"]).filter_contents(["/world/rig_00", "/world/runs/slam_rs/rig"]).reader(index="video_time").to_arrow_table()
        )
    stamps: list[int] = merged["video_time"].combine_chunks().cast(pa.int64()).to_pylist()
    # One translation per row, as a one-element list of xyz; None where the entity has no row at that time.
    rig: list[float | None] = [None if row is None else row[0][0] for row in merged["/world/rig_00:Transform3D:translation"].to_pylist()]
    run: list[float | None] = [None if row is None else row[0][0] for row in merged["/world/runs/slam_rs/rig:Transform3D:translation"].to_pylist()]
    assert {stamp: value for stamp, value in zip(stamps, rig, strict=True) if value is not None} == {10: 1.0, 30: 3.0}
    assert {stamp: value for stamp, value in zip(stamps, run, strict=True) if value is not None} == {10: 99.0, 20: 88.0}


def test_unscored_feed_does_not_open_ground_truth_storage(clock_recording: Path) -> None:
    """Layer generation needs only base cameras/IMU, even if GT storage is server-local."""
    with rr.server.Server(datasets={"test": [clock_recording]}) as server:
        source: CatalogSegment = CatalogSegment(
            server.url(),
            "test",
            "clock",
            dataset=server.client().get_dataset("test"),
            ground_truth_uri="file:///server-only/gt.rrd",
        )
        with open_segment(source, include_ground_truth=False) as feed:
            assert not feed.has_ground_truth
            assert len(feed.ground_truth_between(0, 100)) == 0
            assert len(feed.frame_t_ns) == 2


def test_feed_refuses_missing_calibration_before_decoding(tmp_path: Path) -> None:
    path: Path = tmp_path / "uncalibrated.rrd"
    with rr.RecordingStream("uncalibrated", recording_id="unknown") as recording:
        recording.save(path)
        recording.log("/world/rig_00", rr.AnyValues(reference="imu_00", num_cameras=1), static=True)
        recording.log("/world/rig_00/imu_00", rr.AnyValues(kind="imu", applied_time_shift_ns=0), static=True)
    with pytest.raises(ValueError, match="unknown.*IMU calibration.*gyro_noise_density"), open_segment(LocalSegment(path)):
        pytest.fail("missing calibration was accepted")


def _colsfm_prior(dataset: DatasetEntry, segment: str) -> tuple[list[int], list[float]]:
    """``/world/rig_00`` read the way colsfm reads its default pose prior (``colsfm.catalog.read_catalog_trajectory``):
    across every layer, sorted by stamp, the first row of equal stamps kept. Returns stamps and x positions."""
    entity: str = "/world/rig_00"
    table: pa.Table = dataset.filter_segments(segment).filter_contents([entity]).reader(index="video_time").to_arrow_table()
    table = table.filter(table[f"{entity}:Transform3D:quaternion"].is_valid())
    stamps: np.ndarray = table["video_time"].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64)
    x: np.ndarray = np.asarray(table[f"{entity}:Transform3D:translation"].combine_chunks().flatten().flatten().to_numpy(zero_copy_only=False)).reshape(-1, 3)[:, 0]
    order: np.ndarray = np.argsort(stamps, kind="stable")
    stamps, x = stamps[order], x[order]
    keep: np.ndarray = np.concatenate([[True], np.diff(stamps) > 0])
    return stamps[keep].tolist(), x[keep].tolist()


def test_without_ground_truth_the_catalog_layer_alone_owns_the_rig_pose_prior(
    clock_recording: Path, tmp_path: Path
) -> None:
    """Base (no gt) + the slam_rs layer, whose final poses differ from the raw VIO: colsfm's prior is the final run alone.
    The layer is the run's only writer (the replay writes the same layer), so no raw VIO row reaches the prior."""
    from slam_rs.catalog_layer import POSE_SOURCE, write_layer
    from slam_rs.tracking import SegmentRun
    from slam_rs.trajectory import empty_trajectory

    raw: Trajectory = Trajectory(t_ns=np.array([10, 20, 30], dtype=np.int64), position_m=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
                                 quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)))
    final: Trajectory = Trajectory(t_ns=raw.t_ns, position_m=raw.position_m + [100.0, 0.0, 0.0], quaternion_wxyz=raw.quaternion_wxyz)
    result_path: Path = tmp_path / "slam_rs.rrd"
    write_layer(result_path, "clock", SegmentRun(estimate=raw, final=final, framesets=3, lost=0, wall_s=0.1, ground_truth=empty_trajectory(),
                                                 median_tracker_ms=1.0, config_sha256="test", calibration_sha256="test-calibration"),
                clock_offset_ns=0, profile="fast", backend="cpu", animate_base_rig=True)
    with rr.server.Server(datasets={"test": [clock_recording]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("test")
        dataset.register([result_path.as_uri()], layer_name=POSE_SOURCE).wait()
        assert _colsfm_prior(dataset, "clock") == ([10, 20, 30], [101.0, 102.0, 103.0])
