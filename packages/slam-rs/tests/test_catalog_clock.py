"""Camera-aligned catalog samples keep their relative timing at the estimator boundary."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer
from simplecv.imu_calibration import ImuCalibration

from slam_rs.catalog_feed import CatalogSegment, ImuStream, LocalSegment, open_segment
from slam_rs.trajectory import Trajectory


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
@pytest.mark.parametrize("cache_video", [False, True])
def test_camera_and_imu_share_one_clock_and_window(clock_recording: Path, offset_ns: int, cache_video: bool) -> None:
    """Select the sample at the frame's estimator time, excluding both neighbours."""
    with open_segment(LocalSegment(clock_recording), cache_video=cache_video) as feed:
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
