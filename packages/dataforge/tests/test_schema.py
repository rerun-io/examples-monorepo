import pytest

from dataforge.schema import (
    EXOEGO_SCHEMA_VERSION,
    TIMELINE,
    accel_path,
    cam_path,
    capture_property,
    control_points_path,
    cp_uv_path,
    field_path,
    gyro_path,
    heading_path,
    imu_path,
    mag_path,
    pinhole_path,
    rig_path,
    run_path,
    trail_path,
    trajectory_path,
    video_path,
)


def test_timeline_and_schema_version_match_exoego_v2() -> None:
    assert TIMELINE == "video_time"
    assert EXOEGO_SCHEMA_VERSION == "exoego:v2"


def test_entity_paths_are_zero_padded_exoego_v2() -> None:
    assert rig_path(0) == "/world/rig_00"
    assert cam_path(0, 3) == "/world/rig_00/cam_03"
    assert pinhole_path(1, 0) == "/world/rig_01/cam_00/pinhole"
    assert video_path(1, 0) == "/world/rig_01/cam_00/pinhole/video"
    assert imu_path(0, 0) == "/world/rig_00/imu_00"
    assert gyro_path(0, 0) == "/world/rig_00/imu_00/gyro"
    assert accel_path(0, 0) == "/world/rig_00/imu_00/accel"
    assert mag_path(0, 1) == "/world/rig_00/mag_01"
    assert field_path(0, 1) == "/world/rig_00/mag_01/field"
    assert heading_path(0, 1) == "/world/rig_00/mag_01/heading"


def test_indices_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rig_path(-1)


def test_capture_property_keys() -> None:
    assert capture_property("num_frames") == "property:capture:num_frames"
    assert capture_property("schema") == "property:capture:schema"


def test_run_paths_are_nested_below_the_source() -> None:
    assert run_path("basalt") == "/world/runs/basalt"
    assert trajectory_path("basalt") == "/world/runs/basalt/trajectory"
    assert trail_path("basalt") == "/world/runs/basalt/trail"


def test_surveyed_control_points_live_under_the_world_gt_node() -> None:
    """exoego:v2 §5 keeps ground truth at ``/world/gt/...``, independent of the rig layout."""
    assert control_points_path() == "/world/gt/control_points"


def test_control_point_detections_hang_off_the_camera_that_saw_them() -> None:
    """A 2D detection is a per-camera derived quantity, so it sits under that camera's pinhole."""
    assert cp_uv_path(0, 1) == "/world/rig_00/cam_01/pinhole/cp_uv"
    assert cp_uv_path(0, 1).startswith(pinhole_path(0, 1))
    with pytest.raises(ValueError, match="non-negative"):
        cp_uv_path(0, -1)


def test_magnetometer_is_a_peer_sensor_of_the_imu() -> None:
    """exoego:v2 hangs every sensor off the rig, never off another sensor."""
    assert mag_path(2, 0).rsplit("/", 1)[0] == rig_path(2)
    with pytest.raises(ValueError, match="non-negative"):
        mag_path(0, -1)
