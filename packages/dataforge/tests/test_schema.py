import pytest

from dataforge.schema import (
    EXOEGO_SCHEMA_VERSION,
    TIMELINE,
    accel_path,
    cam_path,
    capture_property,
    gyro_path,
    imu_path,
    pinhole_path,
    rig_path,
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


def test_indices_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rig_path(-1)


def test_capture_property_keys() -> None:
    assert capture_property("num_frames") == "property:capture:num_frames"
    assert capture_property("schema") == "property:capture:schema"
