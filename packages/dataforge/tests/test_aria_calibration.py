"""Aria factory calibrations read without projectaria-tools, against what projectaria-tools read from them.

``fixtures/aria/<device>-calib.json`` are the ``calib_json`` tags of a HOT3D
Aria Gen1 recording (P0015_179e1b84) and an Aria Gen2 Pilot one (clean_0),
verbatim; ``<device>-projectaria.json`` beside each holds what
``device_calibration_from_json_string`` (projectaria-tools 2.3) made of them.
Every value must match exactly.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import vrs_file
from simplecv.camera_parameters import Fisheye624Parameters
from simplecv.se3 import SE3

from dataforge import aria, hands
from dataforge.vrs import VrsFile

FIXTURES: Path = Path(__file__).parent / "fixtures" / "aria"


@pytest.mark.parametrize("device", ["gen1-hot3d-P0015_179e1b84", "gen2-pilot-clean_0"])
def test_device_calibration_is_the_sdks(device: str) -> None:
    calibration = aria.DeviceCalibration.from_json((FIXTURES / f"{device}-calib.json").read_text(), device)
    expected = json.loads((FIXTURES / f"{device}-projectaria.json").read_text())
    assert set(calibration.cameras) == set(expected["cameras"]), "eye-tracking cameras are not FISHEYE624 and are left out"
    for label, camera in expected["cameras"].items():
        actual = calibration.camera(label)
        assert (actual.width, actual.height, actual.valid_radius, actual.max_solid_angle) == (
            camera["width"],
            camera["height"],
            camera["valid_radius"],
            camera["max_solid_angle"],
        ), label
        np.testing.assert_array_equal(actual.params, camera["params"])
        np.testing.assert_array_equal(actual.rig_T_cam.matrix(), camera["device_T_camera"])
    assert set(calibration.device_T_imu) == set(expected["device_T_imu"])
    for label, matrix in expected["device_T_imu"].items():
        np.testing.assert_array_equal(calibration.imu(label).matrix(), matrix)


def test_config_data_overrides_the_defaults_and_unknown_devices_fail() -> None:
    document = json.loads((FIXTURES / "gen1-hot3d-P0015_179e1b84-calib.json").read_text())
    document["CameraCalibrations"][0]["ConfigData"] = {"ImageWidth": 320, "ImageHeight": 240, "MaxSolidAngle": 1.2}
    label = document["CameraCalibrations"][0]["Label"]
    camera = aria.DeviceCalibration.from_json(json.dumps(document), "edited").camera(label)
    assert (camera.width, camera.height, camera.max_solid_angle, camera.valid_radius) == (320, 240, 1.2, None)
    document["DeviceClassInfo"]["DeviceClass"] = "Quest"
    with pytest.raises(ValueError, match="unknown Aria device class Quest"):
        aria.DeviceCalibration.from_json(json.dumps(document), "edited")
    with pytest.raises(ValueError, match="no FISHEYE624 camera-et-left"):
        aria.DeviceCalibration.from_json((FIXTURES / "gen1-hot3d-P0015_179e1b84-calib.json").read_text(), "gen1").camera("camera-et-left")


def test_gen1_rgb_is_fitted_to_its_1408_stream() -> None:
    calibration = aria.DeviceCalibration.from_json((FIXTURES / "gen1-hot3d-P0015_179e1b84-calib.json").read_text(), "gen1")
    factory = calibration.camera("camera-rgb")
    stream = aria.rescale_to_stream(factory, 1408, 1408)
    assert (stream.width, stream.height, stream.valid_radius) == (1408, 1408, 1415.0 * 0.5)
    assert stream.params[1] == 0.5 * (factory.params[1] - 32.0 + 0.5) - 0.5
    assert aria.rescale_to_stream(calibration.camera("camera-slam-left"), 640, 480) is calibration.camera("camera-slam-left")
    with pytest.raises(ValueError, match="no rescale from 2880x2880 to the 1000x1000 stream"):
        aria.rescale_to_stream(factory, 1000, 1000)


def test_read_device_calibration_needs_the_file_tag(tmp_path: Path) -> None:
    text = (FIXTURES / "gen2-pilot-clean_0-calib.json").read_text()
    tagged = tmp_path / "tagged.vrs"
    tagged.write_bytes(vrs_file({214: {}}, [], file_tags={"calib_json": text}))
    assert set(aria.read_device_calibration(VrsFile(tagged)).device_T_imu) == {"imu-left", "imu-right"}
    bare = tmp_path / "bare.vrs"
    bare.write_bytes(vrs_file({214: {}}, []))
    with pytest.raises(ValueError, match="carries no device calibration"):
        aria.read_device_calibration(VrsFile(bare))


@pytest.mark.parametrize("valid_radius,width", [(10.0, 640), (None, 330)])
def test_projection_rejects_valid_radius_and_image_bounds(valid_radius: float | None, width: int) -> None:
    camera = Fisheye624Parameters(
        "test", width, 480, np.array([300.0, 320.0, 240.0, *([0.0] * 12)]), SE3.from_quaternion(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)), 1.5, valid_radius
    )
    positions = np.full((1, 133, 3), np.nan, dtype=np.float32)
    positions[0, 91] = [0.5, 0.0, 1.0]
    projected = aria.project_to_calibration(camera, np.eye(4)[None], positions)
    assert np.isnan(projected).all()
    _, confidence = hands.confidence_rule(projected.astype(np.float32), np.ones((1, 133), dtype=np.float32))
    assert (confidence == 0.0).all()


def test_projection_goes_through_the_device_pose_and_skips_missing_poses_and_points_behind() -> None:
    # The camera sits 1 m along the device's x axis; the device is 2 m along the world's z axis.
    camera = Fisheye624Parameters(
        "test", 640, 480, np.array([300.0, 320.0, 240.0, *([0.0] * 12)]), SE3.from_quaternion(np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])), 1.5, None
    )
    world_T_device = np.stack([np.eye(4), np.eye(4), np.full((4, 4), np.nan)])
    world_T_device[:2, 2, 3] = 2.0
    positions = np.full((3, 133, 3), np.nan, dtype=np.float32)
    positions[:, 0] = [1.0, 0.0, 3.0]  # 1 m in front of the camera, on its optical axis
    positions[:, 1] = [1.0, 0.0, 1.0]  # 1 m behind it
    projected = aria.project_to_calibration(camera, world_T_device, positions)
    np.testing.assert_allclose(projected[:2, 0], [[320.0, 240.0]] * 2, atol=1e-9)
    assert np.isnan(projected[:2, 1:]).all()
    assert np.isnan(projected[2]).all()
