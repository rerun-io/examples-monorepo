"""Calibrated camera records: the committed SelfCap OAK fixture and schema checks."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from rerun.chunk import Chunk, LazyChunkStream

from posekit.cameras import CalibratedCamera, load_calibrated_camera, save_calibrated_camera

FIXTURE: Path = Path(__file__).resolve().parents[1] / "assets" / "examples" / "selfcap_oak_left.rrd"


def test_selfcap_oak_fixture_is_its_factory_calibration() -> None:
    """The OAK left factory calibration of the smoke episode, as dataforge reads it (the retired selfcap_oak_left.json, value for value)."""
    camera: CalibratedCamera = load_calibrated_camera(FIXTURE)
    np.testing.assert_array_equal(
        camera.intrinsics, [[566.8902587890625, 0.0, 637.5352172851562], [0.0, 566.606689453125, 372.6623840332031], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_array_equal(
        camera.distortion,
        [
            4.770694255828857,
            1.44001042842865,
            0.0001279787247767672,
            7.046706741675735e-05,
            0.022044576704502106,
            5.138538837432861,
            2.9000442028045654,
            0.19502846896648407,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.0019323817687109113,
            0.0009225755929946899,
        ],
    )
    assert (camera.width, camera.height, camera.lens) == (1280, 720, "RATIONAL_POLYNOMIAL")
    assert (camera.intrinsics.dtype, camera.distortion.dtype) == (np.float64, np.float64)


def test_round_trip_is_exact(tmp_path: Path) -> None:
    path: Path = tmp_path / "camera.rrd"
    for camera in (
        CalibratedCamera(np.array([[500.25, 0.0, 320.5], [0.0, 499.75, 240.0], [0.0, 0.0, 1.0]]), np.array([0.1, -1e-7, 0.3, 0.4]), 640, 480, lens="KB4"),
        CalibratedCamera(np.eye(3), np.zeros(0), 32, 16, lens="PINHOLE"),
    ):
        save_calibrated_camera(camera, path)
        loaded: CalibratedCamera = load_calibrated_camera(path)
        np.testing.assert_array_equal(loaded.intrinsics, camera.intrinsics)
        np.testing.assert_array_equal(loaded.distortion, camera.distortion)
        assert (loaded.width, loaded.height, loaded.lens) == (camera.width, camera.height, camera.lens)


def test_intrinsics_that_float32_would_round_are_rejected(tmp_path: Path) -> None:
    camera = CalibratedCamera(np.array([[500.1, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]), np.zeros(0), 640, 480, lens="PINHOLE")
    with pytest.raises(ValueError, match="float32"):
        save_calibrated_camera(camera, tmp_path / "camera.rrd")


def test_wrong_coefficient_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="FISHEYE624 needs 12"):
        CalibratedCamera(np.eye(3), np.zeros(14), 640, 480, lens="FISHEYE624")


def test_malformed_file_names_its_source(tmp_path: Path) -> None:
    path: Path = tmp_path / "camera.rrd"
    pinhole_only: Chunk = Chunk.from_columns("/camera", indexes=[], columns=rr.Pinhole.columns(image_from_camera=[np.eye(3)], resolution=[[640, 480]]))
    LazyChunkStream.from_iter([pinhole_only]).write_rrd(path, application_id="posekit-calibration", recording_id="broken")
    with pytest.raises(ValueError, match="camera.rrd"):
        load_calibrated_camera(path)
