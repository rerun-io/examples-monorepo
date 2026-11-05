"""Calibrated camera records: the committed SelfCap OAK fixture and schema checks."""

from pathlib import Path

import numpy as np
import pytest

from posekit.cameras import CalibratedCamera, load_calibrated_camera

FIXTURE: Path = Path(__file__).resolve().parents[1] / "assets" / "examples" / "selfcap_oak_left.json"


def test_selfcap_oak_fixture_is_a_rational_camera() -> None:
    """The OAK left factory calibration of the smoke episode, as dataforge reads it."""
    camera = load_calibrated_camera(FIXTURE)
    assert camera.lens == "RATIONAL_POLYNOMIAL"
    assert (camera.width, camera.height) == (1280, 720)
    assert camera.intrinsics[0, 0] == pytest.approx(566.8902587890625)
    assert camera.distortion[0] == pytest.approx(4.770694255828857)


def test_wrong_coefficient_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="FISHEYE624 needs 12"):
        CalibratedCamera(np.eye(3), np.zeros(14), 640, 480, lens="FISHEYE624")


def test_malformed_file_names_its_source(tmp_path: Path) -> None:
    path = tmp_path / "camera.json"
    path.write_text('{"intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}')
    with pytest.raises(ValueError, match="camera.json"):
        load_calibrated_camera(path)
