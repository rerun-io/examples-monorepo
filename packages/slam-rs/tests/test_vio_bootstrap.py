"""``config.vio_bootstrap`` through the binding, and the replay's mono-proxy option."""

import json
from pathlib import Path

import numpy as np
import pytest

from slam_rs import _core
from slam_rs.config import SLAM_CONFIG_PATH


def test_the_stereo_start_is_the_default_and_is_not_written_out() -> None:
    text: str = (SLAM_CONFIG_PATH.parent / "configs/msdmi_config.json").read_text()
    config: _core.VioConfig = _core.VioConfig.from_json(text)
    assert config.vio_bootstrap == "stereo"
    assert "vio_bootstrap" not in config.to_json()


@pytest.mark.parametrize("value", ["inertial", "auto", "stereo"])
def test_the_key_round_trips(value: str) -> None:
    config: _core.VioConfig = _core.VioConfig()
    config.vio_bootstrap = value
    assert config.vio_bootstrap == value
    written: dict = json.loads(config.to_json())["value0"]
    assert written.get("config.vio_bootstrap", "stereo") == value
    assert _core.VioConfig.from_json(config.to_json()).vio_bootstrap == value


def test_an_unknown_value_is_refused() -> None:
    config: _core.VioConfig = _core.VioConfig()
    with pytest.raises(ValueError, match="stereo, inertial or auto"):
        config.vio_bootstrap = "mono"


def test_a_stereo_pipeline_has_no_bootstrap_report() -> None:
    root: Path = SLAM_CONFIG_PATH.parent
    calibration = _core.Calibration.from_json((root / "crates/slam-rs/tests/fixtures/msdmi_calib.json").read_text())
    config: _core.VioConfig = _core.VioConfig.from_json((root / "configs/msdmi_config.json").read_text())
    assert _core.Vio(calibration, config, backend="cpu").bootstrap_report() is None
    config.vio_bootstrap = "inertial"
    vio: _core.Vio = _core.Vio(calibration, config, backend="cpu")
    assert json.loads(vio.bootstrap_report() or "") == {"attempts": [], "start_t_ns": None, "accepted_at_t_ns": None, "replayed": 0}


def test_blanked_cameras_track_nothing() -> None:
    """A black image gives the frontend nothing to detect or track (DESIGN H3)."""
    root: Path = SLAM_CONFIG_PATH.parent
    calibration = _core.Calibration.from_json((root / "crates/slam-rs/tests/fixtures/msdmi_calib.json").read_text())
    config: _core.VioConfig = _core.VioConfig.from_json((root / "configs/msdmi_config.json").read_text())
    flow = _core.OpticalFlow(calibration, config)
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 256, size=(960, 960), dtype=np.uint8)
    black = np.zeros((960, 960), dtype=np.uint8)
    for k in range(3):
        frame = flow.process(k * 33_333_333, [np.roll(textured, k, axis=1), black])
    assert len(frame.ids(0)) > 0
    assert len(frame.ids(1)) == 0


@pytest.mark.parametrize(
    "dataset",
    ["insta360_config", "insta360_native_config", "tartanair_equirect_config", "tartanair_fisheye_config", "tartanair_pinhole_config"],
)
def test_rigs_that_start_in_motion_bootstrap_inertially(dataset: str) -> None:
    """The Insta360 has no stereo pair and every TartanAir clip starts moving: the stereo start cannot work there."""
    text: str = (SLAM_CONFIG_PATH.parent / f"configs/{dataset}.json").read_text()
    assert _core.VioConfig.from_json(text).vio_bootstrap == "inertial"
