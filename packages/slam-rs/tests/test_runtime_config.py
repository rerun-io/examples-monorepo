"""Normal SLAM configuration is independent of regression definitions."""

from pathlib import Path
from shutil import copytree

import pytest

from slam_rs import _core
from slam_rs.config import SLAM_CONFIG_PATH, SlamConfig, load_slam_config
from slam_rs.tracking import robocap_estimator_files


@pytest.mark.parametrize("benchmark_text", [None, "not valid TOML = ["])
def test_runtime_configuration_needs_no_benchmarks(tmp_path: Path, benchmark_text: str | None) -> None:
    path: Path = tmp_path / "slam.toml"
    path.write_text(SLAM_CONFIG_PATH.read_text())
    copytree(SLAM_CONFIG_PATH.parent / "configs", tmp_path / "configs")
    if benchmark_text is not None:
        (tmp_path / "benchmarks.toml").write_text(benchmark_text)
    settings: SlamConfig = load_slam_config(path)
    assert settings.robocap.camera_names == ("left", "left_front", "right_front", "right")
    assert settings.package_root == tmp_path
    calibration: _core.Calibration
    flow: _core.VioConfig
    text: str
    calibration, flow, text = robocap_estimator_files(settings, "fast")
    assert calibration.camera_count == 4
    assert list(calibration.resolution) == [(640, 360)] * 4
    assert text == settings.vio_config_text("msd-odyssey", "fast")
    assert flow.optical_flow_image_safe_radius == 388.0
