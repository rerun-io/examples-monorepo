"""What an outsider's quickstart relies on: a local catalog by default, the pixi tasks, and a gate tool with a plain name."""

import importlib
import tomllib
from pathlib import Path

import pytest

from slam_rs.config import SlamConfig

PACKAGE: Path = Path(__file__).resolve().parents[1]
REPO: Path = PACKAGE.parents[1]


def test_the_checked_in_settings_point_at_a_local_catalog(settings: SlamConfig) -> None:
    assert settings.catalog_url == "rerun+http://127.0.0.1:51235"


@pytest.mark.parametrize(
    "task",
    ["slam-rs-serve", "slam-rs-register", "slam-rs-gate", "slam-rs-download-smoke", "slam-rs-download-release", "slam-rs-download-all"],
)
def test_the_quickstart_tasks_exist(task: str) -> None:
    with (REPO / "pixi.toml").open("rb") as handle:
        tasks: dict[str, object] = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]
    assert task in tasks


def test_the_download_tasks_pull_from_the_public_dataset() -> None:
    with (REPO / "pixi.toml").open("rb") as handle:
        tasks: dict[str, dict[str, str]] = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]
    for name in ("slam-rs-download-smoke", "slam-rs-download-release", "slam-rs-download-all"):
        assert "pablovela5620/msd-rrd" in tasks[name]["cmd"]
    smoke: str = tasks["slam-rs-download-smoke"]["cmd"]
    for clip in ("MIO10_short_2_panorama", "MGO09_short_1_updown", "MOO09_short_1_updown"):
        assert clip in smoke


def test_the_gate_tool_carries_no_machine_specific_name() -> None:
    assert (PACKAGE / "tools" / "apps" / "gate.py").is_file()
    assert not (PACKAGE / "tools" / "apps" / "fleet_check.py").exists()
    assert importlib.import_module("slam_rs.apis.gate").main is not None
    assert not importlib.util.find_spec("slam_rs.apis.fleet_check")
