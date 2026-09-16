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
    ["slam-rs-serve", "slam-rs-register", "slam-rs-gate", "slam-rs-download-sample", "slam-rs-download-all"],
)
def test_the_quickstart_tasks_exist(task: str) -> None:
    with (REPO / "pixi.toml").open("rb") as handle:
        tasks: dict[str, object] = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]
    assert task in tasks


def test_the_download_tasks_pull_from_the_public_dataset() -> None:
    with (REPO / "pixi.toml").open("rb") as handle:
        tasks: dict[str, dict[str, str]] = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]
    for name in ("slam-rs-download-sample", "slam-rs-download-all"):
        assert "pablovela5620/msd-rrd" in tasks[name]["cmd"]
    assert "--include" not in tasks["slam-rs-download-all"]["cmd"]


def test_the_sample_is_exactly_what_the_smoke_gate_scores() -> None:
    with (REPO / "pixi.toml").open("rb") as handle:
        sample: str = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]["slam-rs-download-sample"]["cmd"]
    with (PACKAGE / "benchmarks.toml").open("rb") as handle:
        smoke: list[str] = [seg["segment_id"] for seg in tomllib.load(handle)["segment"] if seg["tier"] == "smoke"]
    assert smoke and all(segment_id.rsplit("__", 1)[-1] in sample for segment_id in smoke)
    assert sample.count("--include") == len(smoke)


def test_registration_is_dataforges_job() -> None:
    """The download mirrors dataforge's layer-major layout, so its register tool is the one that runs; slam-rs keeps no copy."""
    with (REPO / "pixi.toml").open("rb") as handle:
        task: dict[str, str | dict[str, str]] = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]["slam-rs-register"]
    cmd: str | dict[str, str] = task["cmd"]
    env: str | dict[str, str] = task["env"]
    assert isinstance(cmd, str) and isinstance(env, dict)
    assert "dataforge/tools/apps/register.py msd --device $d" in cmd and "index g2 odyssey" in cmd
    assert env["DATAFORGE_OUTPUT_ROOT"] == "data/msd-rrd"
    assert not importlib.util.find_spec("slam_rs.apis.register")
    assert not (PACKAGE / "tools" / "apps" / "register.py").exists()


def test_the_gate_tool_carries_no_machine_specific_name() -> None:
    assert (PACKAGE / "tools" / "apps" / "gate.py").is_file()
    assert not (PACKAGE / "tools" / "apps" / "fleet_check.py").exists()
    assert importlib.import_module("slam_rs.apis.gate").main is not None
    assert not importlib.util.find_spec("slam_rs.apis.fleet_check")
