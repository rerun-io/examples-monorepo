"""What an outsider's quickstart relies on: a local catalog by default, the pixi tasks, and a gate tool with a plain name."""

import importlib
import tomllib
from pathlib import Path

import pytest

from slam_rs.config import SlamConfig
from slam_rs.reference import Benchmarks

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


DEMO_CLIP: str = "msd-g2__MGO_others__MGO07_mapping_easy"
"""The four-camera clip the README's gif shows; the demo replays it."""


def test_the_sample_holds_the_smoke_clips_and_the_demo_clip(benchmarks: Benchmarks) -> None:
    with (REPO / "pixi.toml").open("rb") as handle:
        sample: str = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]["slam-rs-download-sample"]["cmd"]
    smoke: list[str] = [segment.segment_id for segment in benchmarks.in_tier("smoke")]
    assert smoke and all(segment_id.rsplit("__", 1)[-1] in sample for segment_id in smoke)
    assert DEMO_CLIP.rsplit("__", 1)[-1] in sample
    assert sample.count("--include") == len(smoke) + 1


def test_a_download_is_skipped_only_when_every_layer_of_its_recordings_is_there() -> None:
    """The guard is the recordings themselves across all three layers, not a marker and not the base layer alone."""
    with (REPO / "pixi.toml").open("rb") as handle:
        tasks: dict[str, dict[str, object]] = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]
    for name, count in (("slam-rs-download-sample", 9), ("slam-rs-download-all", 192)):
        cmd: str = str(tasks[name]["cmd"])
        assert ".sample" not in cmd and ".tier" not in cmd and ".all" not in cmd
        assert "outputs" not in tasks[name]
        assert all(layer in cmd for layer in ("base", "gt", "sensor_metadata")), name
        assert str(count) in cmd, f"{name} checks for all {count} files"
    assert "MGO07_mapping_easy" in str(tasks["slam-rs-download-sample"]["cmd"])


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


def test_one_command_runs_the_whole_demo_in_the_viewer() -> None:
    """`pixi run -e slam-rs slam-rs-demo` is the path for someone who just cloned: build, fetch the sample, serve, register, replay on screen.

    The demo owns nothing but the replay: the catalog and the registration are the
    tasks everyone else uses, chained through `depends-on`, so a fix lands once.
    """
    with (REPO / "pixi.toml").open("rb") as handle:
        tasks: dict[str, dict[str, object]] = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]
    demo: dict[str, object] = tasks["slam-rs-demo"]
    assert list(demo["depends-on"]) == ["slam-rs-build", "slam-rs-download-sample", "slam-rs-register"]  # type: ignore[arg-type]
    cmd: str = str(demo["cmd"])
    assert cmd.startswith("python tools/apps/replay.py --stage vio") and "headless" not in cmd, "only the replay, on screen"
    assert f"--segment {DEMO_CLIP}" in cmd, "the four-camera clip from the gif, not the two-camera smoke clip"
    assert demo["cwd"] == "packages/slam-rs"
    assert str(tasks["slam-rs-catalog-up"]["cmd"]) == "python tools/apps/catalog_up.py", "the lifecycle is Python, not a shell string"
    assert list(tasks["slam-rs-register"]["depends-on"]) == ["slam-rs-catalog-up"]  # type: ignore[arg-type]


def test_the_gate_never_rebuilds_the_core_it_is_handed() -> None:
    """`slam-rs-gate --gpu` must not depend on the CPU build: pixi would rebuild the CPU core over a GPU one."""
    with (REPO / "pixi.toml").open("rb") as handle:
        gate: dict[str, object] = tomllib.load(handle)["feature"]["slam-rs"]["tasks"]["slam-rs-gate"]
    assert "depends-on" not in gate
