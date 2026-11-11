"""What a public clone relies on: weights from Hugging Face at pinned revisions, no host paths, a one-command example."""

import tomllib
from pathlib import Path

import pytest

REPO: Path = Path(__file__).resolve().parents[3]
FEATURE: dict = tomllib.loads((REPO / "pixi.toml").read_text())["feature"]["metrictrail"]
WEIGHTS_TASK: str = "_download-metrictrail-weights"
WEIGHT_VARS: tuple[str, ...] = ("VIPE_METRICTRAIL_DPVO_CHECKPOINT", "VIPE_DINOV3_VITS16_CHECKPOINT")


def test_the_activation_env_names_no_host_path() -> None:
    env: dict[str, str] = FEATURE["activation"]["env"]
    host_paths: dict[str, str] = {name: value for name, value in env.items() if "/mnt/" in value or "/home/" in value}
    assert not host_paths


def test_the_weight_defaults_are_repo_local_and_the_shell_wins() -> None:
    env: dict[str, str] = FEATURE["activation"]["env"]
    for name in WEIGHT_VARS:
        assert env[name].startswith(f"${{{name}:-$PIXI_PROJECT_ROOT/packages/metrictrail/data/weights/")
    # ViPE's own downloads (droid.pth, moge-2-vitl) go to the user's torch hub and Hugging Face caches.
    assert "TORCH_HOME" not in env
    assert "VIPE_MOGE2_CHECKPOINT" not in env


def test_the_weights_task_writes_the_files_the_env_names() -> None:
    env: dict[str, str] = FEATURE["activation"]["env"]
    task: dict = FEATURE["tasks"][WEIGHTS_TASK]
    defaults: set[str] = {env[name].removesuffix("}").split("$PIXI_PROJECT_ROOT/", 1)[1] for name in WEIGHT_VARS}
    assert set(task["outputs"]) == defaults


def test_the_weights_task_pins_its_revisions() -> None:
    cmd: str = FEATURE["tasks"][WEIGHTS_TASK]["cmd"]
    assert "nvidia/MetricTrail dino_dpvo.pth --revision f5a3ce1cebd2a6a35c30ffb138caa07122731fff" in cmd
    assert "xysun917/dinov3-vits16 dinov3_vits16_pretrain_lvd1689m-08c60483.pth --revision 4f097416bd15cb1ca527bac01b39e1e64ef31f8a" in cmd


@pytest.mark.parametrize("task", ["metrictrail-run", "metrictrail-app", "metrictrail-example", "tests-integration"])
def test_the_run_tasks_fetch_the_weights_first(task: str) -> None:
    assert WEIGHTS_TASK in FEATURE["tasks"][task]["depends-on"]


def test_the_example_runs_the_iphone_clip_with_the_video_preset() -> None:
    task: dict = FEATURE["tasks"]["metrictrail-example"]
    assert {"lietorch-mojo-build", "dpvo-mojo-build", "vipe-mojo-build", "_download-metrictrail-iphone-data"} <= set(task["depends-on"])
    assert "--input data/iphone/pool.mp4 --benchmark video --output data/" in task["cmd"]
