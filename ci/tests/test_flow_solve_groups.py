"""The stage envs a flow chains share one solve group: one version of each package across its steps (study Q3)."""

import importlib
import sys
from pathlib import Path
from types import ModuleType

import tomllib

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _flow_module(name: str) -> ModuleType:
    """A flow's step list module. The flows and ``pipelines.step`` import only the standard library, so the ci env, which
    does not install the pipelines package, imports them from the source tree."""
    sys.path.insert(0, str(REPO_ROOT / "packages" / "pipelines"))
    try:
        return importlib.import_module(f"pipelines.{name}")
    finally:
        sys.path.remove(str(REPO_ROOT / "packages" / "pipelines"))


def test_every_step_of_a_capture_to_splat_profile_runs_in_one_solve_group(tmp_path: Path) -> None:
    environments: dict[str, dict[str, object]] = tomllib.loads((REPO_ROOT / "pixi.toml").read_text())["environments"]
    flow: ModuleType = _flow_module("capture_to_splat")
    for profile in ("insta360", "robocap"):
        steps = flow.flow_steps(profile, segment=f"{profile}__s", catalog_url="u", output_root=tmp_path)
        groups: set[object] = {environments[task.environment].get("solve-group") for step in steps for task in step.tasks}
        assert len(groups) == 1 and None not in groups, f"{profile}: {groups}"
