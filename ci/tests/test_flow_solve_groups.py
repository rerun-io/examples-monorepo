"""The stage envs a flow chains share one solve group: one version of each package across its steps (study Q3)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import tomllib

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FLOW_FILE: Path = REPO_ROOT / "packages" / "pipelines" / "pipelines" / "capture_to_splat.py"


def _flow_module() -> ModuleType:
    """The flow's step list module, by file: it imports only the standard library, so the ci env can load it."""
    spec = importlib.util.spec_from_file_location("ci_capture_to_splat", FLOW_FILE)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve the module of a class through sys.modules
    spec.loader.exec_module(module)
    return module


def test_every_step_of_a_capture_to_splat_profile_runs_in_one_solve_group(tmp_path: Path) -> None:
    environments: dict[str, dict[str, object]] = tomllib.loads((REPO_ROOT / "pixi.toml").read_text())["environments"]
    flow: ModuleType = _flow_module()
    for profile in ("insta360", "robocap"):
        steps = flow.flow_steps(profile, segment=f"{profile}__s", catalog_url="u", output_root=tmp_path)
        groups: set[object] = {environments[step.environment].get("solve-group") for step in steps}
        assert len(groups) == 1 and None not in groups, f"{profile}: {groups}"
