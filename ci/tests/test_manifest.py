"""Workspace structure checks that need no package environment."""

import re
from pathlib import Path

import pytest
import tomllib

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def test_every_feature_declares_platforms() -> None:
    manifest: dict = tomllib.loads((REPO_ROOT / "pixi.toml").read_text())
    missing: list[str] = [f"feature.{name}" for name, feature in manifest["feature"].items() if "platforms" not in feature]
    assert not missing, "Features without platforms:\n" + "\n".join(missing)


@pytest.fixture(scope="module")
def pyrefly() -> tuple[dict, set[Path]]:
    """pyrefly.toml plus its expanded project-includes, read once for all package cases."""
    config: dict = tomllib.loads((REPO_ROOT / "pyrefly.toml").read_text())
    included: set[Path] = {path for pattern in config["project-includes"] for path in REPO_ROOT.glob(pattern)}
    return config, included


@pytest.mark.parametrize(
    "pyproject",
    [
        pytest.param(
            path,
            id=path.parent.name,
            marks=pytest.mark.xfail(
                strict=True,
                reason="sam2-streaming: sam2 missing project-includes; sam2-streaming-dev missing site-package-path",
            )
            if path.parent.name == "sam2-streaming"
            else (),
        )
        for path in sorted((REPO_ROOT / "packages").glob("*/pyproject.toml"))
        if "project" in tomllib.loads(path.read_text())
    ],
)
def test_runnable_packages_are_registered_with_pyrefly(pyproject: Path, pyrefly: tuple[dict, set[Path]]) -> None:
    config, included = pyrefly
    problems: list[str] = []
    package_dir: Path = pyproject.parent
    # Rust packages may have src/ alongside a flat Python module. Discover
    # both layouts; tool/ (sam3d-body), tools/ and tests/ are not modules.
    modules: list[Path] = [
        init.parent
        for source_root in (package_dir, package_dir / "src")
        for init in sorted(source_root.glob("*/__init__.py"))
        if init.parent.name not in {"tests", "tools", "tool"}
    ]
    assert modules, f"{package_dir.name}: no Python module found"
    for module in modules:
        module_path: str = module.relative_to(REPO_ROOT).as_posix()
        if module / "__init__.py" not in included:
            problems.append(f"{module_path}: missing project-includes")
        if module.parent.relative_to(REPO_ROOT).as_posix() not in config["search-path"]:
            problems.append(f"{module_path}: missing search-path")
    site_pattern: str = rf"\.pixi/envs/{re.escape(package_dir.name)}-dev/lib/python3\.\d+/site-packages"
    if not any(re.fullmatch(site_pattern, path) for path in config["site-package-path"]):
        problems.append(f"{package_dir.name}: missing site-package-path for {package_dir.name}-dev")
    assert not problems, "Missing Pyrefly registrations:\n" + "\n".join(problems)


def test_package_features_have_prod_and_dev_environments() -> None:
    manifest: dict = tomllib.loads((REPO_ROOT / "pixi.toml").read_text())
    environments: dict = manifest["environments"]
    problems: list[str] = []
    for name in sorted(manifest["feature"]):
        if not (REPO_ROOT / "packages" / name).is_dir():
            continue
        for env_name in (name, f"{name}-dev"):
            if env_name not in environments:
                problems.append(f"feature.{name}: missing environment {env_name}")
        if f"{name}-dev" in environments and "dev" not in environments[f"{name}-dev"]["features"]:
            problems.append(f"{name}-dev: missing dev feature")
    assert not problems, "Package environment inconsistencies:\n" + "\n".join(problems)


def _task_commands(table: dict, prefix: str = "") -> dict[str, str | list[str]]:
    """Find tasks at workspace, feature and target scope, including shorthand."""
    commands: dict[str, str | list[str]] = {}
    for key, value in table.items():
        path: str = f"{prefix}.{key}" if prefix else key
        if key == "tasks":
            for name, task in value.items():
                command: str | list[str] | None = task if isinstance(task, str) else task.get("cmd")
                if command is not None:
                    commands[f"{path}.{name}"] = command
        elif isinstance(value, dict):
            commands.update(_task_commands(value, path))
    return commands




@pytest.mark.parametrize(
    ("name", "command"),
    [pytest.param(name, command, id=name) for name, command in _task_commands(tomllib.loads((REPO_ROOT / "pixi.toml").read_text())).items()],
)
def test_task_commands_are_single_line(name: str, command: str | list[str]) -> None:
    """Pixi collapses newlines inside a multiline cmd into spaces (AGENTS.md gotcha), so a
    command split over lines without `&&` / `\\` silently becomes arguments to its first word.
    A trailing newline before the closing quotes is harmless and ignored here."""
    parts: list[str] = [command] if isinstance(command, str) else command
    assert not any("\n" in part.strip() or "\r" in part.strip() for part in parts), f"{name}: cmd contains an inner newline"
