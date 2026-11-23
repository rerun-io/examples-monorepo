"""Workspace structure checks that need no package environment."""

import re
from pathlib import Path

import pytest
import tomllib

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
MANIFEST: dict = tomllib.loads((REPO_ROOT / "pixi.toml").read_text())
# A runnable package is a packages/<x> directory with a root [feature.<x>] (AGENTS.md, "Adding a
# new package"). Vendored dependencies such as sam2-streaming have a pyproject but no feature;
# test_workspace_sources.py covers those.
RUNNABLE_PACKAGES: list[Path] = sorted(REPO_ROOT / "packages" / name for name in MANIFEST["feature"] if (REPO_ROOT / "packages" / name).is_dir())
# Rust-only workspaces have environments, but no Python tooling configuration.
RUNNABLE_PYTHON_PACKAGES: list[Path] = [package_dir for package_dir in RUNNABLE_PACKAGES if (package_dir / "pyproject.toml").is_file()]


def test_every_feature_declares_platforms() -> None:
    missing: list[str] = [f"feature.{name}" for name, feature in MANIFEST["feature"].items() if "platforms" not in feature]
    assert not missing, "Features without platforms:\n" + "\n".join(missing)


def test_one_torchcodec_and_one_ffmpeg() -> None:
    """Only feature.torchcodec declares torchcodec; every environment but robocap-recorder composes feature.ffmpeg."""
    declaring: list[str] = sorted(name for name, feature in MANIFEST["feature"].items() if "torchcodec" in feature.get("dependencies", {}))
    assert declaring == ["torchcodec"], f"torchcodec declared outside feature.torchcodec: {declaring}"
    missing: list[str] = sorted(
        name for name, environment in MANIFEST["environments"].items() if "ffmpeg" not in environment["features"] and name != "robocap-recorder"
    )
    assert not missing, "Environments without the ffmpeg feature:\n" + "\n".join(missing)


def test_osx_features_leave_the_linker_to_xcode() -> None:
    """No feature that solves osx-arm64 installs conda clang there: its ld64 956 cannot link against the macOS 27 SDK.

    That SDK's libSystem.tbd lists arm64e but no arm64 slice, so conda's ld64 skips it as a "malformed file" and every
    Rust build script fails with undefined _NSGetEnviron / _Unwind_* (gsplat-rust-renderer on the Mac mini, macOS 27.2).
    Xcode's /usr/bin/cc links it; a linux-only clang belongs under target.linux-*.
    """
    offending: list[str] = sorted(
        name for name, feature in MANIFEST["feature"].items()
        if "osx-arm64" in feature.get("platforms", []) and ("clang" in feature.get("dependencies", {}) or "clang" in feature.get("target", {}).get("osx-arm64", {}).get("dependencies", {}))
    )
    assert not offending, f"Features that put conda clang (and its ld64) on osx-arm64: {offending}"


@pytest.fixture(scope="module")
def pyrefly() -> tuple[dict, set[Path]]:
    """pyrefly.toml plus its expanded project-includes, read once for all package cases."""
    config: dict = tomllib.loads((REPO_ROOT / "pyrefly.toml").read_text())
    included: set[Path] = {path for pattern in config["project-includes"] for path in REPO_ROOT.glob(pattern)}
    return config, included


@pytest.mark.parametrize("package_dir", RUNNABLE_PYTHON_PACKAGES, ids=lambda package_dir: package_dir.name)
def test_runnable_packages_are_registered_with_pyrefly(package_dir: Path, pyrefly: tuple[dict, set[Path]]) -> None:
    config, included = pyrefly
    problems: list[str] = []
    # Rust packages may have src/ alongside a flat Python module. Discover
    # both layouts; tools/ and tests/ are not modules.
    modules: list[Path] = [
        init.parent
        for source_root in (package_dir, package_dir / "src")
        for init in sorted(source_root.glob("*/__init__.py"))
        if init.parent.name not in {"tests", "tools"}
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
    environments: dict = MANIFEST["environments"]
    problems: list[str] = []
    for name in (package_dir.name for package_dir in RUNNABLE_PACKAGES):
        for env_name in (name, f"{name}-dev"):
            if env_name not in environments:
                problems.append(f"feature.{name}: missing environment {env_name}")
        if f"{name}-dev" in environments and "dev" not in environments[f"{name}-dev"]["features"]:
            problems.append(f"{name}-dev: missing dev feature")
    assert not problems, "Package environment inconsistencies:\n" + "\n".join(problems)


@pytest.mark.parametrize("package_dir", RUNNABLE_PYTHON_PACKAGES, ids=lambda package_dir: package_dir.name)
def test_runnable_packages_configure_test_tiers(package_dir: Path) -> None:
    config: dict = tomllib.loads((package_dir / "pyproject.toml").read_text())
    options: dict = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
    markers: set[str] = {marker.partition(":")[0].strip() for marker in options.get("markers", [])}
    problems: list[str] = []
    for marker in ("integration", "golden"):
        if marker not in markers:
            problems.append(f"{package_dir.name}: missing {marker} marker registration")
    if "-m 'not integration and not golden'" not in options.get("addopts", ""):
        problems.append(f"{package_dir.name}: addopts must contain -m 'not integration and not golden'")
    assert not problems, "Missing test tier configuration:\n" + "\n".join(problems)


def test_packages_do_not_use_retired_pytest_markers() -> None:
    retired: re.Pattern[str] = re.compile(r"\bmark\.(?:slow|slow_cuda|hardware)\b|[\"'](?:slow|slow_cuda|hardware)\s*:")
    paths: list[Path] = sorted([*REPO_ROOT.glob("packages/*/pyproject.toml"), *REPO_ROOT.glob("packages/**/conftest.py")])
    problems: list[str] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if retired.search(line):
                problems.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")
    assert not problems, "Retired pytest markers (use integration or golden):\n" + "\n".join(problems)


def test_package_code_never_imports_beartype() -> None:
    """Beartype runs only under pytest (pytest-beartype): package code never imports it (AGENTS.md, Beartype)."""
    top_level_import: re.Pattern[str] = re.compile(r"^(?:from|import) beartype\b", re.MULTILINE)
    problems: list[str] = []
    for package_dir in RUNNABLE_PYTHON_PACKAGES:
        for path in sorted(package_dir.rglob("*.py")):
            relative: Path = path.relative_to(package_dir)
            if relative.parts[0] in {"tests", ".pixi"} or path.name == "conftest.py":
                continue
            if top_level_import.search(path.read_text(errors="replace")):
                problems.append(str(path.relative_to(REPO_ROOT)))
    assert not problems, "beartype imported outside tests (use the sys.modules lookup from AGENTS.md):\n" + "\n".join(problems)


def test_beartype_packages_fail_when_uncheckable() -> None:
    """Every pytest-beartype config turns the plugin's "uncheckable by beartype" warning into an error (AGENTS.md, Beartype)."""
    entry: str = "error::pytest_beartype.roar.PytestBeartypeOptionPackagesWarning"
    problems: list[str] = []
    for pyproject in sorted(REPO_ROOT.glob("packages/*/pyproject.toml")):
        options: dict = tomllib.loads(pyproject.read_text()).get("tool", {}).get("pytest", {}).get("ini_options", {})
        if "beartype_packages" in options and entry not in options.get("filterwarnings", []):
            problems.append(str(pyproject.relative_to(REPO_ROOT)))
    assert not problems, f"pytest configs without filterwarnings {entry!r}:\n" + "\n".join(problems)


def _module_level_imports(path: Path) -> list[str]:
    """Top-level names a module imports when it loads: outside functions, classes and ``if TYPE_CHECKING:``."""
    import ast

    names: list[str] = []
    pending: list[ast.stmt] = list(ast.parse(path.read_text()).body)
    while pending:
        node: ast.stmt = pending.pop()
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            names.append(node.module.split(".")[0])
        elif isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            pending.extend(node.orelse)
        elif isinstance(node, ast.If | ast.Try | ast.With):
            pending.extend([*node.body, *getattr(node, "orelse", []), *getattr(node, "finalbody", []), *[s for h in getattr(node, "handlers", []) for s in h.body]])
    return names


def test_conftest_never_imports_a_workspace_package_at_module_level() -> None:
    """pytest-beartype installs its import hook in ``pytest_configure``, after conftest.py loads: a package conftest
    imported first stays unchecked (beartype's hooks "silently reduce to a noop"). Imports go inside fixtures or helpers."""
    workspace_modules: set[str] = {
        init.parent.name
        for package_dir in RUNNABLE_PYTHON_PACKAGES
        for init in [*package_dir.glob("*/__init__.py"), *package_dir.glob("src/*/__init__.py")]
        if init.parent.name not in {"tests", "tools"}
    }
    problems: list[str] = []
    for conftest in sorted(REPO_ROOT.glob("packages/*/tests/conftest.py")):
        pending: list[Path] = [conftest]
        seen: set[Path] = set()
        while pending:
            module: Path = pending.pop()
            if module in seen:
                continue
            seen.add(module)
            for name in _module_level_imports(module):
                if name in workspace_modules:
                    problems.append(f"{module.relative_to(REPO_ROOT)} imports {name} (via {conftest.relative_to(REPO_ROOT)})")
                elif (conftest.parent / f"{name}.py").is_file():
                    pending.append(conftest.parent / f"{name}.py")
    assert not problems, "Module-level workspace imports in conftest.py (move them into the fixtures):\n" + "\n".join(problems)


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


@pytest.mark.parametrize(("name", "command"), [pytest.param(name, command, id=name) for name, command in _task_commands(MANIFEST).items()])
def test_task_commands_are_single_line(name: str, command: str | list[str]) -> None:
    """Pixi collapses newlines inside a multiline cmd into spaces (AGENTS.md gotcha), so a
    command split over lines without `&&` / `\\` silently becomes arguments to its first word.
    A trailing newline before the closing quotes is harmless and ignored here."""
    parts: list[str] = [command] if isinstance(command, str) else command
    assert not any("\n" in part.strip() or "\r" in part.strip() for part in parts), f"{name}: cmd contains an inner newline"
