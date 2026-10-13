"""Layer-import check (SPEC §1.3): imports only point down, and dagkit stays a torch-free leaf.

Rules and the allowed matrix live in ci/layers.toml. Stdlib `ast` only, so it runs in the small `ci` environment.
"""

import ast
import fnmatch
import sys
from pathlib import Path

import pytest
import tomllib

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG: dict = tomllib.loads((REPO_ROOT / "ci" / "layers.toml").read_text())
ROOTS: dict[str, Path] = {name: REPO_ROOT / root for name, root in CONFIG["roots"].items()}


def layer_of(path: Path) -> str | None:
    """First matching rule for a file under packages/, or None."""
    relative: str = path.resolve().relative_to(REPO_ROOT / "packages").as_posix()
    return next((rule["layer"] for rule in CONFIG["rules"] if fnmatch.fnmatch(relative, rule["glob"])), None)


def module_file(module: str) -> Path | None:
    """Source file of a workspace module (`pkg.a.b` -> .../pkg/a/b.py or .../pkg/a/b/__init__.py)."""
    root: Path | None = ROOTS.get(module.split(".")[0])
    if root is None:
        return None
    base: Path = root.joinpath(*module.split("."))
    return next((p for p in (base.with_suffix(".py"), base / "__init__.py") if p.exists()), None)


def imports(path: Path) -> list[tuple[str, bool]]:
    """(imported module, inside a function body) for every absolute import in a file."""
    tree: ast.Module = ast.parse(path.read_text())
    lazy: set[int] = {id(n) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef) for n in ast.walk(f)}
    found: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, id(node) in lazy) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module, id(node) in lazy))
    return found


def test_rules_classify_the_layer_homes() -> None:
    packages: Path = REPO_ROOT / "packages"
    assert layer_of(packages / "exo-calib/tools/apps/calibrate.py") == "App"
    assert layer_of(packages / "mamma/src/mamma/graphs/capture.py") == "Graph"
    assert layer_of(packages / "simplecv/simplecv/ops/triangulate.py") == "Op"
    assert layer_of(packages / "trtkit/src/trtkit/runtime.py") == "Runtime"
    assert layer_of(packages / "robocap-live/robocap_live/types.py") == "Types"
    assert layer_of(packages / "exo-calib/exo_calib/catalog_io.py") is None


def test_dagkit_imports_only_its_allowlist() -> None:
    allowed: set[str] = set(CONFIG["dagkit"]["allowed"]) | set(sys.stdlib_module_names) | {"dagkit"}
    problems: list[str] = []
    for path in sorted((ROOTS["dagkit"] / "dagkit").rglob("*.py")):
        for module, in_function in imports(path):
            top: str = module.split(".")[0]
            lazy_ok: bool = in_function and top in CONFIG["dagkit"]["lazy"] and path.name == "app.py"
            if top not in allowed and not lazy_ok:
                problems.append(f"{path.relative_to(REPO_ROOT)}: imports {module}")
    assert not problems, "dagkit must stay a torch-free leaf:\n" + "\n".join(problems)


@pytest.mark.parametrize("package", CONFIG["enforced"])
def test_enforced_package_imports_point_down(package: str) -> None:
    problems: list[str] = []
    for path in sorted((ROOTS[package] / package).rglob("*.py")) + sorted((ROOTS[package] / "tools").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        source: str | None = layer_of(path)
        if source is None:
            problems.append(f"{path.relative_to(REPO_ROOT)}: matches no layer rule")
            continue
        for module, _ in imports(path):
            if module.split(".")[0] == "dagkit":
                ok = "dagkit" in CONFIG["allowed"][source] or any(module.startswith(c) for c in CONFIG["allowed"][source] if c.startswith("dagkit."))
            elif (target_file := module_file(module)) is not None and (target := layer_of(target_file)) is not None:
                ok = target in CONFIG["allowed"][source]
            else:
                continue  # third-party, or a workspace module outside the layer homes
            if not ok:
                problems.append(f"{path.relative_to(REPO_ROOT)} ({source}) imports {module}")
    assert not problems, "imports that point up:\n" + "\n".join(problems)
