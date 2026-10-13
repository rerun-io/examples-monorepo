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


def module_name(path: Path) -> str:
    """Dotted module of a workspace file (`.../exo_calib/ops/x.py` -> `exo_calib.ops.x`)."""
    for name, root in ROOTS.items():
        if path.resolve().is_relative_to(root / name):
            parts: tuple[str, ...] = path.resolve().relative_to(root).with_suffix("").parts
            return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    raise ValueError(f"{path} is in no [roots] package")


def imports(path: Path, module: str | None = None) -> list[tuple[str, bool]]:
    """(absolute imported module, inside a function body) for every import, relative ones resolved.

    `from pkg import name` yields `pkg.name` when that is a module, else `pkg`.
    """
    module = module or module_name(path)
    package: list[str] = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    tree: ast.Module = ast.parse(path.read_text())
    lazy: set[int] = {id(n) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef) for n in ast.walk(f)}
    found: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, id(node) in lazy) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base: str = ".".join(package[: len(package) - node.level + 1] + [node.module or ""]).strip(".") if node.level else node.module or ""
            found += [(f"{base}.{a.name}" if module_file(f"{base}.{a.name}") else base, id(node) in lazy) for a in node.names]
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


def import_problems(path: Path, source: str, module: str | None = None) -> list[str]:
    """Imports of one file (layer `source`) that point up, or into an enforced module no rule classifies."""
    problems: list[str] = []
    for target_module, _ in imports(path, module):
        target_file: Path | None = module_file(target_module)
        root_init: bool = target_file is not None and target_file.parent.name == target_module.split(".")[0] and target_file.name == "__init__.py"
        if target_module.split(".")[0] == "dagkit":
            allowed: list[str] = CONFIG["allowed"][source]
            ok: bool = "dagkit" in allowed or any(target_module.startswith(c) for c in allowed if c.startswith("dagkit."))
        elif target_file is not None and (target := layer_of(target_file)) is not None:
            ok = target in CONFIG["allowed"][source]
        else:  # third-party, the package root (checked to hold only the beartype hook), or a module outside the layer homes
            ok = target_file is None or root_init or target_module.split(".")[0] not in CONFIG["enforced"]
        if not ok:
            problems.append(f"{path} ({source}) imports {target_module}")
    return problems


def test_relative_and_from_imports_resolve_to_modules(tmp_path: Path) -> None:
    probe: Path = tmp_path / "probe.py"
    probe.write_text("from ..apis import refine\nfrom exo_calib import apis\nfrom . import cameras\n")
    found: list[str] = [module for module, _ in imports(probe, "exo_calib.ops.probe")]
    assert found == ["exo_calib.apis.refine", "exo_calib.apis", "exo_calib.ops"]
    assert import_problems(probe, "Op", "exo_calib.ops.probe") == [f"{probe} (Op) imports exo_calib.apis.refine", f"{probe} (Op) imports exo_calib.apis"]


def root_init_problems(path: Path, package: str) -> list[str]:
    """The package root holds only the beartype hook (stdlib + beartype): a re-export there would let any layer import any other."""
    return [
        f"{path}: the package root imports {module} (only the beartype hook belongs there)"
        for module, _ in imports(path, package)
        if module.split(".")[0] not in sys.stdlib_module_names and module != "beartype.claw"
    ]


def test_the_package_root_may_hold_only_the_beartype_hook(tmp_path: Path) -> None:
    init: Path = tmp_path / "__init__.py"
    init.write_text("import os\nimport sys\nfrom beartype.claw import beartype_this_package\nfrom exo_calib.apis import refine\nimport torch\n")
    found: list[str] = root_init_problems(init, "exo_calib")
    assert found == [f"{init}: the package root imports {m} (only the beartype hook belongs there)" for m in ("exo_calib.apis.refine", "torch")]


@pytest.mark.parametrize("package", CONFIG["enforced"])
def test_enforced_package_imports_point_down(package: str) -> None:
    problems: list[str] = []
    for path in sorted((ROOTS[package] / package).rglob("*.py")) + sorted((ROOTS[package] / "tools").rglob("*.py")):
        if path.parent == ROOTS[package] / package and path.name == "__init__.py":
            problems += root_init_problems(path, package)
            continue
        source: str | None = layer_of(path)
        if source is None:
            problems.append(f"{path.relative_to(REPO_ROOT)}: matches no layer rule")
        else:
            problems += import_problems(path, source, module_name(path) if path.is_relative_to(ROOTS[package] / package) else "tools.x")
    assert not problems, "imports that point up:\n" + "\n".join(problems)
