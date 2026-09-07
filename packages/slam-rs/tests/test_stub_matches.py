"""``_core.pyi`` is the only static check on the FFI boundary, so it must stay complete."""

import ast
from pathlib import Path

from slam_rs import _core

STUB_PATH: Path = Path(__file__).resolve().parents[1] / "slam_rs" / "_core.pyi"


def _stub_module_names() -> set[str]:
    """Top-level class, function and annotated-assignment names declared in the stub."""
    tree: ast.Module = ast.parse(STUB_PATH.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _stub_class_members(class_name: str) -> set[str]:
    """Method and property names declared on one stub class."""
    tree: ast.Module = ast.parse(STUB_PATH.read_text())
    members: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
                elif isinstance(item, ast.Assign):
                    members.update(target.id for target in item.targets if isinstance(target, ast.Name))
    return members


def test_every_public_module_name_is_declared() -> None:
    exported: set[str] = {name for name in dir(_core) if not name.startswith("_")}
    assert exported <= _stub_module_names(), f"undeclared in _core.pyi: {sorted(exported - _stub_module_names())}"


def test_the_version_attribute_is_declared() -> None:
    assert "__version__" in _stub_module_names()


def test_every_public_class_member_is_declared() -> None:
    missing: dict[str, list[str]] = {}
    for class_name in ("Vio", "VioResult", "VioStatus"):
        declared: set[str] = _stub_class_members(class_name)
        actual: set[str] = {name for name in vars(getattr(_core, class_name)) if not name.startswith("_")}
        if actual - declared:
            missing[class_name] = sorted(actual - declared)
    assert not missing, f"undeclared in _core.pyi: {missing}"
