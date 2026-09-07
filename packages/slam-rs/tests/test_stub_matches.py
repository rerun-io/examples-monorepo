"""``_core.pyi`` is the only static check on the FFI boundary, so it must stay honest.

The checks run both ways: nothing the extension exports may be missing from the
stub, and nothing the stub declares may be missing from the extension.
"""

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from slam_rs import _core

STUB_PATH: Path = Path(__file__).resolve().parents[1] / "slam_rs" / "_core.pyi"
STUB_TREE: ast.Module = ast.parse(STUB_PATH.read_text())
CLASS_NAMES: tuple[str, ...] = ("Vio", "VioResult", "VioStatus", "VioConfig", "Calibration", "OpticalFlow", "FlowFrame")


def _stub_module_names() -> set[str]:
    """Top-level class, function and annotated-assignment names declared in the stub."""
    names: set[str] = set()
    for node in STUB_TREE.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _stub_class_members(class_name: str) -> set[str]:
    """Method, property and attribute names declared on one stub class."""
    members: set[str] = set()
    for node in STUB_TREE.body:
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


def test_every_declared_module_name_exists_at_runtime() -> None:
    missing: list[str] = [name for name in sorted(_stub_module_names()) if not hasattr(_core, name)]
    assert not missing, f"declared in _core.pyi but absent from the extension: {missing}"


def test_every_public_class_member_is_declared() -> None:
    missing: dict[str, list[str]] = {}
    for class_name in CLASS_NAMES:
        declared: set[str] = _stub_class_members(class_name)
        actual: set[str] = {name for name in vars(getattr(_core, class_name)) if not name.startswith("_")}
        if actual - declared:
            missing[class_name] = sorted(actual - declared)
    assert not missing, f"undeclared in _core.pyi: {missing}"


def test_every_declared_class_member_exists_at_runtime() -> None:
    missing: dict[str, list[str]] = {}
    for class_name in CLASS_NAMES:
        runtime_class: type = getattr(_core, class_name)
        absent: list[str] = [name for name in sorted(_stub_class_members(class_name)) if not hasattr(runtime_class, name)]
        if absent:
            missing[class_name] = absent
    assert not missing, f"declared in _core.pyi but absent from the extension: {missing}"


def test_vio_status_behaves_as_the_stub_describes() -> None:
    """The stub calls VioStatus a plain PyO3 class; hold it to exactly that contract."""
    status: _core.VioStatus = _core.VioStatus.NeedMoreImu
    assert int(status) == 1
    assert status == _core.VioStatus.NeedMoreImu
    assert status != _core.VioStatus.Tracking
    assert repr(status) == "VioStatus.NeedMoreImu"
    # Not an enum.Enum: no name/value, unhashable, not constructible.
    assert not hasattr(status, "name")
    assert not hasattr(status, "value")
    with pytest.raises(TypeError):
        hash(status)
    # Routed through a Callable: the stub declares no constructor arguments, so a
    # direct `VioStatus(1)` is a static error — which is exactly the promise here.
    constructor: Callable[..., object] = _core.VioStatus
    with pytest.raises(TypeError):
        constructor(1)
