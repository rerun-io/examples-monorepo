"""``_core.pyi`` is the only static check on the FFI boundary, so it must stay honest.

The checks run both ways and go past the names: nothing the extension exports may
be missing from the stub, nothing the stub declares may be missing from the
extension, and every signature the stub gives must be the one PyO3 built — same
parameters, same order, same keyword-only split, same defaults. PyO3 writes a
``__text_signature__`` for every method and constructor, which is what
:func:`inspect.signature` reads, so the runtime side is the extension's own
account of itself rather than a second transcription.

Dunder methods are compared as names and kinds only: a slot wrapper reports
CPython's own parameter names (``VioStatus.__eq__`` is ``(self, value, /)``,
whatever the stub calls its argument), so their parameters say nothing about this
extension. ``__init__`` is the exception and is compared, because PyO3 does build
that signature.
"""

import ast
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from slam_rs import _core

STUB_PATH: Path = Path(__file__).resolve().parents[1] / "slam_rs" / "_core.pyi"
STUB_TREE: ast.Module = ast.parse(STUB_PATH.read_text())
CLASS_NAMES: tuple[str, ...] = tuple(node.name for node in STUB_TREE.body if isinstance(node, ast.ClassDef))
"""The stub's own class list, so a class added to it is checked without editing this file."""


@dataclass(frozen=True, slots=True)
class Parameter:
    """One parameter of a signature, in the terms both sides can report."""

    name: str
    """Parameter name."""
    keyword_only: bool
    """Whether it sits after the ``*`` and can only be passed by keyword."""
    default: str | None
    """The default's source text, or None where there is none."""


def _stub_class(class_name: str) -> ast.ClassDef:
    """The stub's declaration of one class."""
    for node in STUB_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} is not declared in {STUB_PATH.name}")


def _stub_functions(class_name: str) -> dict[str, ast.FunctionDef]:
    """The stub's ``def``s on one class, by name."""
    return {item.name: item for item in _stub_class(class_name).body if isinstance(item, ast.FunctionDef)}


def _decorators(function: ast.FunctionDef) -> set[str]:
    """The bare decorator names on one stub ``def``."""
    return {decorator.id for decorator in function.decorator_list if isinstance(decorator, ast.Name)}


def stub_parameters(function: ast.FunctionDef) -> tuple[Parameter, ...]:
    """One stub ``def``'s parameters, with ``self`` dropped.

    Args:
        function: A ``def`` from the stub's AST.

    Returns:
        Its parameters in declaration order, positional ones before keyword-only.
    """
    positional: list[ast.arg] = [*function.args.posonlyargs, *function.args.args]
    # `defaults` covers the tail of the positional parameters.
    padding: list[ast.expr | None] = [None] * (len(positional) - len(function.args.defaults))
    parameters: list[Parameter] = [
        Parameter(name=argument.arg, keyword_only=False, default=None if default is None else ast.unparse(default))
        for argument, default in zip(positional, padding + list(function.args.defaults), strict=True)
        if argument.arg != "self"
    ]
    parameters.extend(
        Parameter(name=argument.arg, keyword_only=True, default=None if default is None else ast.unparse(default))
        for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults, strict=True)
    )
    return tuple(parameters)


def runtime_parameters(callable_object: object) -> tuple[Parameter, ...]:
    """One runtime callable's parameters, from the ``__text_signature__`` PyO3 wrote.

    Args:
        callable_object: A class, method descriptor or static method of ``_core``.

    Returns:
        Its parameters in declaration order, with ``self`` dropped.
    """
    # `getattr` on a class hands back an `object`; everything reaching here is a
    # class or a descriptor, both of which `inspect.signature` reads.
    signature: inspect.Signature = inspect.signature(cast("Callable[..., object]", callable_object))
    return tuple(
        Parameter(
            name=parameter.name,
            keyword_only=parameter.kind is inspect.Parameter.KEYWORD_ONLY,
            default=None if parameter.default is inspect.Parameter.empty else repr(parameter.default),
        )
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    )


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
    for item in _stub_class(class_name).body:
        if isinstance(item, ast.FunctionDef):
            members.add(item.name)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            members.add(item.target.id)
        elif isinstance(item, ast.Assign):
            members.update(target.id for target in item.targets if isinstance(target, ast.Name))
    return members


def test_the_stub_declares_exactly_the_extension_classes() -> None:
    """Every check below walks :data:`CLASS_NAMES`, so the two sides must agree on it."""
    exported: set[str] = {name for name in dir(_core) if isinstance(getattr(_core, name), type)}
    assert set(CLASS_NAMES) == exported


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


def test_every_method_signature_matches_the_stub() -> None:
    """Names, order, the keyword-only split and defaults, per method."""
    wrong: dict[str, tuple[tuple[Parameter, ...], tuple[Parameter, ...]]] = {}
    for class_name in CLASS_NAMES:
        runtime_class: type = getattr(_core, class_name)
        for name, function in _stub_functions(class_name).items():
            if name.startswith("__") or "property" in _decorators(function):
                continue
            declared: tuple[Parameter, ...] = stub_parameters(function)
            actual: tuple[Parameter, ...] = runtime_parameters(getattr(runtime_class, name))
            if declared != actual:
                wrong[f"{class_name}.{name}"] = (declared, actual)
    assert not wrong, f"_core.pyi disagrees with the extension: {wrong}"


def test_every_constructor_signature_matches_the_stub() -> None:
    """``__init__`` is the one dunder PyO3 builds a signature for."""
    wrong: dict[str, tuple[tuple[Parameter, ...], tuple[Parameter, ...]]] = {}
    for class_name in CLASS_NAMES:
        function: ast.FunctionDef | None = _stub_functions(class_name).get("__init__")
        if function is None:
            continue
        declared: tuple[Parameter, ...] = stub_parameters(function)
        actual: tuple[Parameter, ...] = runtime_parameters(getattr(_core, class_name))
        if declared != actual:
            wrong[class_name] = (declared, actual)
    assert not wrong, f"_core.pyi disagrees with the extension: {wrong}"


def test_a_class_the_stub_gives_no_constructor_cannot_be_constructed() -> None:
    """The stub declares no ``__init__`` for these, so calling them must fail."""
    for class_name in CLASS_NAMES:
        if "__init__" in _stub_functions(class_name):
            continue
        constructor: Callable[..., object] = getattr(_core, class_name)
        with pytest.raises(TypeError):
            constructor()


def test_the_stub_and_the_extension_agree_on_what_is_a_property() -> None:
    """A getter declared as a method — or the reverse — is a lie a caller trips over."""
    disagreements: list[str] = []
    for class_name in CLASS_NAMES:
        runtime_class: type = getattr(_core, class_name)
        for name, function in _stub_functions(class_name).items():
            if name.startswith("__"):
                continue
            member: object = vars(runtime_class).get(name)
            declared_property: bool = "property" in _decorators(function)
            is_property: bool = isinstance(member, property) or type(member).__name__ == "getset_descriptor"
            if declared_property != is_property:
                shape: str = "a property" if declared_property else "a method"
                disagreements.append(f"{class_name}.{name} is {shape} in the stub but {type(member).__name__} at runtime")
    assert not disagreements, disagreements


def test_a_static_method_is_static_on_both_sides() -> None:
    for class_name in CLASS_NAMES:
        for name, function in _stub_functions(class_name).items():
            declared_static: bool = "staticmethod" in _decorators(function)
            is_static: bool = isinstance(vars(getattr(_core, class_name)).get(name), staticmethod)
            assert declared_static == is_static, f"{class_name}.{name}: stub static={declared_static}, runtime static={is_static}"


def test_vio_status_behaves_as_the_stub_describes() -> None:
    """The stub calls VioStatus a plain PyO3 class; hold it to exactly that contract."""
    status: _core.VioStatus = _core.VioStatus.NeedMoreImu
    assert int(status) == 0
    assert int(_core.VioStatus.Tracking) == 1
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
