"""The public API surface of lietorch (or lietorch_mojo) as plain JSON data.

`test_api_snapshot.py` compares lietorch_mojo against `fixtures/lietorch_api.json`,
a snapshot of upstream lietorch (VSLAM-LAB/lietorch @ 3e2695dd, the ai-demos
`lietorch 1.0` build 2) taken by this file. The test needs no lietorch.

Regenerate the snapshot (in an environment with the compiled lietorch, e.g.
`mast3r-slam-dev`), from `packages/lietorch-mojo`:

    python tests/api_surface.py

What the surface holds, per module (`lietorch`, `.groups`, `.group_ops`,
`.broadcasting`, `.gradcheck`) and for `lietorch_backends`:
- every public name the package itself defines or re-exports (imports from
  torch, numpy or typing are not part of the API and are left out);
- for functions: the signature (parameter names, kinds and default reprs);
- for classes: every public member defined in the package's own classes
  (inherited torch members are left out), with its kind (function,
  classmethod, staticmethod, property, constant) and signature, plus the
  values of int/str/tensor class constants (`group_id`, `manifold_dim`,
  `embedded_dim`, `id_elem`, ...) and the backend function each autograd op
  calls (`forward_op`, `backward_op`);
- for `lietorch_backends` (pybind11): each function's argument count.
"""

import importlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

import torch

FIXTURE: Path = Path(__file__).parent / "fixtures" / "lietorch_api.json"
SUBMODULES: tuple[str, ...] = ("", ".groups", ".group_ops", ".broadcasting", ".gradcheck")
# Dunder members that are part of the API (operators, constructors, repr).
API_DUNDERS: frozenset[str] = frozenset({"__init__", "__new__", "__mul__", "__getitem__", "__setitem__", "__repr__"})

Json = dict[str, object]


def signature_of(fn: object) -> list[list[object]]:
    """[[name, kind, default repr or None], ...] of `fn` (annotations are ignored: upstream has none)."""
    params = inspect.signature(fn).parameters.values()  # type: ignore[arg-type]
    return [[p.name, p.kind.name, None if p.default is inspect.Parameter.empty else repr(p.default)] for p in params]


def _member(attr: object, name: str) -> Json:
    if isinstance(attr, staticmethod | classmethod):
        return {"kind": type(attr).__name__, "signature": signature_of(attr.__func__)}
    if isinstance(attr, property):
        return {"kind": "property"}
    if name in ("forward_op", "backward_op"):
        # The backend function an autograd op wraps (a pybind11 builtin upstream, a Python function here).
        return {"kind": "backend_op", "name": None if attr is None else attr.__name__}  # type: ignore[union-attr]
    if inspect.isfunction(attr):
        return {"kind": "function", "signature": signature_of(attr)}
    if isinstance(attr, torch.Tensor):
        return {"kind": "constant", "value": {"dtype": str(attr.dtype), "values": attr.tolist()}}
    if isinstance(attr, int | str | float | bool):
        return {"kind": "constant", "value": attr}
    return {"kind": type(attr).__name__}


def class_surface(cls: type, root: str) -> Json:
    """Public members defined in `cls` and its bases from the `root` package (most-derived definition wins)."""
    members: Json = {}
    for klass in cls.__mro__:
        if not klass.__module__.startswith(root):
            continue
        for name, attr in vars(klass).items():
            if name in members or (name.startswith("_") and name not in API_DUNDERS):
                continue
            members[name] = _member(attr, name)
    return {"kind": "class", "bases": [b.__name__ for b in cls.__bases__], "members": dict(sorted(members.items()))}


def module_surface(mod: ModuleType, root: str) -> Json:
    names: Json = {}
    for name, obj in vars(mod).items():
        if name.startswith("_"):
            continue
        if isinstance(obj, ModuleType):
            if obj.__name__.startswith(root + "."):
                names[name] = {"kind": "module", "name": obj.__name__.removeprefix(root)}
            continue
        origin: str = getattr(obj, "__module__", None) or ""
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if not origin.startswith(root):
                continue  # imported from torch, numpy, typing, ...
            if origin != mod.__name__:
                names[name] = {"kind": "reexport", "of": f"{origin.removeprefix(root)}.{obj.__name__}"}
            elif inspect.isclass(obj):
                names[name] = class_surface(obj, root)
            else:
                names[name] = {"kind": "function", "signature": signature_of(obj)}
        elif isinstance(obj, int | str | float) and not isinstance(obj, bool):
            names[name] = {"kind": "constant", "type": type(obj).__name__}
    return dict(sorted(names.items()))


def api_surface(root: str) -> Json:
    """The surface of package `root` (`lietorch` or `lietorch_mojo`), keyed by submodule (`.` is the package itself)."""
    surface: Json = {}
    for suffix in SUBMODULES:
        surface[suffix or "."] = module_surface(importlib.import_module(root + suffix), root)
    return surface


def lietorch_backends_arity() -> dict[str, int]:
    """Argument count of each lietorch_backends function, from the pybind11 signature line `name(arg0: int, ...) -> ...`."""
    import lietorch_backends

    arity: dict[str, int] = {}
    for name in sorted(n for n in dir(lietorch_backends) if not n.startswith("_")):
        line: str = getattr(lietorch_backends, name).__doc__.splitlines()[0]
        arity[name] = line.split("(", 1)[1].split(")")[0].count("arg")
    return arity


def snapshot() -> Json:
    return {
        "source": "lietorch (VSLAM-LAB/lietorch @ 3e2695dd, ai-demos lietorch 1.0 build 2)",
        "regenerate": "python tests/api_surface.py  (from packages/lietorch-mojo, in an env with the compiled lietorch)",
        "torch": torch.__version__,
        "modules": api_surface("lietorch"),
        "backends": lietorch_backends_arity(),
    }


if __name__ == "__main__":
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(snapshot(), indent=1) + "\n")
    print(f"wrote {FIXTURE}", file=sys.stderr)
