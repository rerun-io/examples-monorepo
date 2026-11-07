"""Stand-ins for the `vipe_ext` functions vipe_mojo does not port (MetricTrail never calls them): they exist and raise
NotImplementedError naming the op when called."""

import types
from collections.abc import Callable
from typing import Any, NoReturn


def unported(namespace: str, name: str) -> Callable[..., NoReturn]:
    """A function that raises NotImplementedError naming `namespace.name`."""

    def raise_unported(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise NotImplementedError(f"vipe_mojo: {namespace}.{name} is not ported (MetricTrail never calls it; ViPE's vipe_ext is not built)")

    raise_unported.__name__ = name
    return raise_unported


def module_getattr(namespace: str) -> Callable[[str], Callable[..., NoReturn]]:
    """A PEP 562 module `__getattr__`: every public name not defined in the module is an unported function."""

    def __getattr__(name: str) -> Callable[..., NoReturn]:
        if name.startswith("_"):
            raise AttributeError(name)
        return unported(namespace, name)

    return __getattr__


def unported_namespace(namespace: str) -> types.ModuleType:
    """A module whose every public attribute is an unported function."""
    module: types.ModuleType = types.ModuleType(f"vipe_mojo.{namespace}", f"vipe_ext.{namespace}: not ported to Mojo.")
    module.__getattr__ = module_getattr(namespace)  # type: ignore[method-assign]
    return module
