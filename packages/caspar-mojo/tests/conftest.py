"""Integration prerequisites in one place: the built binaries and a GPU.

A test skips only when a binary is absent (with the task that builds it) or when the extension reports no GPU.
Once both hold, an import, link, launch or solver error fails the test. Workspace modules are imported inside the
fixtures, after pytest-beartype has installed its hook.
"""

import ctypes
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE = Path(__file__).parents[1] / "caspar_mojo"
EXTENSION = PACKAGE / "_caspar_mojo.so"
LIBRARY = PACKAGE / "libcaspar_mojo.so"


@pytest.fixture(scope="session")
def native() -> ModuleType:
    """The built Python extension, on a host where it sees a GPU."""
    if not EXTENSION.exists():
        pytest.skip(f"{EXTENSION.name} is not built: pixi run -e caspar-mojo-dev caspar-mojo-build")
    from caspar_mojo import _caspar_mojo

    if _caspar_mojo.device_count() == 0:
        pytest.skip("the Mojo runtime sees no GPU")
    return _caspar_mojo


@pytest.fixture(scope="session")
def c_library(native: ModuleType) -> ctypes.CDLL:
    """libcaspar_mojo.so, loaded as a C caller loads it."""
    if not LIBRARY.exists():
        pytest.skip(f"{LIBRARY.name} is not built: pixi run -e caspar-mojo-dev caspar-mojo-build-c")
    return ctypes.CDLL(str(LIBRARY))
