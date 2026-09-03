"""Every ``tools/apps`` shim must import: they are thin wiring over ``exo_calib.apis`` and nothing else exercises them."""

import importlib.util
from pathlib import Path

import pytest

SHIMS: list[Path] = sorted((Path(__file__).resolve().parents[1] / "tools" / "apps").glob("*.py"))


@pytest.mark.parametrize("shim", SHIMS, ids=[p.stem for p in SHIMS])
def test_tool_shim_imports(shim: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"exo_calib_shim_{shim.stem}", shim)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # ``__name__`` is not ``__main__``, so nothing runs
    assert callable(module.main)
