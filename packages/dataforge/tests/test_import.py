"""Skeleton gate: the package imports and beartype activates under dev mode."""

import importlib
import os
import sys


def test_package_imports_with_beartype_activation(monkeypatch) -> None:
    monkeypatch.setenv("PIXI_DEV_MODE", "1")
    sys.modules.pop("dataforge", None)
    module = importlib.import_module("dataforge")
    assert module.__name__ == "dataforge"
    assert os.environ["PIXI_DEV_MODE"] == "1"
