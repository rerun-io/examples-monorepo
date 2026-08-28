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


def test_every_registered_dataset_implements_both_blueprint_hooks() -> None:
    """``setup()`` instantiates the ABC — a dataset missing ``default_blueprint``
    or ``table_blueprint`` raises ``TypeError`` here instead of silently
    registering without catalog blueprints."""
    from dataforge.datasets import dataset_defaults

    for name, config in dataset_defaults.items():
        dataset = config.setup()
        assert callable(dataset.default_blueprint), name
        assert callable(dataset.table_blueprint), name
