"""The SLAM presets, their CLI and their record in the recording."""

import dataclasses

import pytest

from mast3r_slam.config import SLAM_PRESETS, SlamConfig, flat_config, load_config, merge_config


def _flat_yaml(name: str) -> dict:
    merged: dict = merge_config({}, load_config(f"config/{name}.yaml"))
    merged.pop("inherit", None)
    flat: dict = {}
    for key, value in merged.items():
        if isinstance(value, dict):
            flat.update({f"{key}.{leaf}": v for leaf, v in value.items()})
        else:
            flat[key] = value
    return flat


@pytest.mark.parametrize("name", list(SLAM_PRESETS))
def test_preset_equals_its_yaml(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every preset holds the values, and the value types, of the merged YAML file it replaces."""
    from conftest import PACKAGE_DIR

    monkeypatch.chdir(PACKAGE_DIR)
    want: dict = _flat_yaml(name)
    got: dict = flat_config(SLAM_PRESETS[name][1])
    assert got == want
    assert {k: type(v) for k, v in got.items()} == {k: type(v) for k, v in want.items()}


def test_presets_are_frozen_and_validated() -> None:
    cfg: SlamConfig = SlamConfig.fast()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.use_calib = True  # type: ignore[misc]
    with pytest.raises(ValueError, match="subsample"):
        cfg.with_subsample(0)
