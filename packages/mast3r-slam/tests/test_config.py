"""The SLAM presets, their CLI and their record in the recording."""

import dataclasses
from pathlib import Path

import pytest
import tyro

from mast3r_slam.config import SLAM_PRESETS, SlamConfig, SlamPresetName, flat_config, load_config, merge_config

PACKAGE_DIR: Path = Path(__file__).parent.parent


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
def test_preset_equals_its_yaml(name: SlamPresetName, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every preset holds the values, and the value types, of the merged YAML file it replaces."""
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


def _cli(args: list[str], tmp_path: Path):
    from mast3r_slam.api.inference import InferenceConfig

    return tyro.cli(InferenceConfig, args=[*args, "--rr-config.save", str(tmp_path / "run.rrd"), "--rr-config.headless"])


def test_cli_preset_with_an_override(tmp_path: Path) -> None:
    """A preset subcommand, a field override, and a root flag after the preset name."""
    settings = _cli(["fast", "--slam.tracking.max-iters", "30", "--dataset", "x.mp4"], tmp_path)
    fast: SlamConfig = SlamConfig.fast()
    assert settings.slam == dataclasses.replace(fast, tracking=dataclasses.replace(fast.tracking, max_iters=30))
    assert settings.dataset == "x.mp4"


def test_cli_defaults_to_base(tmp_path: Path) -> None:
    assert _cli([], tmp_path).slam == SlamConfig.base()
    assert _cli(["--slam.single-thread"], tmp_path).slam == dataclasses.replace(SlamConfig.base(), single_thread=True)
    # tyro 1.0.17 routes a bare `--slam.no-<flag>` to no preset; with the preset named it works.
    assert _cli(["test", "--slam.no-single-thread"], tmp_path).slam == SlamConfig.fast()


def test_cli_rejects_an_unknown_filtering_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _cli(["--slam.tracking.filtering-mode", "nearest"], tmp_path)
