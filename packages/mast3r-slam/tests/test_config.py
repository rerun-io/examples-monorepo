"""The SLAM presets, their CLI and their record in the recording."""

import dataclasses
from pathlib import Path

import pytest
import tyro

from mast3r_slam.config import SLAM_PRESETS, SlamConfig, flat_config


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


def test_flat_config_names_every_field_once() -> None:
    flat: dict[str, bool | int | float | str] = flat_config(SLAM_PRESETS["eth3d"][1])
    assert flat["reloc.strict"] is False
    assert flat["dataset.center_principle_point"] is False
    assert flat["use_calib"] is True
    assert all(isinstance(v, bool | int | float | str) for v in flat.values())
