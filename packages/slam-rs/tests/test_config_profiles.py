"""Profile overlays use the Rust schema without changing resolved config bytes."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from slam_rs.config import DatasetProperties, SlamConfig, load_slam_config


def test_rust_defaulted_keys_are_valid_profile_overlays(tmp_path: Path) -> None:
    """A Rust schema field need not be repeated in each dataset config."""
    from slam_rs.config import profiled_config_text

    base: Path = tmp_path / "base.json"
    base.write_text('{"value0":{}}')
    (tmp_path / "custom.json").write_text('{"port.klt_exit_step_px":0.05}')
    assert json.loads(profiled_config_text(base, "custom", tmp_path)) == {"value0": {"port.klt_exit_step_px": 0.05}}


def test_profile_values_are_validated_by_rust(tmp_path: Path) -> None:
    from slam_rs.config import profiled_config_text

    base: Path = tmp_path / "base.json"
    base.write_text('{"value0":{}}')
    (tmp_path / "custom.json").write_text('{"port.frontend_lag":"yes"}')
    with pytest.raises(ValueError):
        profiled_config_text(base, "custom", tmp_path)


def test_reference_profile_preserves_vendored_text() -> None:
    settings: SlamConfig = load_slam_config()
    for dataset in settings.datasets:
        assert settings.vio_config_text(dataset.name) == (settings.package_root / dataset.vio_config).read_text()


def test_fast_profile_changes_only_its_declared_keys() -> None:
    """The fast profile changes only its four declared keys.

    The port-prefixed keys select demand-based detection (D75) and the
    keyframe-gated joint solve (D76), and KLT exit step. Only the overlay supplies them.
    """
    settings: SlamConfig = load_slam_config()
    for dataset in settings.datasets:
        reference: dict = json.loads((settings.package_root / dataset.vio_config).read_text())
        fast: dict = json.loads(settings.vio_config_text(dataset.name, profile="fast"))
        assert fast["value0"].pop("config.vio_max_iterations") == 7
        assert reference["value0"].pop("config.vio_max_iterations") == 7
        assert fast["value0"].pop("port.redetect_survivor_ratio") == 0.85
        assert fast["value0"].pop("port.frame_update_max_iterations") == 5
        assert fast["value0"].pop("port.klt_exit_step_px") == 0.05
        assert all(key.startswith("config.") for key in reference["value0"])
        assert fast == reference


def test_no_vendored_config_carries_a_port_key() -> None:
    """Port-only knobs are supplied by overlays, leaving dataset configurations unchanged."""
    settings: SlamConfig = load_slam_config()
    for dataset in settings.datasets:
        values: dict = json.loads((settings.package_root / dataset.vio_config).read_text())["value0"]
        assert all(key.startswith("config.") for key in values), dataset.name


@pytest.mark.parametrize(
    ("overlay", "rejected"),
    [
        pytest.param('{"port.redetect_survivor_ration": 0.5}', "port.redetect_survivor_ration", id="invented-port-key"),
        pytest.param('{"config.misspelled_iterations": 4}', "config.misspelled_iterations", id="unknown-config-key"),
    ],
)
def test_an_unknown_overlay_key_is_rejected(tmp_path: Path, overlay: str, rejected: str) -> None:
    """Neither namespace is an escape hatch for a typo.

    Both namespaces belong to the Rust schema: a
    misspelled knob names itself instead of silently doing nothing.
    """
    settings: SlamConfig = load_slam_config()
    dataset: DatasetProperties = settings.datasets[0]
    config_path: Path = tmp_path / dataset.vio_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text((settings.package_root / dataset.vio_config).read_text())
    profiles: Path = tmp_path / "configs/profiles"
    profiles.mkdir()
    (profiles / "fast.json").write_text(overlay)
    with pytest.raises(ValueError, match=rejected):
        replace(settings, package_root=tmp_path).vio_config_text(dataset.name, profile="fast")
