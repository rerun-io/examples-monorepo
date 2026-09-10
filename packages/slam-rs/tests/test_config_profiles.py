"""Config profiles preserve the reference contract without constructing Rust objects."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from slam_rs.reference import PORT_CONFIG_KEYS, DatasetProperties, ReferenceManifest, load_manifest


def test_reference_profile_preserves_vendored_text() -> None:
    manifest: ReferenceManifest = load_manifest()
    for dataset in manifest.datasets:
        assert manifest.vio_config_text(dataset.name) == (manifest.package_root / dataset.vio_config).read_text()


def test_fast_profile_changes_only_the_lm_cap_and_the_redetect_gate() -> None:
    """The two keys the fast profile owns, and nothing else moves.

    ``port.redetect_survivor_ratio`` is not one of basalt's: it is the port's own
    redetect-on-demand gate (D75), so it is absent from every vendored file and
    the overlay is the only thing that ever sets it.
    """
    manifest: ReferenceManifest = load_manifest()
    for dataset in manifest.datasets:
        reference: dict = json.loads((manifest.package_root / dataset.vio_config).read_text())
        fast: dict = json.loads(manifest.vio_config_text(dataset.name, profile="fast"))
        assert fast["value0"].pop("config.vio_max_iterations") == 4
        assert reference["value0"].pop("config.vio_max_iterations") == 7
        assert fast["value0"].pop("port.redetect_survivor_ratio") == 0.85
        assert "port.redetect_survivor_ratio" not in reference["value0"]
        assert fast == reference


def test_no_vendored_config_carries_a_port_key() -> None:
    """The vendored files stay the documents the C++ runs read, key for key.

    ``tests/test_cpp_reference.py`` compares them against every run manifest, so
    a port-only knob is inserted by the overlay and defaulted by
    :class:`slam_rs._core.VioConfig` instead of being written into them.
    """
    manifest: ReferenceManifest = load_manifest()
    for dataset in manifest.datasets:
        values: dict = json.loads((manifest.package_root / dataset.vio_config).read_text())["value0"]
        assert not (set(values) & PORT_CONFIG_KEYS), dataset.name
        assert all(key.startswith("config.") for key in values), dataset.name


def test_an_invented_port_key_is_rejected(tmp_path: Path) -> None:
    """The ``port.`` namespace is an allowlist, not an escape hatch for typos."""
    manifest: ReferenceManifest = load_manifest()
    dataset: DatasetProperties = manifest.datasets[0]
    config_path: Path = tmp_path / dataset.vio_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text((manifest.package_root / dataset.vio_config).read_text())
    profiles: Path = tmp_path / "configs/profiles"
    profiles.mkdir()
    (profiles / "fast.json").write_text('{"port.redetect_survivor_ration": 0.5}')
    with pytest.raises(KeyError, match="port.redetect_survivor_ration"):
        replace(manifest, package_root=tmp_path).vio_config_text(dataset.name, profile="fast")


def test_unknown_overlay_key_is_rejected(tmp_path: Path) -> None:
    manifest: ReferenceManifest = load_manifest()
    dataset: DatasetProperties = manifest.datasets[0]
    config_path: Path = tmp_path / dataset.vio_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text((manifest.package_root / dataset.vio_config).read_text())
    profiles: Path = tmp_path / "configs/profiles"
    profiles.mkdir()
    (profiles / "fast.json").write_text('{"config.misspelled_iterations": 4}')
    with pytest.raises(KeyError, match="config.misspelled_iterations"):
        replace(manifest, package_root=tmp_path).vio_config_text(dataset.name, profile="fast")
