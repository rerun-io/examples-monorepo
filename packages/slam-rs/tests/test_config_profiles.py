"""Config profiles preserve the reference contract without constructing Rust objects."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from slam_rs.reference import DatasetProperties, ReferenceManifest, load_manifest


def test_reference_profile_preserves_vendored_text() -> None:
    manifest: ReferenceManifest = load_manifest()
    for dataset in manifest.datasets:
        assert manifest.vio_config_text(dataset.name) == (manifest.package_root / dataset.vio_config).read_text()


def test_fast_profile_changes_only_lm_cap() -> None:
    manifest: ReferenceManifest = load_manifest()
    for dataset in manifest.datasets:
        reference: dict = json.loads((manifest.package_root / dataset.vio_config).read_text())
        fast: dict = json.loads(manifest.vio_config_text(dataset.name, profile="fast"))
        assert fast["value0"].pop("config.vio_max_iterations") == 4
        assert reference["value0"].pop("config.vio_max_iterations") == 7
        assert fast == reference


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
