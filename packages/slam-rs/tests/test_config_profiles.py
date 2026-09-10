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


def test_fast_profile_changes_only_the_lm_cap_and_the_two_port_gates() -> None:
    """The three keys the fast profile owns, and nothing else moves.

    Neither ``port.`` key is one of basalt's: they are the port's own
    redetect-on-demand gate (D75) and its keyframe-gated joint solve (D76), so
    both are absent from every vendored file and the overlay is the only thing
    that ever sets them.
    """
    manifest: ReferenceManifest = load_manifest()
    for dataset in manifest.datasets:
        reference: dict = json.loads((manifest.package_root / dataset.vio_config).read_text())
        fast: dict = json.loads(manifest.vio_config_text(dataset.name, profile="fast"))
        assert fast["value0"].pop("config.vio_max_iterations") == 7
        assert reference["value0"].pop("config.vio_max_iterations") == 7
        assert fast["value0"].pop("port.redetect_survivor_ratio") == 0.85
        assert fast["value0"].pop("port.frame_update_max_iterations") == 5
        assert not (set(reference["value0"]) & PORT_CONFIG_KEYS)
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


@pytest.mark.parametrize(
    ("overlay", "rejected"),
    [
        pytest.param('{"port.redetect_survivor_ration": 0.5}', "port.redetect_survivor_ration", id="invented-port-key"),
        pytest.param('{"config.misspelled_iterations": 4}', "config.misspelled_iterations", id="unknown-config-key"),
    ],
)
def test_an_unknown_overlay_key_is_rejected(tmp_path: Path, overlay: str, rejected: str) -> None:
    """Neither namespace is an escape hatch for a typo.

    ``port.`` is an allowlist rather than an open namespace (D75), and a
    ``config.`` key the Rust schema does not carry is refused the same way: a
    misspelled knob names itself instead of silently doing nothing.
    """
    manifest: ReferenceManifest = load_manifest()
    dataset: DatasetProperties = manifest.datasets[0]
    config_path: Path = tmp_path / dataset.vio_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text((manifest.package_root / dataset.vio_config).read_text())
    profiles: Path = tmp_path / "configs/profiles"
    profiles.mkdir()
    (profiles / "fast.json").write_text(overlay)
    with pytest.raises(KeyError, match=rejected):
        replace(manifest, package_root=tmp_path).vio_config_text(dataset.name, profile="fast")
