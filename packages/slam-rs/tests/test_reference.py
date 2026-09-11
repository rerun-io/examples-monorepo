"""The gate parses and preserves tier, baseline and sensor-model rules."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from slam_rs import _core, reference
from slam_rs.reference import (
    DECODE_PATH_BY_NAME,
    MANIFEST_PATH,
    SMOKE_SEGMENTS,
    TIER_BY_NAME,
    ReferenceManifest,
    ReferenceSegment,
    load_manifest,
    resolved_flow_config,
)


def test_the_manifest_holds_ten_segments(manifest: ReferenceManifest) -> None:
    assert len(manifest.segments) == 10
    assert manifest.schema_version == 10


def test_segment_ids_are_unique(manifest: ReferenceManifest) -> None:
    identifiers: list[str] = [segment.segment_id for segment in manifest.segments]
    assert len(set(identifiers)) == len(identifiers)


def test_every_tier_and_decode_path_is_known(manifest: ReferenceManifest) -> None:
    for segment in manifest.segments:
        assert segment.tier in TIER_BY_NAME
        assert segment.decode_path in DECODE_PATH_BY_NAME
    assert {segment.tier for segment in manifest.segments} == set(TIER_BY_NAME)


def test_the_tiers_are_the_ones_the_plan_names(manifest: ReferenceManifest) -> None:
    assert len(manifest.in_tier("smoke")) == 2
    assert len(manifest.in_tier("release")) == 3
    assert len(manifest.in_tier("listed")) == 5
    assert {segment.dataset_name for segment in manifest.in_tier("smoke")} == {"msd-index", "msd-g2"}


def test_the_msd_imu_block_is_basalts(manifest: ReferenceManifest) -> None:
    for segment in manifest.segments:
        assert manifest.dataset(segment.dataset_name).imu.rate_hz == 1000.0
        assert manifest.dataset(segment.dataset_name).imu.gyro_noise_std == 0.000282
        assert manifest.dataset(segment.dataset_name).imu.accel_noise_std == 0.016
        assert manifest.dataset(segment.dataset_name).imu.gyro_bias_std == 0.0001
        assert manifest.dataset(segment.dataset_name).imu.accel_bias_std == 0.001
        assert manifest.dataset(segment.dataset_name).imu.cam_time_offset_ns == 0


def test_any_robocap_session_of_the_device_replays_without_a_reference(manifest: ReferenceManifest) -> None:
    listed = manifest.robocap.session("s00000015")
    assert listed.reference_csv is not None and manifest.robocap.is_listed("s00000015")
    other = manifest.robocap.session("s00000099")
    assert other.segment_id == f"robocap__{manifest.robocap.device_id}__s00000099"
    assert other.reference_csv is None and not manifest.robocap.is_listed("s00000099")


def test_robocap_carries_s15_and_no_ground_truth(manifest: ReferenceManifest) -> None:
    assert [session.session_id for session in manifest.robocap.sessions] == ["s00000015"]
    assert manifest.robocap.has_ground_truth is False
    assert manifest.robocap.imu.cam_time_offset_ns == 14_902_432
    assert manifest.robocap.imu.rate_hz == 200.0


def test_robocap_names_the_configuration_the_cpp_ran(manifest: ReferenceManifest) -> None:
    """The four cameras, the downscale, the two rig rules and the two configured JSON files, all present."""
    assert manifest.robocap.camera_names == ("left", "left_front", "right_front", "right")
    assert manifest.robocap.downscale == 3
    assert manifest.robocap.decode_path == "cpu_gray8_swscale_area_downscale3"
    # The manifest defines frameset tolerance and inertial pairing.
    # because the two inertial channels are on their own clocks.
    assert manifest.robocap.frameset_tolerance_ns == 1_000_000
    assert manifest.robocap.interpolate_accel_onto_gyro is True
    # The recorder stamps the device clock on the video timeline, so an export
    # adds nothing; MSD's `video_time` is relative to its capture start.
    assert manifest.robocap.video_time_is_absolute is True
    assert "config.vio_marg_lost_landmarks" in (manifest.package_root / manifest.robocap.vio_config).read_text()
    assert '"camera_type": "kb4"' in (manifest.package_root / manifest.robocap.calibration).read_text()


def test_a_selector_the_manifest_cannot_satisfy_is_a_typed_error(manifest: ReferenceManifest) -> None:
    """Every accessor that resolves a name names the ones there are, as a ``ValueError``.

    These three are what a tool resolves a command line through, and a
    ``KeyError`` reads as a dictionary miss: ``fleet_check --segments <clip>
    typo`` reached one after replaying the valid clip. The loader has always
    promised ``ValueError`` for a manifest it cannot read; a selector it cannot
    satisfy is the same kind of answer.
    """
    with pytest.raises(ValueError, match="MIO10_typo.*MIO10_short_2_panorama"):
        manifest.by_id("MIO10_typo")
    with pytest.raises(ValueError, match="s15.*s00000015"):
        manifest.robocap.session("s15")
    with pytest.raises(ValueError, match="msd-nope.*msd-index"):
        manifest.dataset("msd-nope")


def test_flow_config_loads_the_datasets_own_basalt_config(manifest: ReferenceManifest) -> None:
    """What the estimator is built with is the file, not constructor defaults.

    The difference is one key — ``vio_marg_lost_landmarks``, true in both MSD
    files and false in the constructor (C72) — so the assertion is on the config
    the binding hands back, written out again, rather than on the manifest's own
    text: a ``resolved_flow_config`` that quietly stopped reading the file would pass a
    test that only compared the JSON on disk.
    """
    default: str = _core.VioConfig().to_json()
    for segment in manifest.segments:
        config: _core.VioConfig
        config, _text = resolved_flow_config(manifest, segment)
        loaded: dict[str, Any] = json.loads(config.to_json())["value0"]
        assert loaded["config.vio_marg_lost_landmarks"] is True, segment.segment_id
        assert json.loads(default)["value0"]["config.vio_marg_lost_landmarks"] is False
        assert config.to_json() != default, segment.segment_id
    # The two devices differ in the radius and the port sees that difference.
    radii: set[float] = {resolved_flow_config(manifest, segment)[0].optical_flow_image_safe_radius for segment in manifest.segments}
    assert radii == {472.0, 340.0}


def test_resolved_flow_config_hands_back_the_very_string_it_parsed(manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The text beside the config is the one string ``VioConfig.from_json`` received, not a second read of the file.

    Every read returns a different text here, so a function that resolved the
    file twice could not hand back what it parsed; and the parse is recorded
    through the module's own ``_core`` name, so the string is compared byte for
    byte rather than through a re-serialization that would hide whitespace.
    """
    reads: list[int] = []
    original = ReferenceManifest.vio_config_text

    def differing(self: ReferenceManifest, name: str, profile: str = "reference") -> str:
        reads.append(len(reads))
        return original(self, name, profile) + "\n" * len(reads)

    parsed: list[str] = []

    class RecordingVioConfig:
        @staticmethod
        def from_json(text: str) -> _core.VioConfig:
            parsed.append(text)
            return _core.VioConfig.from_json(text)

    monkeypatch.setattr(ReferenceManifest, "vio_config_text", differing)
    monkeypatch.setattr(reference, "_core", SimpleNamespace(VioConfig=RecordingVioConfig))
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENTS[1])
    config: _core.VioConfig
    text: str
    config, text = resolved_flow_config(manifest, segment)
    assert parsed == [text]
    assert reads == [0]
    assert text != manifest.vio_config_text(segment.dataset_name)
    assert config.to_json() == _core.VioConfig.from_json(text).to_json()


def test_every_dataset_names_a_vendored_config_that_parses(manifest: ReferenceManifest) -> None:
    """Both files are checked in beside the manifest and use the accepted JSON formats."""
    for dataset in manifest.datasets:
        path: Path = manifest.package_root / dataset.vio_config
        assert path.is_file(), dataset.name
        assert path.parent == manifest.package_root / "configs"
        text: str = manifest.vio_config_text(dataset.name)
        assert text == path.read_text()
        radius: float = json.loads(text)["value0"]["config.optical_flow_image_safe_radius"]
        assert _core.VioConfig.from_json(text).optical_flow_image_safe_radius == radius, dataset.name


def test_an_unknown_tier_is_rejected(tmp_path: Path) -> None:
    broken: Path = tmp_path / "broken.toml"
    broken.write_text(MANIFEST_PATH.read_text().replace('tier = "smoke"', 'tier = "sometimes"', 1))
    with pytest.raises(ValueError, match="broken.toml.*sometimes.*smoke.*release.*listed"):
        load_manifest(broken)


def test_a_missing_robocap_key_is_a_typed_error(tmp_path: Path) -> None:
    """The loader promises `ValueError`; direct indexing raised a bare `KeyError`."""
    broken: Path = tmp_path / "no-downscale.toml"
    broken.write_text(MANIFEST_PATH.read_text().replace("downscale = 3\n", "", 1))
    with pytest.raises(ValueError, match="no-downscale.toml.*downscale"):
        load_manifest(broken)


def test_a_missing_robocap_table_is_a_typed_error(tmp_path: Path) -> None:
    """The `[robocap]` block runs to the end of the file, so cutting it off removes it."""
    broken: Path = tmp_path / "no-robocap.toml"
    text: str = MANIFEST_PATH.read_text()
    broken.write_text(text[: text.index("[robocap]")])
    with pytest.raises(ValueError, match="no-robocap.toml.*robocap"):
        load_manifest(broken)


def test_a_duplicate_segment_id_is_rejected(manifest: ReferenceManifest, tmp_path: Path) -> None:
    text: str = MANIFEST_PATH.read_text()
    first: ReferenceSegment = manifest.segments[0]
    second: ReferenceSegment = manifest.segments[1]
    broken: Path = tmp_path / "duplicate.toml"
    broken.write_text(text.replace(f'segment_id = "{second.segment_id}"', f'segment_id = "{first.segment_id}"', 1))
    with pytest.raises(ValueError, match="duplicate segment ids"):
        load_manifest(broken)


def test_an_unsupported_manifest_schema_is_refused(tmp_path: Path) -> None:
    path: Path = tmp_path / "old.toml"
    path.write_text(MANIFEST_PATH.read_text().replace("schema_version = 10", "schema_version = 8"))
    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(path)


def test_malformed_toml_is_refused_with_the_file_name(tmp_path: Path) -> None:
    path = tmp_path / "gate.toml"
    path.write_text("schema_version = 10\n[[dataset]\nname = 'broken'\n")
    with pytest.raises(ValueError, match=str(path)):
        load_manifest(path)


def test_dataset_imu_covers_unlisted_odyssey_and_preserves_holdouts(manifest: ReferenceManifest) -> None:
    assert {dataset.name for dataset in manifest.datasets} == {"msd-index", "msd-g2", "msd-odyssey"}
    assert all(dataset.imu == manifest.datasets[0].imu for dataset in manifest.datasets)
    assert sum(segment.hold_out for segment in manifest.segments) == 2
    assert {segment.segment_id.split("__")[-1].split("_")[0] for segment in manifest.in_tier("release")} == {"MIO07", "MGO07", "MIO14"}


@pytest.mark.parametrize("field", ["gt_rmse_cm", "median_tracker_ms"])
@pytest.mark.parametrize("value", ["0.0", "-1.0", "nan", "inf", "-inf"])
def test_invalid_baseline_measurement_is_rejected(tmp_path: Path, field: str, value: str) -> None:
    import re

    path: Path = tmp_path / "invalid-baseline.toml"
    path.write_text(re.sub(rf"{field} = [^\n]+", f"{field} = {value}", MANIFEST_PATH.read_text(), count=1))
    with pytest.raises(ValueError, match=field):
        load_manifest(path)


def test_duplicate_baseline_is_rejected(tmp_path: Path) -> None:
    text: str = MANIFEST_PATH.read_text()
    start: int = text.index("[[segment.baseline]]")
    end: int = text.index("\n[", start + 1)
    path: Path = tmp_path / "duplicate-baseline.toml"
    path.write_text(text[:end] + "\n" + text[start:end] + text[end:])
    with pytest.raises(ValueError, match="duplicate baseline"):
        load_manifest(path)


def test_gate_round_trips_through_toml(manifest: ReferenceManifest) -> None:
    import tomllib

    from serde.toml import from_toml, to_toml

    serialized: str = to_toml(manifest)
    assert from_toml(ReferenceManifest, serialized) == manifest
    assert tomllib.loads(serialized) == tomllib.loads(MANIFEST_PATH.read_text())


@pytest.mark.parametrize("replacement", ["typo = 1\n", 'schema_version = "invalid"\n'])
def test_gate_refuses_unknown_keys_and_wrong_types_with_path(tmp_path: Path, replacement: str) -> None:
    broken: Path = tmp_path / "broken-gate.toml"
    text: str = MANIFEST_PATH.read_text()
    broken.write_text(
        text.replace("schema_version = 10\n", replacement if replacement.startswith("schema_version") else replacement + "schema_version = 10\n")
    )
    with pytest.raises(ValueError, match="broken-gate.toml"):
        load_manifest(broken)


def test_gate_refuses_an_unknown_dataset_with_table_and_path(tmp_path: Path) -> None:
    broken: Path = tmp_path / "unknown-dataset.toml"
    broken.write_text(MANIFEST_PATH.read_text().replace('dataset_name = "msd-index"', 'dataset_name = "missing"', 1))
    with pytest.raises(ValueError, match="unknown-dataset.toml.*segment.*missing"):
        load_manifest(broken)
