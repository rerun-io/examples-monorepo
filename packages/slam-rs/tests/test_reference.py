"""The gate parses and preserves tier, baseline and sensor-model rules."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from slam_rs import _core, reference
from slam_rs.config import SLAM_CONFIG_PATH, SlamConfig, load_slam_config
from slam_rs.reference import (
    BENCHMARKS_PATH,
    DECODE_PATH_BY_NAME,
    SMOKE_SEGMENTS,
    TIER_BY_NAME,
    Benchmarks,
    ReferenceSegment,
    load_benchmarks,
    resolved_flow_config,
)


def test_benchmarks_hold_ten_segments(benchmarks: Benchmarks) -> None:
    assert len(benchmarks.segments) == 10
    assert benchmarks.schema_version == 1


def test_segment_ids_are_unique(benchmarks: Benchmarks) -> None:
    identifiers: list[str] = [segment.segment_id for segment in benchmarks.segments]
    assert len(set(identifiers)) == len(identifiers)


def test_every_tier_and_decode_path_is_known(benchmarks: Benchmarks) -> None:
    for segment in benchmarks.segments:
        assert segment.tier in TIER_BY_NAME
        assert segment.decode_path in DECODE_PATH_BY_NAME
    assert {segment.tier for segment in benchmarks.segments} == set(TIER_BY_NAME)


def test_the_tiers_are_the_ones_the_plan_names(benchmarks: Benchmarks) -> None:
    assert len(benchmarks.in_tier("smoke")) == 2
    assert len(benchmarks.in_tier("release")) == 3
    assert len(benchmarks.in_tier("listed")) == 5
    assert {segment.dataset_name for segment in benchmarks.in_tier("smoke")} == {"msd-index", "msd-g2"}


def test_any_robocap_session_of_the_device_replays_without_a_reference(benchmarks: Benchmarks, settings: SlamConfig) -> None:
    listed = benchmarks.robocap.session("s00000015")
    assert listed.reference_csv is not None and benchmarks.robocap.is_listed("s00000015")
    other = benchmarks.robocap.session("s00000099")
    assert other.segment_id == f"robocap__{settings.robocap.device_id}__s00000099"
    assert other.reference_csv is None and not benchmarks.robocap.is_listed("s00000099")


def test_robocap_carries_s15_and_no_ground_truth(benchmarks: Benchmarks) -> None:
    assert [session.session_id for session in benchmarks.robocap.sessions] == ["s00000015"]
    assert benchmarks.robocap.has_ground_truth is False


def test_robocap_names_the_configuration_the_cpp_ran(benchmarks: Benchmarks, settings: SlamConfig) -> None:
    """The four cameras, the downscale, the two rig rules and the two configured JSON files, all present."""
    assert settings.robocap.camera_names == ("left", "left_front", "right_front", "right")
    assert settings.robocap.downscale == 3
    assert benchmarks.robocap.decode_path == "cpu_gray8_swscale_area_downscale3"
    # The settings defines frameset tolerance and inertial pairing.
    # because the two inertial channels are on their own clocks.
    assert settings.robocap.frameset_tolerance_ns == 1_000_000
    assert settings.robocap.interpolate_accel_onto_gyro is True
    # The recorder stamps the device clock on the video timeline, so an export
    # adds nothing; MSD's `video_time` is relative to its capture start.
    assert settings.robocap.video_time_is_absolute is True
    assert "config.vio_marg_lost_landmarks" in (settings.package_root / settings.robocap.vio_config).read_text()
    assert '"camera_type": "kb4"' in (settings.package_root / settings.robocap.calibration).read_text()


def test_an_unknown_selector_is_a_typed_error(benchmarks: Benchmarks, settings: SlamConfig) -> None:
    """Every accessor that resolves a name names the ones there are, as a ``ValueError``.

    These three are what a tool resolves a command line through, and a
    ``KeyError`` reads as a dictionary miss: ``fleet_check --segments <clip>
    typo`` reached one after replaying the valid clip. The loader has always
    promised ``ValueError`` for a settings it cannot read; a selector it cannot
    satisfy is the same kind of answer.
    """
    with pytest.raises(ValueError, match="MIO10_typo.*MIO10_short_2_panorama"):
        benchmarks.by_id("MIO10_typo")
    with pytest.raises(ValueError, match="s15.*s00000015"):
        benchmarks.robocap.session("s15")
    with pytest.raises(ValueError, match="msd-nope.*msd-index"):
        settings.dataset("msd-nope")


def test_flow_config_loads_the_datasets_own_basalt_config(benchmarks: Benchmarks, settings: SlamConfig) -> None:
    """What the estimator is built with is the file, not constructor defaults.

    The difference is one key — ``vio_marg_lost_landmarks``, true in both MSD
    files and false in the constructor (C72) — so the assertion is on the config
    the binding hands back, written out again, rather than on the settings’ own
    text: a ``resolved_flow_config`` that quietly stopped reading the file would pass a
    test that only compared the JSON on disk.
    """
    default: str = _core.VioConfig().to_json()
    for segment in benchmarks.segments:
        config: _core.VioConfig
        config, _text = resolved_flow_config(settings, segment)
        loaded: dict[str, Any] = json.loads(config.to_json())["value0"]
        assert loaded["config.vio_marg_lost_landmarks"] is True, segment.segment_id
        assert json.loads(default)["value0"]["config.vio_marg_lost_landmarks"] is False
        assert config.to_json() != default, segment.segment_id
    # The two devices differ in the radius and the port sees that difference.
    radii: set[float] = {resolved_flow_config(settings, segment)[0].optical_flow_image_safe_radius for segment in benchmarks.segments}
    assert radii == {472.0, 340.0}


def test_resolved_flow_config_hands_back_the_very_string_it_parsed(
    benchmarks: Benchmarks, settings: SlamConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The text beside the config is the one string ``VioConfig.from_json`` received, not a second read of the file.

    Every read returns a different text here, so a function that resolved the
    file twice could not hand back what it parsed; and the parse is recorded
    through the module's own ``_core`` name, so the string is compared byte for
    byte rather than through a re-serialization that would hide whitespace.
    """
    reads: list[int] = []
    original = SlamConfig.vio_config_text

    def differing(self: SlamConfig, name: str, profile: str = "reference") -> str:
        reads.append(len(reads))
        return original(self, name, profile) + "\n" * len(reads)

    parsed: list[str] = []

    class RecordingVioConfig:
        @staticmethod
        def from_json(text: str) -> _core.VioConfig:
            parsed.append(text)
            return _core.VioConfig.from_json(text)

    monkeypatch.setattr(SlamConfig, "vio_config_text", differing)
    monkeypatch.setattr(reference, "_core", SimpleNamespace(VioConfig=RecordingVioConfig))
    segment: ReferenceSegment = benchmarks.by_id(SMOKE_SEGMENTS[1])
    config: _core.VioConfig
    text: str
    config, text = resolved_flow_config(settings, segment)
    assert parsed == [text]
    assert reads == [0]
    assert text != settings.vio_config_text(segment.dataset_name)
    assert config.to_json() == _core.VioConfig.from_json(text).to_json()


def test_every_dataset_names_a_vendored_config_that_parses(settings: SlamConfig) -> None:
    """Both files are checked in beside the settings and use the accepted JSON formats."""
    for dataset in settings.datasets:
        path: Path = settings.package_root / dataset.vio_config
        assert path.is_file(), dataset.name
        assert path.parent == settings.package_root / "configs"
        text: str = settings.vio_config_text(dataset.name)
        assert text == path.read_text()
        radius: float = json.loads(text)["value0"]["config.optical_flow_image_safe_radius"]
        assert _core.VioConfig.from_json(text).optical_flow_image_safe_radius == radius, dataset.name


def test_an_unknown_tier_is_rejected(settings: SlamConfig, tmp_path: Path) -> None:
    broken: Path = tmp_path / "broken.toml"
    broken.write_text(BENCHMARKS_PATH.read_text().replace('tier = "smoke"', 'tier = "sometimes"', 1))
    with pytest.raises(ValueError, match="broken.toml.*sometimes.*smoke.*release.*listed"):
        load_benchmarks(settings, broken)


def test_a_missing_robocap_key_is_a_typed_error(tmp_path: Path) -> None:
    """The loader promises `ValueError`; direct indexing raised a bare `KeyError`."""
    broken: Path = tmp_path / "no-downscale.toml"
    broken.write_text(SLAM_CONFIG_PATH.read_text().replace("downscale = 3\n", "", 1))
    with pytest.raises(ValueError, match="no-downscale.toml.*downscale"):
        load_slam_config(broken)


def test_a_missing_robocap_table_is_a_typed_error(tmp_path: Path) -> None:
    """The `[robocap]` block runs to the end of the file, so cutting it off removes it."""
    broken: Path = tmp_path / "no-robocap.toml"
    text: str = SLAM_CONFIG_PATH.read_text()
    broken.write_text(text[: text.index("[robocap]")])
    with pytest.raises(ValueError, match="no-robocap.toml.*robocap"):
        load_slam_config(broken)


def test_a_duplicate_segment_id_is_rejected(benchmarks: Benchmarks, settings: SlamConfig, tmp_path: Path) -> None:
    text: str = BENCHMARKS_PATH.read_text()
    first: ReferenceSegment = benchmarks.segments[0]
    second: ReferenceSegment = benchmarks.segments[1]
    broken: Path = tmp_path / "duplicate.toml"
    broken.write_text(text.replace(f'segment_id = "{second.segment_id}"', f'segment_id = "{first.segment_id}"', 1))
    with pytest.raises(ValueError, match="duplicate segment ids"):
        load_benchmarks(settings, broken)


def test_an_unsupported_runtime_schema_is_refused(tmp_path: Path) -> None:
    path: Path = tmp_path / "old.toml"
    path.write_text(SLAM_CONFIG_PATH.read_text().replace("schema_version = 2", "schema_version = 8"))
    with pytest.raises(ValueError, match="schema_version"):
        load_slam_config(path)


def test_malformed_toml_is_refused_with_the_file_name(tmp_path: Path) -> None:
    path = tmp_path / "slam.toml"
    path.write_text("schema_version = 1\n[[dataset]\nname = 'broken'\n")
    with pytest.raises(ValueError, match=str(path)):
        load_slam_config(path)


def test_dataset_settings_cover_unlisted_odyssey_and_preserve_holdouts(benchmarks: Benchmarks, settings: SlamConfig) -> None:
    assert {dataset.name for dataset in settings.datasets} == {"msd-index", "msd-g2", "msd-odyssey"}
    assert sum(segment.hold_out for segment in benchmarks.segments) == 2
    assert {segment.segment_id.split("__")[-1].split("_")[0] for segment in benchmarks.in_tier("release")} == {"MIO07", "MGO07", "MIO14"}


@pytest.mark.parametrize("field", ["gt_rmse_cm", "median_tracker_ms"])
@pytest.mark.parametrize("value", ["0.0", "-1.0", "nan", "inf", "-inf"])
def test_invalid_baseline_measurement_is_rejected(settings: SlamConfig, tmp_path: Path, field: str, value: str) -> None:
    import re

    path: Path = tmp_path / "invalid-baseline.toml"
    path.write_text(re.sub(rf"{field} = [^\n]+", f"{field} = {value}", BENCHMARKS_PATH.read_text(), count=1))
    with pytest.raises(ValueError, match=field):
        load_benchmarks(settings, path)


def test_duplicate_baseline_is_rejected(settings: SlamConfig, tmp_path: Path) -> None:
    text: str = BENCHMARKS_PATH.read_text()
    start: int = text.index("[[segment.baseline]]")
    end: int = text.index("\n[", start + 1)
    path: Path = tmp_path / "duplicate-baseline.toml"
    path.write_text(text[:end] + "\n" + text[start:end] + text[end:])
    with pytest.raises(ValueError, match="duplicate baseline"):
        load_benchmarks(settings, path)


@given(host=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1), duplicate=st.booleans())
def test_baseline_duplicates_include_host(settings: SlamConfig, host: str, duplicate: bool) -> None:
    text: str = BENCHMARKS_PATH.read_text()
    start: int = text.index("[[segment.baseline]]")
    end: int = text.index("\n[", start + 1)
    row: str = text[start:end]
    if not duplicate:
        row = "\n".join(f'host = "{host}"' if line.startswith("host = ") else line for line in row.splitlines())
    with TemporaryDirectory() as directory:
        path: Path = Path(directory) / "hosts.toml"
        path.write_text(text[:end] + "\n" + row + text[end:])
        if duplicate:
            with pytest.raises(ValueError, match="duplicate baseline"):
                load_benchmarks(settings, path)
        else:
            parsed: Benchmarks = load_benchmarks(settings, path)
            assert parsed.segments[0].baseline[1].host == host


def test_runtime_settings_round_trip_through_toml(settings: SlamConfig) -> None:
    import tomllib

    from serde.toml import from_toml, to_toml

    serialized: str = to_toml(settings)
    assert from_toml(SlamConfig, serialized) == settings
    assert tomllib.loads(serialized) == tomllib.loads(SLAM_CONFIG_PATH.read_text())


@pytest.mark.parametrize("replacement", ["typo = 1\n", 'schema_version = "invalid"\n'])
def test_runtime_settings_refuse_unknown_keys_and_wrong_types_with_path(tmp_path: Path, replacement: str) -> None:
    broken: Path = tmp_path / "broken-slam.toml"
    text: str = SLAM_CONFIG_PATH.read_text()
    broken.write_text(
        text.replace("schema_version = 2\n", replacement if replacement.startswith("schema_version") else replacement + "schema_version = 2\n")
    )
    with pytest.raises(ValueError, match="broken-slam.toml"):
        load_slam_config(broken)


def test_benchmarks_refuse_an_unknown_dataset_with_table_and_path(settings: SlamConfig, tmp_path: Path) -> None:
    broken: Path = tmp_path / "unknown-dataset.toml"
    broken.write_text(BENCHMARKS_PATH.read_text().replace('dataset_name = "msd-index"', 'dataset_name = "missing"', 1))
    with pytest.raises(ValueError, match="unknown-dataset.toml.*segment.*missing"):
        load_benchmarks(settings, broken)


@pytest.mark.parametrize("suffix", [".attlocal.net", ".office.example"])
def test_dotted_baseline_host_is_rejected(settings: SlamConfig, tmp_path: Path, suffix: str) -> None:
    path: Path = tmp_path / "dotted-host.toml"
    path.write_text(BENCHMARKS_PATH.read_text().replace('host = "pablo-dl-server"', f'host = "pablo-dl-server{suffix}"', 1))
    with pytest.raises(ValueError, match=r"dotted-host.toml.*host.*short.*no"):
        load_benchmarks(settings, path)


def test_benchmarks_round_trip_through_toml(benchmarks: Benchmarks) -> None:
    import tomllib

    from serde.toml import from_toml, to_toml

    serialized: str = to_toml(benchmarks)
    assert from_toml(Benchmarks, serialized) == benchmarks
    assert tomllib.loads(serialized) == tomllib.loads(BENCHMARKS_PATH.read_text())


@pytest.mark.parametrize("replacement", ["schema_version = 8\n", 'schema_version = "invalid"\n', "typo = 1\nschema_version = 1\n"])
def test_benchmarks_refuse_invalid_schema_with_path(settings: SlamConfig, tmp_path: Path, replacement: str) -> None:
    broken: Path = tmp_path / "broken-benchmarks.toml"
    broken.write_text(BENCHMARKS_PATH.read_text().replace("schema_version = 1\n", replacement, 1))
    with pytest.raises(ValueError, match="broken-benchmarks.toml"):
        load_benchmarks(settings, broken)


def test_malformed_benchmarks_are_refused_with_path(settings: SlamConfig, tmp_path: Path) -> None:
    broken: Path = tmp_path / "broken-benchmarks.toml"
    broken.write_text("schema_version = 1\n[[segment]\n")
    with pytest.raises(ValueError, match="broken-benchmarks.toml"):
        load_benchmarks(settings, broken)
