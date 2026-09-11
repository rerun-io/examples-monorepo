"""The reference manifest parses, is internally consistent, and still matches the catalog."""

import json
import socket
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from beartype.roar import BeartypeException

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

CATALOG_CONNECT_TIMEOUT_S: float = 3.0
"""How long the slow test waits for the catalog before it skips."""


def test_the_manifest_holds_ten_segments(manifest: ReferenceManifest) -> None:
    assert len(manifest.segments) == 10
    assert manifest.schema_version == 9


def test_every_segment_carries_both_layer_fingerprints(manifest: ReferenceManifest) -> None:
    for segment in manifest.segments:
        assert set(segment.layers) == {"base", "gt"}
        for name, layer in segment.layers.items():
            assert layer.size_bytes > 0, f"{segment.segment_id}/{name}"
            assert len(layer.schema_sha256) == 64
            assert set(layer.schema_sha256) <= set("0123456789abcdef")
        # The base layer is the video; the gt layer is poses only, so it is smaller.
        assert segment.layers["gt"].size_bytes < segment.layers["base"].size_bytes


def test_the_schema_hash_is_shared_within_a_dataset_and_layer(manifest: ReferenceManifest) -> None:
    """One conversion wrote each dataset, so a differing hash means a partial re-conversion.

    The two datasets do not share a schema: msd-g2 has four cameras and a
    magnetometer where msd-index has two cameras and neither.
    """
    for dataset in manifest.datasets:
        for layer_name in ("base", "gt"):
            digests: set[str] = {s.layers[layer_name].schema_sha256 for s in manifest.segments if s.dataset_name == dataset.name}
            assert len(digests) == (1 if any(s.dataset_name == dataset.name for s in manifest.segments) else 0), (
                f"{dataset.name} {layer_name} layers disagree on schema: {sorted(digests)}"
            )
    index_base: str = manifest.by_id("msd-index__MIO_others__MIO10_short_2_panorama").layers["base"].schema_sha256
    g2_base: str = manifest.by_id("msd-g2__MGO_others__MGO09_short_1_updown").layers["base"].schema_sha256
    assert index_base != g2_base


def test_the_datasets_describe_the_two_rigs(manifest: ReferenceManifest) -> None:
    index = manifest.dataset("msd-index")
    assert index.num_cameras == 2
    assert index.camera_resolution_wh == ((960, 960), (960, 960))
    assert index.image_rotation_cw_deg == (0, 0)
    g2 = manifest.dataset("msd-g2")
    assert g2.num_cameras == 4
    # Portrait, and the two pairs face opposite ways: an estimator that assumes a
    # shared orientation is wrong on this rig.
    assert g2.camera_resolution_wh == ((480, 640), (480, 640), (480, 640), (480, 640))
    assert g2.image_rotation_cw_deg == (90, 90, 270, 270)


def test_every_segment_matches_its_dataset_entry(manifest: ReferenceManifest) -> None:
    for segment in manifest.segments:
        dataset = manifest.dataset(segment.dataset_name)
        assert segment.dataset_entry_id == dataset.entry_id
        assert segment.capture.num_cameras == dataset.num_cameras


def test_the_capture_start_time_is_the_absolute_clock_offset(manifest: ReferenceManifest) -> None:
    for segment in manifest.segments:
        assert segment.capture.start_time_ns > 0
    # The dossier's worked example on the smoke segment.
    assert manifest.by_id("msd-index__MIO_others__MIO10_short_2_panorama").capture.start_time_ns == 10_433_867_587_166


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


def test_camera_counts_match_the_device(manifest: ReferenceManifest) -> None:
    for segment in manifest.segments:
        expected: int = 2 if segment.dataset_name == "msd-index" else 4
        assert segment.capture.num_cameras == expected
        assert segment.gt.source == ("lighthouse" if segment.dataset_name == "msd-index" else "mocap")


def test_the_msd_imu_block_is_basalts(manifest: ReferenceManifest) -> None:
    for segment in manifest.segments:
        assert segment.imu.rate_hz == 1000.0
        assert segment.imu.gyro_noise_std == 0.000282
        assert segment.imu.accel_noise_std == 0.016
        assert segment.imu.gyro_bias_std == 0.0001
        assert segment.imu.accel_bias_std == 0.001
        assert segment.imu.cam_time_offset_ns == 0


def test_robocap_carries_s15_and_no_ground_truth(manifest: ReferenceManifest) -> None:
    assert [session.session_id for session in manifest.robocap.sessions] == ["s00000015"]
    assert manifest.robocap.has_ground_truth is False
    assert manifest.robocap.imu.cam_time_offset_ns == 14_902_432
    assert manifest.robocap.imu.rate_hz == 200.0


def test_robocap_names_the_configuration_the_cpp_ran(manifest: ReferenceManifest) -> None:
    """The four cameras, the downscale, the two rig rules and basalt's own two files, all present."""
    assert manifest.robocap.camera_names == ("left", "left_front", "right_front", "right")
    assert manifest.robocap.downscale == 3
    assert manifest.robocap.decode_path == "cpu_gray8_swscale_area_downscale3"
    # basalt's `dataset_io_robocap.cpp` tolerance, and the pairing its reader does
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
    with pytest.raises(ValueError, match="s00000099.*s00000015"):
        manifest.robocap.session("s00000099")
    with pytest.raises(ValueError, match="msd-nope.*msd-index"):
        manifest.dataset("msd-nope")


def test_flow_config_loads_the_datasets_own_basalt_config(manifest: ReferenceManifest) -> None:
    """What the estimator is built with is the file, not basalt's constructor defaults.

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
    """Both files are checked in beside the manifest and are basalt's own shape."""
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
    with pytest.raises(ValueError, match="unknown tier"):
        load_manifest(broken)


def test_a_missing_robocap_key_is_a_typed_error(tmp_path: Path) -> None:
    """The loader promises `ValueError`; direct indexing raised a bare `KeyError`."""
    broken: Path = tmp_path / "no-downscale.toml"
    broken.write_text(MANIFEST_PATH.read_text().replace("downscale = 3\n", "", 1))
    with pytest.raises(ValueError, match=r"\[robocap\] table is missing the key 'downscale'"):
        load_manifest(broken)


def test_a_missing_robocap_table_is_a_typed_error(tmp_path: Path) -> None:
    """The `[robocap]` block runs to the end of the file, so cutting it off removes it."""
    broken: Path = tmp_path / "no-robocap.toml"
    text: str = MANIFEST_PATH.read_text()
    broken.write_text(text[: text.index("[robocap]")])
    with pytest.raises(ValueError, match=r"has no \[robocap\] table"):
        load_manifest(broken)


def test_a_duplicate_segment_id_is_rejected(manifest: ReferenceManifest, tmp_path: Path) -> None:
    text: str = MANIFEST_PATH.read_text()
    first: ReferenceSegment = manifest.segments[0]
    second: ReferenceSegment = manifest.segments[1]
    broken: Path = tmp_path / "duplicate.toml"
    broken.write_text(text.replace(f'segment_id = "{second.segment_id}"', f'segment_id = "{first.segment_id}"', 1))
    with pytest.raises(ValueError, match="duplicate segment ids"):
        load_manifest(broken)


def _catalog_is_reachable(url: str) -> bool:
    """Whether the catalog's gRPC port accepts a TCP connection within the timeout.

    ``CatalogClient`` has no connect timeout, and an unreachable host makes it
    hang far longer than a test should, so the reachability question is answered
    by a plain socket first.
    """
    parsed: urllib.parse.ParseResult = urllib.parse.urlparse(url.replace("rerun+", ""))
    if parsed.hostname is None or parsed.port is None:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=CATALOG_CONNECT_TIMEOUT_S):
            return True
    except BeartypeException:
        raise
    except OSError:
        return False


@pytest.mark.slow
def test_the_catalog_still_reports_the_manifest_properties(manifest: ReferenceManifest) -> None:
    """One properties query per dataset covers all its segments; skip when the server is unreachable."""
    import pyarrow as pa
    from rerun.catalog import CatalogClient

    if not _catalog_is_reachable(manifest.catalog_url):
        pytest.skip(f"catalog {manifest.catalog_url} is unreachable within {CATALOG_CONNECT_TIMEOUT_S} s")
    client: CatalogClient = CatalogClient(manifest.catalog_url)
    entries: dict[str, str] = {entry.name: str(entry.id) for entry in client.entries()}

    for dataset_name in sorted({segment.dataset_name for segment in manifest.segments}):
        assert dataset_name in entries, f"{dataset_name} is not registered on {manifest.catalog_url}"
        dataset = client.get_dataset(dataset_name)
        table: pa.Table = dataset.filter_contents(["/__properties", "/__properties/**"]).reader(index=None).to_arrow_table()
        rows: dict[str, dict[str, object]] = {}
        segment_ids: list[str] = table["rerun_segment_id"].combine_chunks().to_pylist()
        for row_index, segment_id in enumerate(segment_ids):
            rows[segment_id] = {name: table[name][row_index].as_py() for name in table.column_names if name.startswith("property:")}

        for segment in (s for s in manifest.segments if s.dataset_name == dataset_name):
            assert segment.dataset_entry_id == entries[dataset_name]
            assert segment.segment_id in rows, f"{segment.segment_id} is no longer a segment of {dataset_name}"
            properties: dict[str, object] = rows[segment.segment_id]

            def scalar(key: str, properties: dict[str, object] = properties) -> object:
                value: object = properties[key]
                return value[0] if isinstance(value, list) else value

            assert int(np.asarray(scalar("property:capture:duration_ns")).item()) == segment.capture.duration_ns
            assert int(np.asarray(scalar("property:capture:num_frames")).item()) == segment.capture.num_frames
            assert int(np.asarray(scalar("property:capture:num_cameras")).item()) == segment.capture.num_cameras
            assert int(np.asarray(scalar("property:capture:start_time_ns")).item()) == segment.capture.start_time_ns
            assert int(np.asarray(scalar("property:gt:num_poses")).item()) == segment.gt.num_poses
            assert str(scalar("property:gt:source")) == segment.gt.source


@pytest.mark.slow
def test_the_catalog_still_reports_the_manifest_layer_fingerprints(manifest: ReferenceManifest) -> None:
    """Sizes and schema hashes, one ``manifest()`` round-trip per dataset."""
    import pyarrow as pa
    from rerun.catalog import CatalogClient

    if not _catalog_is_reachable(manifest.catalog_url):
        pytest.skip(f"catalog {manifest.catalog_url} is unreachable within {CATALOG_CONNECT_TIMEOUT_S} s")
    client: CatalogClient = CatalogClient(manifest.catalog_url)

    for dataset_name in sorted({segment.dataset_name for segment in manifest.segments}):
        # Deprecated but the only surface that reports size and schema digest.
        table: pa.Table = client.get_dataset(dataset_name).manifest().to_arrow_table()
        live: dict[tuple[str, str], tuple[int, str]] = {
            (table["rerun_segment_id"][row].as_py(), table["rerun_layer_name"][row].as_py()): (
                int(table["rerun_size_bytes"][row].as_py()),
                bytes(table["rerun_schema_sha256"][row].as_py()).hex(),
            )
            for row in range(table.num_rows)
        }
        for segment in (s for s in manifest.segments if s.dataset_name == dataset_name):
            for layer_name, fingerprint in segment.layers.items():
                key: tuple[str, str] = (segment.segment_id, layer_name)
                assert key in live, f"{layer_name} layer of {segment.segment_id} is no longer registered"
                assert live[key] == (fingerprint.size_bytes, fingerprint.schema_sha256), (
                    f"{segment.segment_id}/{layer_name}: catalog reports {live[key]}, manifest pins "
                    f"{(fingerprint.size_bytes, fingerprint.schema_sha256)}"
                )


@pytest.mark.slow
def test_the_catalog_still_reports_the_manifest_rig_geometry(manifest: ReferenceManifest) -> None:
    """Per-camera resolution and image rotation, one statics round-trip per dataset."""
    import pyarrow as pa
    from rerun.catalog import CatalogClient

    if not _catalog_is_reachable(manifest.catalog_url):
        pytest.skip(f"catalog {manifest.catalog_url} is unreachable within {CATALOG_CONNECT_TIMEOUT_S} s")
    client: CatalogClient = CatalogClient(manifest.catalog_url)

    for dataset in manifest.datasets:
        entities: list[str] = [f"/world/rig_00/cam_{index:02d}" for index in range(dataset.num_cameras)]
        statics: pa.Table = (
            client.get_dataset(dataset.name)
            .filter_contents(entities + [f"{entity}/pinhole" for entity in entities])
            .reader(index=None)
            .to_arrow_table()
        )
        for index, entity in enumerate(entities):
            # Calibration is byte-identical across a dataset's segments; assert it
            # rather than assume it, because a re-conversion could break it.
            resolutions: set[tuple[int, int]] = set()
            for row in range(statics.num_rows):
                values = np.asarray(statics[f"{entity}/pinhole:Pinhole:resolution"][row].values.to_pylist(), dtype=np.float64).ravel()
                resolutions.add((int(values[0]), int(values[1])))
            assert resolutions == {dataset.camera_resolution_wh[index]}, f"{dataset.name} cam_{index:02d}: {sorted(resolutions)}"

            rotation_column: str = f"{entity}:image_rotation_cw_deg"
            rotation: int = 0
            if rotation_column in statics.column_names and statics[rotation_column][0].is_valid:
                rotation = int(np.asarray(statics[rotation_column][0].values.to_pylist(), dtype=np.float64).ravel()[0])
            assert rotation == dataset.image_rotation_cw_deg[index], f"{dataset.name} cam_{index:02d} rotation"
