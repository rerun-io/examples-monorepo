"""The reference manifest parses, is internally consistent, and still matches the catalog."""

import socket
import urllib.parse
from pathlib import Path

import numpy as np
import pytest
from beartype.roar import BeartypeException

from slam_rs.reference import (
    DECODE_PATH_BY_NAME,
    MANIFEST_PATH,
    TIER_BY_NAME,
    ReferenceManifest,
    ReferenceSegment,
    load_manifest,
)

CATALOG_CONNECT_TIMEOUT_S: float = 3.0
"""How long the slow test waits for the catalog before it skips."""


@pytest.fixture(scope="module")
def manifest() -> ReferenceManifest:
    """The checked-in reference manifest."""
    return load_manifest()


def test_the_manifest_holds_ten_segments(manifest: ReferenceManifest) -> None:
    assert len(manifest.segments) == 10
    assert manifest.schema_version == 1


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
    assert len(manifest.in_tier("accuracy")) == 6
    assert len(manifest.in_tier("long")) == 2
    assert {segment.dataset_name for segment in manifest.in_tier("smoke")} == {"msd-index", "msd-g2"}


def test_urls_and_sidecars_agree_with_the_segment_id(manifest: ReferenceManifest) -> None:
    for segment in manifest.segments:
        assert segment.base_url == f"file:///mnt/nas/datasets/msd-rrd/base/{segment.segment_id}.rrd"
        assert segment.gt_url == f"file:///mnt/nas/datasets/msd-rrd/gt/{segment.segment_id}.rrd"
        assert segment.gt_csv == Path(f"/mnt/nas/datasets/msd-rrd/sidecars/{segment.segment_id}/gt.csv")
        assert segment.segment_id.startswith(segment.dataset_name)


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


def test_robocap_is_session_fifteen_with_no_ground_truth(manifest: ReferenceManifest) -> None:
    assert manifest.robocap.session_id == "s00000015"
    assert manifest.robocap.has_ground_truth is False
    assert manifest.robocap.basalt_num_poses == 1588
    assert manifest.robocap.imu.cam_time_offset_ns == 14_902_432
    assert manifest.robocap.imu.rate_hz == 200.0


def test_the_robocap_fixtures_are_checked_in(manifest: ReferenceManifest) -> None:
    for relative in (manifest.robocap.fixtures.golden, manifest.robocap.fixtures.candidate):
        assert (manifest.package_root / relative).is_file()


def test_an_unknown_tier_is_rejected(tmp_path: Path) -> None:
    broken: Path = tmp_path / "broken.toml"
    broken.write_text(MANIFEST_PATH.read_text().replace('tier = "smoke"', 'tier = "sometimes"', 1))
    with pytest.raises(ValueError, match="unknown tier"):
        load_manifest(broken)


def test_a_duplicate_segment_id_is_rejected(tmp_path: Path) -> None:
    text: str = MANIFEST_PATH.read_text()
    first: ReferenceSegment = load_manifest().segments[0]
    second: ReferenceSegment = load_manifest().segments[1]
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
            assert int(np.asarray(scalar("property:gt:num_poses")).item()) == segment.gt.num_poses
            assert str(scalar("property:gt:source")) == segment.gt.source
