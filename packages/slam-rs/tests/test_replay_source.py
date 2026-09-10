"""Which recording the replay tool opens: the manifest's files, a file of the caller's, or a catalog server.

The catalog case is what a machine with no NAS mount uses, so it is pinned
without a network: the segment is resolved and the source built before
:func:`slam_rs.catalog_feed.open_segment` is reached, which the test replaces.
"""

from pathlib import Path

import pytest
from simplecv.rerun_log_utils import RerunTyroConfig

from slam_rs.apis import replay
from slam_rs.apis.replay import Config, main
from slam_rs.catalog_feed import CatalogSegment, LocalSegment, SegmentSource
from slam_rs.reference import SMOKE_SEGMENTS, ImuParameters, ReferenceManifest

CATALOG: str = "rerun+http://dgx-spark:9988"
UNLISTED: str = "msd-g2__MGO_others__MGO11_short_3_backandforth"
"""A catalog segment of a known dataset that the reference set does not list."""


class _Opened(Exception):
    """Raised by the stand-in feed so ``main`` stops once the source is chosen."""


def _capture_source(monkeypatch: pytest.MonkeyPatch, parameters: list[ImuParameters] | None = None) -> list[SegmentSource]:
    seen: list[SegmentSource] = []

    def opened(source: SegmentSource, imu: ImuParameters, *_args: object, **_kwargs: object) -> None:
        seen.append(source)
        if parameters is not None:
            parameters.append(imu)
        raise _Opened

    monkeypatch.setattr(replay, "open_segment", opened)
    return seen


def test_a_catalog_url_replaces_the_manifests_file_paths(manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--catalog`` opens the segment by id on the server, ground truth included, with no file path involved."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    segment = manifest.by_id(SMOKE_SEGMENTS[1])
    with pytest.raises(_Opened):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=segment.segment_id, catalog=CATALOG))
    assert seen == [CatalogSegment(url=CATALOG, dataset_name=segment.dataset_name, segment_id=segment.segment_id)]


def test_without_a_catalog_the_manifests_files_are_opened(manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default source is unchanged: the manifest's base and ground-truth recordings."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    segment = manifest.by_id(SMOKE_SEGMENTS[1])
    with pytest.raises(_Opened):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=segment.segment_id))
    assert seen == [LocalSegment(base_rrd=segment.base_path, gt_rrd=segment.gt_path)]


def test_a_catalog_and_a_file_are_two_sources_and_refused(manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--catalog`` with ``--rrd`` would silently pick one; the run stops with a sentence instead."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    with pytest.raises(ValueError, match="two sources"):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=SMOKE_SEGMENTS[1], catalog=CATALOG, rrd=tmp_path / "base.rrd"))
    assert seen == []


def test_an_unlisted_catalog_segment_takes_its_datasets_parameters(manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any segment of a known dataset replays from the catalog with that dataset's IMU model; no manifest entry is needed."""
    parameters: list[ImuParameters] = []
    seen: list[SegmentSource] = _capture_source(monkeypatch, parameters)
    with pytest.raises(_Opened):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=UNLISTED, catalog=CATALOG))
    assert seen == [CatalogSegment(url=CATALOG, dataset_name="msd-g2", segment_id=UNLISTED)]
    assert parameters == [next(s.imu for s in manifest.segments if s.dataset_name == "msd-g2")]


def test_an_unlisted_segment_without_a_catalog_is_refused_with_the_way_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest has files only for the reference set; the refusal names --catalog."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    with pytest.raises(ValueError, match="not in the reference set.*--catalog"):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=UNLISTED))
    assert seen == []


def test_a_catalog_segment_of_an_unknown_dataset_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dataset the manifest has no config or IMU model for cannot be replayed, catalog or not."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    with pytest.raises(ValueError, match="'msd-nowhere' is not in the reference set"):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment="msd-nowhere__X__Y", catalog=CATALOG))
    assert seen == []
