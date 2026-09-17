"""Which recording the replay tool opens: the settings’ files, a file of the caller's, or a catalog server.

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
from slam_rs.config import SlamConfig
from slam_rs.reference import SMOKE_SEGMENTS, Benchmarks

CATALOG: str = "rerun+http://127.0.0.1:51235"
UNLISTED: str = "msd-g2__MGO_others__MGO11_short_3_backandforth"
"""A catalog segment of a known dataset that the reference set does not list."""


class _Opened(Exception):
    """Raised by the stand-in feed so ``main`` stops once the source is chosen."""


def _capture_source(monkeypatch: pytest.MonkeyPatch) -> list[SegmentSource]:
    seen: list[SegmentSource] = []

    def opened(source: SegmentSource, *_args: object, **_kwargs: object) -> None:
        seen.append(source)
        raise _Opened

    monkeypatch.setattr(replay, "open_segment", opened)
    return seen


def test_a_catalog_url_replaces_the_manifests_file_paths(benchmarks: Benchmarks, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--catalog`` opens the segment by id on the server, ground truth included, with no file path involved."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    segment = benchmarks.by_id(SMOKE_SEGMENTS[1])
    with pytest.raises(_Opened):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=segment.segment_id, catalog=CATALOG))
    assert seen == [CatalogSegment(url=CATALOG, dataset_name=segment.dataset_name, segment_id=segment.segment_id)]


def test_without_an_override_the_manifest_catalog_is_used(benchmarks: Benchmarks, settings: SlamConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default source is the catalog URL in the settings."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    segment = benchmarks.by_id(SMOKE_SEGMENTS[1])
    with pytest.raises(_Opened):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=segment.segment_id))
    assert seen == [CatalogSegment(settings.catalog_url, segment.dataset_name, segment.segment_id)]


def test_a_catalog_and_a_file_are_two_sources_and_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--catalog`` with ``--rrd`` would silently pick one; the run stops with a sentence instead."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    with pytest.raises(ValueError, match="two sources"):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=SMOKE_SEGMENTS[1], catalog=CATALOG, rrd=tmp_path / "base.rrd"))
    assert seen == []


def test_an_unlisted_catalog_segment_opens_without_a_benchmark_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any segment of a known dataset replays from the catalog with that dataset's VIO configuration; no settings entry is needed."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    with pytest.raises(_Opened):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=UNLISTED, catalog=CATALOG))
    assert seen == [CatalogSegment(url=CATALOG, dataset_name="msd-g2", segment_id=UNLISTED)]


def test_an_unlisted_segment_uses_the_default_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlisted segments use the catalog and their dataset configuration."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    with pytest.raises(_Opened):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=UNLISTED))
    assert seen == [CatalogSegment(CATALOG, "msd-g2", UNLISTED)]


def test_a_catalog_segment_of_an_unknown_dataset_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dataset the settings has no VIO config for cannot be replayed, catalog or not."""
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    with pytest.raises(ValueError, match="'msd-nowhere' has no runtime settings"):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment="msd-nowhere__X__Y", catalog=CATALOG))
    assert seen == []


@pytest.mark.parametrize("with_gt", [False, True])
def test_explicit_local_recording_pair_is_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_gt: bool) -> None:
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    base: Path = tmp_path / "base.rrd"
    truth: Path | None = tmp_path / "truth.rrd" if with_gt else None
    with pytest.raises(_Opened):
        main(Config(rr_config=RerunTyroConfig(headless=True), rrd=base, gt_rrd=truth))
    assert seen == [LocalSegment(base, truth)]


def test_unlisted_odyssey_opens_without_a_segment_row(settings: SlamConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[SegmentSource] = _capture_source(monkeypatch)
    identifier: str = "msd-odyssey__MOO_others__MOO15_seated_screen"
    with pytest.raises(_Opened):
        main(Config(rr_config=RerunTyroConfig(headless=True), segment=identifier))
    assert seen == [CatalogSegment(settings.catalog_url, "msd-odyssey", identifier)]
