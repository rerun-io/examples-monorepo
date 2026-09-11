"""Synthetic scoring tests; no recording or external source is needed."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from slam_rs.apis import fleet_check
from slam_rs.apis.fleet_check import ClipResult, Config, main, measure
from slam_rs.catalog_feed import CatalogSegment
from slam_rs.machine import this_machine
from slam_rs.reference import Baseline, ReferenceManifest, ReferenceSegment
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import Trajectory, shift_clock


@pytest.mark.parametrize("wrong_dataset", [False, True])
def test_measure_rejects_mismatched_source_before_replay(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, wrong_dataset: bool
) -> None:
    segment: ReferenceSegment = manifest.segments[0]
    source: CatalogSegment = CatalogSegment(
        "stub://catalog",
        "other-dataset" if wrong_dataset else segment.dataset_name,
        segment.segment_id if wrong_dataset else "other-segment",
        has_ground_truth=True,
    )
    monkeypatch.setattr(fleet_check, "run_segment", lambda *_args, **_kwargs: pytest.fail("replayed mismatched source"))
    with pytest.raises(ValueError, match=f"source {source.dataset_name}/{source.segment_id}.*segment {segment.dataset_name}/{segment.segment_id}"):
        measure(manifest, segment, source=source)


@pytest.mark.parametrize("clock_offset", [0, 100_000_000_000])
def test_scoring_uses_ground_truth_and_rejects_wrong_clock(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock_offset: int
) -> None:
    truth: Trajectory = Trajectory(
        t_ns=np.arange(30, dtype=np.int64) * 10_000_000,
        position_m=np.random.default_rng(7).normal(size=(30, 3)),
        quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (30, 1)),
    )
    run: SegmentRun = SegmentRun(
        estimate=shift_clock(truth, clock_offset), framesets=30, lost=0, wall_s=1.0, config_sha256="a" * 64, ground_truth=truth, median_tracker_ms=2.0
    )
    monkeypatch.setattr(fleet_check, "resolve_catalog_segments", lambda sources, **_kwargs: tuple(replace(source, has_ground_truth=True) for source in sources))
    monkeypatch.setattr(fleet_check, "run_segment", lambda *_args, **_kwargs: run)
    reference: Baseline = manifest.segments[0].baseline[0]
    host_baseline: Baseline = replace(reference, lane="cpu", profile="fast", host=this_machine().hostname, gt_rmse_cm=3.0)
    segment: ReferenceSegment = replace(manifest.segments[0], baseline=(replace(host_baseline, host="reference-host"), host_baseline))
    result: ClipResult = measure(manifest, segment)
    assert result.baseline == host_baseline
    assert result.speed_gated
    assert result.measurement.associated == (30 if clock_offset == 0 else 0)
    assert bool(result.failures) == bool(clock_offset)
    output: Path = tmp_path / "new" / "fleet.json"
    config: Config = Config(segments=(manifest.segments[0].segment_id,), output_json=output)
    if clock_offset:
        with pytest.raises(SystemExit, match="associated"):
            main(config)
    else:
        main(config)
        assert result.measurement.gt_rmse_cm < 1e-10
        assert "no baseline" in replace(result, baseline=None).verdict
    assert list(json.loads(output.read_text())["clips"][0]) == [
        "segment_id", "framesets", "tracked", "lost", "gt_rmse_cm", "wall_s", "peak_rss_mb",
        "gt_allowed_cm", "baseline_gt_rmse_cm", "median_tracker_ms", "speed_gated", "verdict",
    ]
    assert "NaN" not in output.read_text()
    if clock_offset:
        assert json.loads(output.read_text())["clips"][0]["gt_rmse_cm"] is None
    assert json.loads(output.read_text())["config_sha256"] == {manifest.segments[0].dataset_name: "a" * 64}


def test_empty_selection_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no clip"):
        main(Config(segments=(), output_json=tmp_path / "empty.json"))


@pytest.mark.parametrize("bad_reference", [False, True])
def test_nonfinite_scoring_keeps_costs_and_reports_refusal(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, bad_reference: bool
) -> None:
    finite: Trajectory = Trajectory(np.arange(20, dtype=np.int64), np.zeros((20, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (20, 1)))
    bad: Trajectory = replace(finite, position_m=finite.position_m.copy())
    bad.position_m[3, 1] = np.nan
    run: SegmentRun = SegmentRun(finite if bad_reference else bad, 20, 0, 2.0, bad if bad_reference else finite, 4.0, "a" * 64)
    monkeypatch.setattr(fleet_check, "resolve_catalog_segments", lambda sources, **_kwargs: tuple(replace(source, has_ground_truth=True) for source in sources))
    monkeypatch.setattr(fleet_check, "run_segment", lambda *_args, **_kwargs: run)
    result: ClipResult = measure(manifest, manifest.segments[0])
    assert result.wall_s == 2.0
    assert result.measurement.median_tracker_ms == 4.0
    assert result.unscored is not None and "first at 3 ns" in result.unscored
    assert any(clause.startswith("scoring:") for clause in result.failures)
