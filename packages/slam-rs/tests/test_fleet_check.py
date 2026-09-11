"""Synthetic scoring tests; no recording or external source is needed."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from slam_rs.apis import fleet_check
from slam_rs.apis.fleet_check import ClipResult, Config, main, measure
from slam_rs.reference import ReferenceManifest
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import Trajectory, shift_clock


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
    monkeypatch.setattr(fleet_check, "check_scoring_inputs", lambda *_args: None)
    monkeypatch.setattr(fleet_check, "run_segment", lambda *_args, **_kwargs: run)
    result: ClipResult = measure(manifest, manifest.segments[0])
    assert result.gt_associated == (30 if clock_offset == 0 else 0)
    assert bool(result.failures) == bool(clock_offset)
    output: Path = tmp_path / "new" / "fleet.json"
    config: Config = Config(segments=(manifest.segments[0].segment_id,), output_json=output)
    if clock_offset:
        with pytest.raises(SystemExit, match="associated"):
            main(config)
    else:
        main(config)
        assert result.gt_rmse_cm < 1e-10
        assert "no baseline" in replace(result, baseline=None).verdict
    assert set(json.loads(output.read_text())["clips"][0]) == {
        "segment_id", "framesets", "tracked", "lost", "gt_rmse_cm", "wall_s", "peak_rss_mb",
        "gt_allowed_cm", "baseline_gt_rmse_cm", "median_tracker_ms", "speed_gated", "verdict",
    }
    assert json.loads(output.read_text())["config_sha256"] == {manifest.segments[0].dataset_name: "a" * 64}


def test_empty_selection_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no clip"):
        main(Config(segments=(), output_json=tmp_path / "empty.json"))
