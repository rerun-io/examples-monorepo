"""RoboCap regression agreement is reported without an accuracy gate."""

from pathlib import Path

import numpy as np
import pytest

from slam_rs.apis import robocap_fleet
from slam_rs.apis.robocap_fleet import Config, RobocapRow, measure
from slam_rs.machine import Machine
from slam_rs.reference import ReferenceManifest, RobocapSession
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import Trajectory, empty_trajectory, write_trajectory


def test_reference_csv_is_scored_without_a_ground_truth_gate(manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    truth: Trajectory = Trajectory(
        t_ns=np.arange(30, dtype=np.int64) * 10_000_000,
        position_m=np.random.default_rng(9).normal(size=(30, 3)),
        quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (30, 1)),
    )
    path: Path = tmp_path / "reference.csv"
    write_trajectory(path, truth)
    run: SegmentRun = SegmentRun(truth, 30, 0, 3.0, empty_trajectory(), 2.0, "a" * 64)
    monkeypatch.setattr(robocap_fleet, "run_robocap", lambda *_args, **_kwargs: run)
    measurement: tuple[RobocapRow, Trajectory] = measure(manifest, manifest.robocap.sessions[0], Config(reference_csv=path), Machine("test", "x86_64", "test", 1))
    row: RobocapRow = measurement[0]
    estimate: Trajectory = measurement[1]
    assert len(estimate) == 30
    assert row.reference_rmse_cm < 1e-8
    assert row.ms_per_frameset == 100.0
    assert row.realtime_factor_15fps == pytest.approx(2 / 3)
    assert row.unscored is None


@pytest.mark.slow
def test_catalog_robocap_regression_reference(manifest: ReferenceManifest) -> None:
    session: RobocapSession = manifest.robocap.session("s00000015")
    if session.reference_csv is None:
        pytest.skip("RoboCap reference has not been recorded")
    measurement: tuple[RobocapRow, Trajectory] = measure(manifest, session, Config(seconds=2.0), Machine("test", "x86_64", "test", 1))
    row: RobocapRow = measurement[0]
    estimate: Trajectory = measurement[1]
    assert len(estimate) > 10
    assert row.lost == 0
    assert row.unscored is None
    assert np.isfinite(row.reference_rmse_cm)


@pytest.mark.parametrize("failure", ["estimate", "reference", "clock", "no-reference"])
def test_scoring_refusal_preserves_outputs_and_cost(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    import json
    from dataclasses import replace

    truth: Trajectory = Trajectory(np.arange(20, dtype=np.int64), np.zeros((20, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (20, 1)))
    estimate: Trajectory = replace(truth, position_m=truth.position_m.copy())
    if failure in ("estimate", "no-reference"):
        estimate.position_m[3, 1] = np.nan
    elif failure == "reference":
        truth.position_m[3, 1] = np.inf
    else:
        estimate = replace(estimate, t_ns=estimate.t_ns + 100_000_000_000)
    reference: Path = tmp_path / "reference.csv"
    write_trajectory(reference, truth)
    output: Path = tmp_path / "output.json"
    run: SegmentRun = SegmentRun(estimate, 20, 0, 2.0, empty_trajectory(), 4.0, "a" * 64)
    if failure == "no-reference":
        manifest = replace(manifest, robocap=replace(manifest.robocap, sessions=(replace(manifest.robocap.sessions[0], reference_csv=None),)))
    monkeypatch.setattr(robocap_fleet, "load_manifest", lambda: manifest)
    monkeypatch.setattr(robocap_fleet, "run_robocap", lambda *_args, **_kwargs: run)
    with pytest.raises(SystemExit, match="associated" if failure == "clock" else "not finite"):
        robocap_fleet.main(Config(reference_csv=None if failure == "no-reference" else reference, output_json=output))
    payload: dict[str, object] = json.loads(output.read_text())
    assert payload["wall_s"] == 2.0
    assert payload["ms_per_frameset"] == 100.0
    assert payload["unscored"] is not None
    assert output.with_suffix(".csv").exists()
