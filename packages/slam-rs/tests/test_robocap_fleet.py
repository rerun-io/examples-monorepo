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
