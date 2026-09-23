"""A SLAM result augments the base rig on its original timeline."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rerun.chunk as rrc
from fixture_types import Rows, RowsReader

from slam_rs.catalog_layer import write_layer
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import Trajectory, empty_trajectory


@pytest.fixture
def run() -> SegmentRun:
    """Two finite poses on an absolute inertial clock."""
    return SegmentRun(
        estimate=Trajectory(
            t_ns=np.array([8_014_902_432, 9_014_902_432], dtype=np.int64),
            position_m=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
            quaternion_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        ),
        framesets=2, lost=0, wall_s=0.1, ground_truth=empty_trajectory(), median_tracker_ms=10.0, config_sha256="test-config",
    )


def test_layer_animates_the_base_rig_without_replacing_camera_data(tmp_path: Path, read_rows: RowsReader, run: SegmentRun) -> None:
    """Absolute inertial poses return to camera time; the result carries no camera statics."""
    path: Path = tmp_path / "result.rrd"
    write_layer(path, "session", run, clock_offset_ns=7_014_902_432, profile="fast", backend="cpu")
    rows: Rows = read_rows(path)
    assert [row.t_ns for row in rows["/world/rig_00"]] == [1_000_000_000, 2_000_000_000]
    np.testing.assert_allclose(rows["/world/rig_00"][-1].values["Transform3D:translation"], [[1, 2, 3]])
    np.testing.assert_allclose(rows["/world/rig_00"][-1].values["Transform3D:quaternion"], [[0, 0, 0, 1]])
    assert [row.t_ns for row in rows["/world/runs/slam_rs/trail"]] == [2_000_000_000]
    entities: set[str] = {chunk.entity_path for chunk in rrc.RrdReader(path).stream().collect().stream()}
    assert {"/world/runs/slam_rs/trajectory", "/world/runs/slam_rs/endpoints", "/world/runs/slam_rs"} <= entities
    assert not any("/cam_" in entity or "/imu_" in entity for entity in entities)


def test_incomplete_run_does_not_replace_a_previous_result(tmp_path: Path, run: SegmentRun) -> None:
    path: Path = tmp_path / "result.rrd"
    write_layer(path, "session", run, clock_offset_ns=0, profile="fast", backend="cpu")
    previous: bytes = path.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        write_layer(path, "session", replace(run, lost=1), clock_offset_ns=0, profile="fast", backend="cpu")
    assert path.read_bytes() == previous
