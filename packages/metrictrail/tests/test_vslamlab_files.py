"""Unit tests for VSLAM-LAB file formats (round trips in a temporary directory)."""

from pathlib import Path

import numpy as np
import pytest

from metrictrail.eval.vslamlab_files import (
    AteCsvRow,
    PoseTable,
    RgbCsv,
    read_ate_csv,
    read_pose_csv,
    read_rgb_csv,
    trajectory_csv_name,
    write_pose_csv,
)


def test_pose_csv_round_trip_keeps_integer_stamps_and_sorts_by_time(tmp_path: Path) -> None:
    table = PoseTable(
        timestamps_ns=np.asarray([1305031102211214080, 1305031102175303936], dtype=np.int64),
        xyz_m=np.asarray([[1.0, 2.0, 3.0], [0.123456789, -4.0, 5.5]]),
        quat_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]]),
    )
    path: Path = tmp_path / trajectory_csv_name(0)
    write_pose_csv(path, table)
    assert path.name == "00000_KeyFrameTrajectory.csv"
    assert path.read_text().splitlines()[0] == "ts (ns),tx (m),ty (m),tz (m),qx,qy,qz,qw"
    back: PoseTable = read_pose_csv(path)
    assert back.timestamps_ns.tolist() == [1305031102175303936, 1305031102211214080]
    np.testing.assert_allclose(back.xyz_m, table.xyz_m[::-1], atol=1e-9)
    np.testing.assert_allclose(back.quat_xyzw, table.quat_xyzw[::-1], atol=1e-12)


def test_read_pose_csv_refuses_a_foreign_header(tmp_path: Path) -> None:
    path: Path = tmp_path / "groundtruth.csv"
    path.write_text("timestamp,x,y,z,qx,qy,qz,qw\n0,0,0,0,0,0,0,1\n")
    with pytest.raises(ValueError, match="expected header"):
        read_pose_csv(path)


def test_read_rgb_csv(tmp_path: Path) -> None:
    path: Path = tmp_path / "rgb.csv"
    path.write_text("ts_rgb_0 (ns),path_rgb_0,ts_depth_0 (ns),path_depth_0\n1000000000000,rgb_0/000000.png,1000000000000,depth_0/000000.png\n")
    rgb: RgbCsv = read_rgb_csv(path)
    assert rgb.rows[0].timestamp_ns == 1000000000000
    assert rgb.rows[0].path == "rgb_0/000000.png"
    assert len(rgb.rows[0].cells) == len(rgb.header) == 4


def test_read_ate_csv_parses_evo_res_table_with_vslamlab_counts(tmp_path: Path) -> None:
    path: Path = tmp_path / "ate.csv"
    path.write_text(
        "traj_name,max,mean,median,min,rmse,sse,std,num_frames,num_tracked_frames,num_evaluated_frames\n"
        "00001_KeyFrameTrajectory.txt,0.1,0.02,0.018,0.001,0.0254,0.38,0.01,596,596,596\n"
    )
    rows: list[AteCsvRow] = read_ate_csv(path)
    assert rows == [AteCsvRow(exp_it=1, rmse_m=0.0254, num_frames=596, num_tracked_frames=596, num_evaluated_frames=596)]
