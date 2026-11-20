"""The slam-rs trajectory CSV door of `colsfm.apis.trajectory_layer`: the CSV's own video_time offset, quaternion order, row order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from colsfm.apis.trajectory_layer import read_slam_csv


def test_a_slam_rs_csv_is_shifted_onto_video_time_with_a_w_first_quaternion(tmp_path: Path) -> None:
    csv: Path = tmp_path / "slam_rs.csv"
    half: float = float(np.sqrt(0.5))
    csv.write_text(
        "#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n#video_time_offset_ns=1000000000\n"
        f"1000000123,1.0,2.0,3.0,1.0,0.0,0.0,0.0\n1033366123,4.0,5.0,6.0,{half},0.0,0.0,{half}\n"
    )

    times_ns, world_T_rig = read_slam_csv(csv)

    assert times_ns.tolist() == [123, 33_366_123]
    expected_first: np.ndarray = np.eye(4)
    expected_first[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(world_T_rig[0], expected_first)
    # (w, x, y, z) = (cos 45, 0, 0, sin 45) is +90 deg about z: it maps x onto y (a w-last read would not).
    np.testing.assert_allclose(world_T_rig[1, :3, :3] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(world_T_rig[1, :3, 3], [4.0, 5.0, 6.0])


def test_a_csv_that_does_not_state_its_video_time_offset_is_refused(tmp_path: Path) -> None:
    """colsfm does not guess slam-rs's export clock; an older export has to be written again."""
    csv: Path = tmp_path / "slam_rs.csv"
    csv.write_text("#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n1000000123,1.0,2.0,3.0,1.0,0.0,0.0,0.0\n")
    with pytest.raises(ValueError, match="video_time_offset_ns"):
        read_slam_csv(csv)
