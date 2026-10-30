import numpy as np

from dataforge.quest_joint_poses import read_quest_joint_poses


def test_reader_preserves_all_26_joint_orientations(tmp_path):
    source = tmp_path / "left_hand_poses.csv"
    names = ["ts_ns"] + [f"joint_{j}_{axis}" for j in range(26) for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    row = ["42"] + [str(v) for j in range(26) for v in (j, 1, 2, 0, 0, 0.6, 0.8)]
    source.write_text(",".join(names) + "\n" + ",".join(row) + "\n")
    data = read_quest_joint_poses(source)
    assert data.positions.shape == (1, 26, 3)
    np.testing.assert_array_equal(data.times_ns, [42])
    np.testing.assert_allclose(data.orientations[0, 25], [0, 0, 0.6, 0.8])


def test_position_only_and_untracked_rows_remain_missing(tmp_path):
    source = tmp_path / "right_hand_poses.csv"
    names = ["ts_ns"] + [f"joint_{j}_{axis}" for j in range(26) for axis in ("x", "y", "z")]
    source.write_text(",".join(names) + "\n" + ",".join(["9"] + ["0"] * 78) + "\n")
    data = read_quest_joint_poses(source)
    assert data.orientations is None
    assert np.isnan(data.positions).all()
