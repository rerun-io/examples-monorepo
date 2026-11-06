"""Readers of upstream's artifact formats that need no npz (the PLY), and the VSLAM-LAB subprocess environment."""

from pathlib import Path

import numpy as np
import pytest

from metrictrail.apis.vslamlab_eval import Config, vslamlab_command, vslamlab_environment
from metrictrail.artifacts import ColoredCloud, read_binary_ply, subsample_cloud


def test_read_binary_ply_upstream_layout(tmp_path: Path) -> None:
    header: bytes = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    vertices = np.array(
        [(1.0, 2.0, 3.0, 10, 20, 30), (-1.0, 0.5, 9.0, 255, 0, 1)],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    (tmp_path / "cloud.ply").write_bytes(header + vertices.tobytes())
    cloud: ColoredCloud = read_binary_ply(tmp_path / "cloud.ply")
    np.testing.assert_array_equal(cloud.points_xyz, [[1.0, 2.0, 3.0], [-1.0, 0.5, 9.0]])
    assert cloud.colors_rgb.tolist() == [[10, 20, 30], [255, 0, 1]]
    assert subsample_cloud(cloud, 1).points_xyz.shape == (1, 3)


def test_read_binary_ply_refuses_other_layouts(tmp_path: Path) -> None:
    (tmp_path / "ascii.ply").write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n")
    with pytest.raises(ValueError, match="binary little-endian"):
        read_binary_ply(tmp_path / "ascii.ply")


def test_vslamlab_subprocess_gets_a_private_evo_home(tmp_path: Path) -> None:
    config = Config(runs=[tmp_path], dataset="rgbdtum", out_dir=tmp_path / "out", vslamlab_dir=tmp_path / "VSLAM-LAB")
    parent: dict[str, str] = {"HOME": "/home/someone", "PIXI_PROJECT_ROOT": "/repo", "CONDA_PREFIX": "/env", "PATH": "/bin"}
    env: dict[str, str] = vslamlab_environment(config, parent, tmp_path / "evo_home")
    assert env["HOME"] == str(tmp_path / "evo_home")
    assert env["RATTLER_CACHE_DIR"] == "/home/someone/.cache/rattler"  # the user's package cache, not the private HOME's
    assert not any(key.startswith(("PIXI_", "CONDA_")) for key in env)
    assert env["PYTHONPATH"] == str((tmp_path / "VSLAM-LAB").resolve())
    command: list[str] = vslamlab_command(config, tmp_path / "exp", ["rgbd_dataset_freiburg1_desk"])
    assert command[:7] == ["pixi", "run", "--frozen", "--manifest-path", str((tmp_path / "VSLAM-LAB" / "pixi.toml").resolve()), "-e", "vslamlab"]
    assert Path(command[8]).is_file() and command[8].endswith("vslamlab_driver/evaluate_sequences.py")
