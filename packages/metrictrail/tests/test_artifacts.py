"""The VSLAM-LAB subprocess environment."""

from pathlib import Path

from metrictrail.apis.vslamlab_eval import Config, vslamlab_command, vslamlab_environment


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
