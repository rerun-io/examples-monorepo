"""A child pixi task's environment and log: the parent env's activation stays out, the log stays readable."""

import os
from pathlib import Path

from simplecv.process_utils import child_environment, run_command


def test_a_child_does_not_inherit_the_parents_env_activation() -> None:
    parent: dict[str, str] = {
        "CONDA_PREFIX": "/ws/.pixi/envs/runner", "PATH": "/ws/.pixi/envs/runner/bin:/usr/bin", "LD_LIBRARY_PATH": "/ws/.pixi/envs/runner/lib",
        "PIXI_DEV_MODE": "1", "HOME": "/home/u",
    }
    child: dict[str, str] = child_environment(parent)
    assert child["PATH"] == "/usr/bin" and "LD_LIBRARY_PATH" not in child and "PIXI_DEV_MODE" not in child and child["HOME"] == "/home/u"


def test_the_task_log_is_readable_even_with_a_restrictive_umask(tmp_path: Path) -> None:
    log_path: Path = tmp_path / "task.log"
    previous: int = os.umask(0o777)
    try:
        assert run_command(["/bin/true"], {}, log_path) == 0
    finally:
        os.umask(previous)
    assert log_path.stat().st_mode & 0o777 == 0o644


def test_the_task_gets_its_overlay_and_its_output_lands_in_the_log(tmp_path: Path) -> None:
    log_path: Path = tmp_path / "task.log"
    assert run_command(["/bin/sh", "-c", 'echo "value=$TASK_VALUE"; exit 4'], {"TASK_VALUE": "7"}, log_path) == 4
    assert log_path.read_text() == "value=7\n"
