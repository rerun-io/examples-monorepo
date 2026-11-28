"""Start a pixi task of another environment as a child process: its environment and its log.

Used by the runners that start stage tasks (``selfcap_curate.apis.run``, ``pipelines.runner``).
"""

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path


def child_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """The environment of a nested ``pixi run``, without the activation of the env the parent runs in.

    pixi keeps what the new env does not set: the parent env's LD_LIBRARY_PATH would load its libraries into the task's
    env, and a test's PIXI_DEV_MODE flag would keep torch.compile off in the tasks.
    """
    prefix: str | None = parent.get("CONDA_PREFIX")
    child: dict[str, str] = {}
    for name, value in parent.items():
        entries: list[str] = value.split(os.pathsep)
        kept: list[str] = entries if prefix is None else [entry for entry in entries if prefix not in entry]
        if kept and name != "PIXI_DEV_MODE":
            child[name] = os.pathsep.join(kept)
    return child


def run_command(command: list[str], env: Mapping[str, str], log_path: Path) -> int:
    """Run one task with its output in ``log_path``, its environment the parent's without its activation plus ``env``;
    return its exit code."""
    with log_path.open("w") as log:
        # The NAS SSD client caches new files as mode 000; chmod repairs its cached attributes.
        log_path.chmod(0o644)
        completed: subprocess.CompletedProcess[bytes] = subprocess.run(
            command,
            env={**child_environment(os.environ), **env},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode
