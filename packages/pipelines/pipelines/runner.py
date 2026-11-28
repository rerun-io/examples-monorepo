"""Run a flow's steps in order, each as its own process: ``pixi run --manifest-path <m> --frozen -e <env> <task>``.

A step runs under the GPU lock and a memory cap, with its output in its own log. A step is done when its process exits
0 and the catalog holds its layer's run record from this run; anything else stops the flow, so no later step reads a
missing or stale layer. The runner keeps no state: the catalog layers and their run records are the only record of what
ran (``simplecv.run_record``).
"""

import shlex
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rerun.catalog import CatalogClient
from simplecv.process_utils import run_command
from simplecv.run_record import RunRecord, read_run_record

from pipelines.step import Step


@dataclass(frozen=True, slots=True)
class Launch:
    """How a step's process is started."""

    manifest: Path
    """The workspace ``pixi.toml``."""
    pixi: str = "pixi"
    """The pixi executable."""
    gpu_lock: Path | None = Path("/tmp/gpu-5090.lock")
    """Held (``flock -x``) for the whole step; None runs without it."""
    memory_max: str | None = "64G"
    """``MemoryMax`` of the step's systemd scope; None runs without one."""


def step_command(step: Step, launch: Launch) -> list[str]:
    """The command of one step: the pixi task, inside its memory scope, inside the GPU lock."""
    command: list[str] = [launch.pixi, "run", "--manifest-path", str(launch.manifest), "--frozen", "-e", step.environment, step.task, *step.arguments]
    if launch.memory_max is not None:
        command = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={launch.memory_max}", *command]
    if launch.gpu_lock is not None:
        command = ["flock", "-x", str(launch.gpu_lock), *command]
    return command


def last_line(log: Path) -> str:
    """The log's last line with a letter or digit, read line by line: a failed Python stage ends with its exception."""
    last: str = "its log is empty"
    with log.open(errors="replace") as lines:
        for line in lines:
            if any(character.isalnum() for character in line):
                last = line.strip()[:300]
    return last


def run_flow(steps: Sequence[Step], launch: Launch, *, client: CatalogClient, logs: Path) -> int:
    """Run every step in order, stopping at the first that fails or publishes no layer.

    Args:
        steps: The flow's steps.
        launch: How each step's process is started.
        client: The catalog the steps publish on; each step's run record is read from its own dataset and segment.
        logs: Directory of the step logs (``<nn>-<step>.log``).

    Returns:
        0 when every step published its layer, else the failed step's exit code (1 when it exited 0 without publishing).
    """
    logs.mkdir(parents=True, exist_ok=True)
    for index, step in enumerate(steps):
        command: list[str] = step_command(step, launch)
        log: Path = logs / f"{index:02d}-{step.name}.log"
        print(f"[{step.name}] {shlex.join(command)}\n[{step.name}] log {log}", flush=True)
        started: datetime = datetime.now(UTC)
        clock: float = time.monotonic()
        code: int = run_command(command, {}, log)
        if code != 0:
            print(f"[{step.name}] FAILED (exit {code}): {last_line(log)}", flush=True)
            return code
        try:
            record: RunRecord = read_run_record(client.get_dataset(name=step.dataset), step.segment, step.layer)
        except ValueError as error:
            print(f"[{step.name}] FAILED: exit 0 but {error}", flush=True)
            return 1
        if datetime.fromisoformat(record.started_at) < started:
            print(f"[{step.name}] FAILED: exit 0 but layer {step.layer}'s run record is from {record.started_at}, before this step", flush=True)
            return 1
        print(f"[{step.name}] {step.layer} published in {time.monotonic() - clock:.0f} s (the stage's own wall {record.wall_s:.0f} s)", flush=True)
    return 0
