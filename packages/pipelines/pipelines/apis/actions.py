"""The three actions of every flow's command: ``plan`` prints each task's command, ``run`` runs every step from the first,
every time (nothing is skipped and nothing is partial), ``show`` prints the run record of each step's layer."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias

from rerun.catalog import CatalogClient
from simplecv.run_record import RunRecord, read_run_record

from pipelines.runner import Launch, run_flow, task_command
from pipelines.step import Step

Action: TypeAlias = Literal["plan", "run", "show"]

WORKSPACE_MANIFEST: Path = Path(__file__).resolve().parents[4] / "pixi.toml"
"""The workspace ``pixi.toml`` holding every stage's task."""


def act(action: Action, steps: Sequence[Step], launch: Launch, *, catalog_url: str, logs: Path, title: str) -> None:
    """Plan, run or show ``steps``.

    Args:
        action: What to do.
        steps: The flow's steps.
        launch: How each task's process is started.
        catalog_url: The catalog the steps publish on.
        logs: Directory of the task logs of a ``run``.
        title: The flow and its input, for the closing line of a ``run``.

    Raises:
        SystemExit: If ``run`` stops at a failed step (its exit code).
    """
    if action == "plan":
        for step in steps:
            for task in step.tasks:
                print(f"{step.name} -> layer {step.layer}: {' '.join(task_command(task, launch))}")
        return
    client: CatalogClient = CatalogClient(catalog_url)
    if action == "show":
        for step in steps:
            try:
                record: RunRecord = read_run_record(client.get_dataset(name=step.dataset), step.segment, step.layer)
            except ValueError:
                print(f"{step.name} -> layer {step.layer}: no run record")
                continue
            dirty: str = f" dirty {record.git_dirty_sha256[:12]}" if record.git_dirty_sha256 else ""
            print(f"{step.name} -> layer {step.layer}: {record.started_at}, {record.wall_s:.0f} s, {record.git_commit[:12]}{dirty}, "
                  f"env {record.pixi_environment}, {record.host}\n    {record.command}")
        return
    code: int = run_flow(steps, launch, client=client, logs=logs)
    if code != 0:
        raise SystemExit(code)
    print(f"{title}: {len(steps)} steps published")
