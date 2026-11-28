"""``pipelines-capture-to-splat {plan,run,show}``: the capture_to_splat flow on one registered base segment.

``plan`` prints each step's command. ``run`` runs every step from the first, every time: nothing is skipped and nothing
is partial. ``show`` prints the run record of each step's layer from the catalog. A single step is re-run from its run
record (``simplecv.run_record``), not through this command.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import tyro
from rerun.catalog import CatalogClient
from simplecv.run_record import RunRecord, read_run_record

from pipelines.capture_to_splat import InputProfile, flow_steps
from pipelines.runner import Launch, run_flow, step_command
from pipelines.step import Step

WORKSPACE_MANIFEST: Path = Path(__file__).resolve().parents[4] / "pixi.toml"
"""The workspace ``pixi.toml`` holding every stage's task."""


@dataclass
class Config:
    """Plan, run or show the capture_to_splat flow on one segment."""

    action: tyro.conf.Positional[Literal["plan", "run", "show"]]
    """``plan``: print the steps; ``run``: run them all, in order; ``show``: print each layer's run record."""
    profile: InputProfile
    """The input profile: ``insta360`` (lens masks, then masked SLAM) or ``robocap`` (SLAM)."""
    segment: str
    """The base's segment id, registered on the catalog in the dataset named by its prefix."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """The catalog holding the base; every step reads from it and registers its layer on it."""
    output_root: Path = Path("data/pipelines")
    """Where each step writes its layer file (``<output_root>/<layer>/<segment>.rrd``); the catalog server must read it
    at the same path."""
    logs: Path | None = None
    """Directory of the step logs; ``<output_root>/logs/<segment>/<UTC start>`` by default."""
    gpu_lock: Path | None = Path("/tmp/gpu-5090.lock")
    """Held for each step; ``None`` runs without it."""
    memory_max: str | None = "64G"
    """Memory cap of each step's systemd scope; ``None`` runs without one."""
    pixi: str = "pixi"
    """The pixi executable that starts each step."""


def main(config: Config) -> None:
    """Plan, run or show the flow.

    Raises:
        SystemExit: If ``run`` stops at a failed step (its exit code).
    """
    steps: tuple[Step, ...] = flow_steps(config.profile, segment=config.segment, catalog_url=config.catalog_url, output_root=config.output_root)
    launch: Launch = Launch(manifest=WORKSPACE_MANIFEST, pixi=config.pixi, gpu_lock=config.gpu_lock, memory_max=config.memory_max)
    if config.action == "plan":
        for step in steps:
            print(f"{step.name} -> layer {step.layer}: {' '.join(step_command(step, launch))}")
        return
    client: CatalogClient = CatalogClient(config.catalog_url)
    if config.action == "show":
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
    logs: Path = config.logs or config.output_root.resolve() / "logs" / config.segment / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    code: int = run_flow(steps, launch, client=client, logs=logs)
    if code != 0:
        raise SystemExit(code)
    print(f"capture_to_splat ({config.profile}) on {config.segment}: {len(steps)} steps published")
