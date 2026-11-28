"""``pipelines-person-to-4d {plan,run,show}``: the person_to_4d flow on one person video.

``plan`` prints each task's command. ``run`` runs every step from the first, every time: nothing is skipped and nothing
is partial. ``show`` prints the run record of each step's layer from the catalog. Set ``FDANYONE_MODEL_DIR`` to the
4DAnyone weights (``/mnt/nas/ssd-8tb/fdanyone-rerun/models`` on our hosts); otherwise the first task downloads ~21 GB.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import tyro

from pipelines.apis.actions import WORKSPACE_MANIFEST, Action, act
from pipelines.person_to_4d import RUNS_SCRATCH, flow_steps, views_segment
from pipelines.runner import Launch


@dataclass
class Config:
    """Plan, run or show the person_to_4d flow on one video."""

    action: tyro.conf.Positional[Action]
    """``plan``: print the tasks; ``run``: run every step, in order; ``show``: print each layer's run record."""
    video: Path
    """The person video: one person, full or upper body, mild camera motion, at least 121 frames."""
    clip: str
    """Short name of the clip (lower-case letters, digits and dashes): the views' segment is ``<clip>__v06-seed<seed>``."""
    seed: int = 42
    """4DAnyone's noise seed."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """The catalog every step registers on; the server must read the layer files at their paths."""
    output_root: Path = Path("data/pipelines/person_to_4d")
    """Where the tasks write: scratch run directories, layer files, training runs."""
    train_steps: int = 25_000
    """TRiGS steps (the app's 25,000; fewer for a short proof)."""
    logs: Path | None = None
    """Directory of the task logs; ``<output_root>/logs/<segment>/<UTC start>`` by default."""
    gpu_lock: Path | None = Path("/tmp/gpu-5090.lock")
    """Held for each task; ``None`` runs without it."""
    memory_max: str | None = "64G"
    """Memory cap of each task's systemd scope; ``None`` runs without one."""
    pixi: str = "pixi"
    """The pixi executable that starts each task."""


def main(config: Config) -> None:
    """Plan, run or show the flow.

    Raises:
        SystemExit: If ``run`` stops at a failed step (its exit code).
    """
    segment: str = views_segment(config.clip, config.seed)
    logs: Path = config.logs or config.output_root.resolve() / "logs" / segment / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if config.action == "run":
        (config.output_root.resolve() / RUNS_SCRATCH).mkdir(parents=True, exist_ok=True)
    act(
        config.action,
        flow_steps(video=config.video, clip=config.clip, seed=config.seed, catalog_url=config.catalog_url, output_root=config.output_root,
                   train_steps=config.train_steps),
        Launch(manifest=WORKSPACE_MANIFEST, pixi=config.pixi, gpu_lock=config.gpu_lock, memory_max=config.memory_max),
        catalog_url=config.catalog_url,
        logs=logs,
        title=f"person_to_4d on {config.video.name} ({segment})",
    )
