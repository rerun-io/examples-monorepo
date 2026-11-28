"""``fdanyone-log-run``: log a finished 4DAnyone run directory to Rerun (no GPU), and with ``--train`` its 4D person."""

from dataclasses import dataclass, field
from pathlib import Path

from open4dgs.trigs_adapter import require_fused_ssim
from simplecv.rerun_log_utils import RerunTyroConfig

from fdanyone_rerun.assets import model_dir
from fdanyone_rerun.outputs import RunMetadata
from fdanyone_rerun.person4d import PersonConfig, train_person
from fdanyone_rerun.rerun_log import log_run


@dataclass
class Config:
    """Log a run directory (ours or upstream's ``inference.py`` output) with the default layout."""

    rr_config: RerunTyroConfig
    """Viewer / save options; pass --rr-config.headless without a display. No default: building one calls rr.init and
    spawns or connects a viewer, so tyro must build it once, from the flags."""

    output_dir: Path
    """The run directory (videos/, cameras.json, metadata.json, gvhmr/)."""
    video: Path | None = None
    """The input video: the run's canonical clip is cut from it again (from the run's start time) for the source camera."""
    train: bool = False
    """Also train the run's 4D person and log it as Animated SOG (``world/person/splats``; needs the GPU)."""
    person: PersonConfig = field(default_factory=PersonConfig)
    """The person's training, with ``--train``."""


def main(config: Config) -> RunMetadata:
    """Log the run (and, with ``--train``, its 4D person)."""
    if config.train:
        require_fused_ssim()  # before logging and mattes, which would otherwise run first
    output_dir: Path = config.output_dir.expanduser().resolve()
    recording = config.rr_config.rec_stream
    metadata: RunMetadata = log_run(output_dir, config.video, model_dir(), recording)
    if config.train:
        train_person(output_dir, config.person, recording)
    return metadata
