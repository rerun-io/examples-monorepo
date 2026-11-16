"""``fdanyone-log-run``: log a finished 4DAnyone run directory to Rerun (no GPU)."""

from dataclasses import dataclass
from pathlib import Path

from simplecv.rerun_log_utils import RerunTyroConfig

from fdanyone_rerun.assets import model_dir
from fdanyone_rerun.outputs import RunMetadata
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


def main(config: Config) -> RunMetadata:
    """Log the run."""
    return log_run(config.output_dir.expanduser().resolve(), config.video, model_dir(), config.rr_config.rec_stream)
