"""``fdanyone-log-run``: log a finished 4DAnyone run directory to Rerun (no GPU)."""

from dataclasses import dataclass, field
from pathlib import Path

from simplecv.rerun_log_utils import RerunTyroConfig

from fdanyone_rerun.assets import model_dir
from fdanyone_rerun.outputs import RunMetadata
from fdanyone_rerun.rerun_log import log_run


@dataclass
class Config:
    """Log a run directory (ours or upstream's ``inference.py`` output) with the default layout."""

    output_dir: Path
    """The run directory (videos/, cameras.json, metadata.json, gvhmr/)."""
    video: Path | None = None
    """The input video, to show it in the source camera; it lines up when the run started at 0 s."""
    rr_config: RerunTyroConfig = field(default_factory=lambda: RerunTyroConfig(application_id="fdanyone-rerun"))
    """Viewer / save options; pass --rr-config.headless without a display."""


def main(config: Config) -> RunMetadata:
    """Log the run."""
    return log_run(config.output_dir.expanduser().resolve(), config.video, model_dir(), config.rr_config.rec_stream)
