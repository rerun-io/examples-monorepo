"""``fdanyone-run``: generate novel views of the person in a video and stream the run into Rerun."""

from dataclasses import dataclass, field

from simplecv.rerun_log_utils import RerunTyroConfig
from taehv import TAEHV

from fdanyone_rerun.assets import model_dir
from fdanyone_rerun.core import RunConfig
from fdanyone_rerun.outputs import RunMetadata
from fdanyone_rerun.previews import load_preview_decoder
from fdanyone_rerun.stream import RunStreamer

PREVIEW_CHECKPOINT: str = "taehv/taew2_2.pth"
"""TAEW2.2 inside the weights directory."""


@dataclass
class Config:
    """Run 4DAnyone-Turbo on one video; the source, the body, the denoising previews and the views stream into Rerun."""

    run: RunConfig = field(default_factory=RunConfig)
    """What to generate."""
    previews: bool = True
    """Decode each denoising step's x_0 estimate with TAEW2.2 and log it (``previews/<NN>``)."""
    rr_config: RerunTyroConfig = field(default_factory=lambda: RerunTyroConfig(application_id="fdanyone-rerun"))
    """Viewer / save options; pass --rr-config.headless without a display."""


def main(config: Config) -> RunMetadata:
    """Run and stream all four phases."""
    decoder: TAEHV | None = load_preview_decoder(model_dir() / PREVIEW_CHECKPOINT, "cuda") if config.previews else None
    return RunStreamer(config.run, config.rr_config.rec_stream, decoder).run()
