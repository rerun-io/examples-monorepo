"""``fdanyone-run``: generate novel views of the person in a video and stream the run into Rerun."""

from dataclasses import dataclass, field

from simplecv.rerun_log_utils import RerunTyroConfig

from fdanyone_rerun.core import RunConfig
from fdanyone_rerun.outputs import RunMetadata
from fdanyone_rerun.person4d import PersonConfig
from fdanyone_rerun.stream import RunStreamer


@dataclass
class Config:
    """Run 4DAnyone-Turbo on one video; the source, the body, the denoising previews and the views stream into Rerun."""

    rr_config: RerunTyroConfig
    """Viewer / save options; pass --rr-config.headless without a display. No default: building one calls rr.init and
    spawns or connects a viewer, so tyro must build it once, from the flags."""

    run: RunConfig = field(default_factory=RunConfig)
    """What to generate."""
    previews: bool = True
    """Decode each denoising step's x_0 estimate with TAEW2.2 and log it (``previews/<NN>``)."""
    train: bool = False
    """Then train the 4D person on the views and log it as Animated SOG (``world/person/splats``)."""
    person: PersonConfig = field(default_factory=PersonConfig)
    """The person's training, with ``--train``."""


def main(config: Config) -> RunMetadata:
    """Run and stream all phases."""
    return RunStreamer(config.run, config.rr_config.rec_stream, config.previews, config.person if config.train else None).run()
