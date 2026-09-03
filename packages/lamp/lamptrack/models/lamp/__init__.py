"""Tyro registry for the released LAMP tracker configuration."""

from typing import TYPE_CHECKING

import tyro

from lamptrack.models.lamp.predictor import Frameset, LampConfig, LampStep, LampTimings, LampTracker, PersonState

lamp_tracker_defaults: dict[str, LampConfig] = {"lamp": LampConfig()}

if TYPE_CHECKING:
    LampTrackerUnion = LampConfig
else:
    LampTrackerUnion = tyro.extras.subcommand_type_from_defaults(lamp_tracker_defaults, prefix_names=False)

AnnotatedLampTrackerUnion = tyro.conf.OmitSubcommandPrefixes[LampTrackerUnion]

__all__ = (
    "AnnotatedLampTrackerUnion",
    "Frameset",
    "LampConfig",
    "LampStep",
    "LampTimings",
    "LampTracker",
    "LampTrackerUnion",
    "PersonState",
    "lamp_tracker_defaults",
)
