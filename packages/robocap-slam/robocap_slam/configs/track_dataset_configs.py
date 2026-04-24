"""Dataset registry with tyro subcommand generation.

To add a new dataset, add one entry to ``track_dataset_defaults``.
tyro automatically generates CLI subcommands from the dict keys.
"""

from typing import TYPE_CHECKING

import tyro

from robocap_slam.data.base import BaseTrackDatasetConfig
from robocap_slam.data.robocap import RobocapTrackConfig

track_dataset_defaults: dict[str, BaseTrackDatasetConfig] = {
    "robocap": RobocapTrackConfig(),
}

if TYPE_CHECKING:
    TrackDatasetUnion = BaseTrackDatasetConfig
elif len(track_dataset_defaults) >= 2:
    TrackDatasetUnion = tyro.extras.subcommand_type_from_defaults(track_dataset_defaults, prefix_names=False)
else:
    # tyro.extras.subcommand_type_from_defaults requires >= 2 entries.
    # With a single dataset there's nothing to choose between, so skip
    # the subcommand layer and use the config type directly.
    TrackDatasetUnion = type(next(iter(track_dataset_defaults.values())))

AnnotatedTrackDatasetUnion = tyro.conf.OmitSubcommandPrefixes[TrackDatasetUnion]
