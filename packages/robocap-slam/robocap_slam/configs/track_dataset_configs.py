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
else:
    TrackDatasetUnion = tyro.extras.subcommand_type_from_defaults(track_dataset_defaults, prefix_names=False)

AnnotatedTrackDatasetUnion = tyro.conf.OmitSubcommandPrefixes[TrackDatasetUnion]
