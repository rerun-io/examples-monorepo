"""Dataset registry with tyro subcommand generation.

To add a new dataset, add one entry to ``track_dataset_defaults``.
tyro automatically generates CLI subcommands from the dict keys.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import tyro

from robocap_slam.data.base import BaseTrackDatasetConfig
from robocap_slam.data.robocap import RobocapTrackConfig


@dataclass
class _PlaceholderTrackConfig(BaseTrackDatasetConfig):
    """Stub subcommand to satisfy tyro's ≥2-defaults requirement.

    ``tyro.extras.subcommand_type_from_defaults`` builds a ``Union`` and
    asserts ``len(defaults) >= 2`` (tyro 0.9.x). With only the ``robocap``
    dataset registered the assertion fires at module import time, so this
    placeholder exists purely to pad the registry to two entries. Delete
    it the moment a real second dataset is added.
    """

    # `_target` is required by InstantiateConfig; we override setup() so
    # the target is never actually instantiated.
    _target: type = field(default_factory=lambda: _PlaceholderTrackConfig)

    def setup(self, *args, **kwargs):
        raise NotImplementedError(
            "_placeholder is not a real dataset — it only exists so tyro can "
            "build a subcommand Union. Pick the `robocap` subcommand instead."
        )


track_dataset_defaults: dict[str, BaseTrackDatasetConfig] = {
    "robocap": RobocapTrackConfig(),
    # See `_PlaceholderTrackConfig` above. Remove this entry when a real
    # second dataset lands.
    "_placeholder": _PlaceholderTrackConfig(),
}

if TYPE_CHECKING:
    TrackDatasetUnion = BaseTrackDatasetConfig
else:
    TrackDatasetUnion = tyro.extras.subcommand_type_from_defaults(track_dataset_defaults, prefix_names=False)

AnnotatedTrackDatasetUnion = tyro.conf.OmitSubcommandPrefixes[TrackDatasetUnion]
