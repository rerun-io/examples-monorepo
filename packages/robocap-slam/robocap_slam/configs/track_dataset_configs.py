"""Dataset registry with tyro subcommand generation.

To add a new dataset, add one entry to ``track_dataset_defaults``.
tyro automatically generates CLI subcommands from the dict keys.

When the registry has only one entry, ``subcommand_type_from_defaults``
in tyro 0.9.x asserts ``len(defaults) >= 2`` and refuses to build a
``Union``. We avoid the assertion (and the noisy placeholder subcommand
it would otherwise force) by falling back to the single config's type
directly when the registry is small.
"""

from dataclasses import dataclass

import tyro

from robocap_slam.data.base import BaseTrackDatasetConfig
from robocap_slam.data.robocap import RobocapTrackConfig


@dataclass
class _TypeCheckOnlyTrackDatasetUnion(BaseTrackDatasetConfig):
    """Type-checker-only stand-in for the runtime ``Union`` tyro builds.

    We use a ``TYPE_CHECKING``-style branch (always-false at runtime via
    a type alias) to expose ``BaseTrackDatasetConfig`` to static type
    checkers; the runtime value is computed below.
    """


track_dataset_defaults: dict[str, BaseTrackDatasetConfig] = {
    "robocap": RobocapTrackConfig(),
}


def _build_track_dataset_union() -> type[BaseTrackDatasetConfig]:
    """Pick the right tyro construction based on registry size.

    With ≥2 entries, ``subcommand_type_from_defaults`` builds a Union of
    the dataset configs and tyro renders one subcommand per key. With a
    single entry tyro can't build a Union, so we hand back the lone
    config's concrete type — tyro then uses it directly without any
    subcommand layer (no fake `_placeholder` choice in `--help`).
    """
    if len(track_dataset_defaults) >= 2:
        return tyro.extras.subcommand_type_from_defaults(track_dataset_defaults, prefix_names=False)
    only_default = next(iter(track_dataset_defaults.values()))
    return type(only_default)


TrackDatasetUnion: type[BaseTrackDatasetConfig] = _build_track_dataset_union()
AnnotatedTrackDatasetUnion = tyro.conf.OmitSubcommandPrefixes[TrackDatasetUnion]
