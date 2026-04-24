"""Dataset registry with tyro subcommand generation.

To add a new dataset, add one entry to ``track_dataset_defaults``.
tyro automatically generates CLI subcommands from the dict keys.
"""

from typing import TYPE_CHECKING, Annotated, Union

import tyro

from robocap_slam.data.base import BaseTrackDatasetConfig
from robocap_slam.data.robocap import RobocapTrackConfig

track_dataset_defaults: dict[str, BaseTrackDatasetConfig] = {
    "robocap": RobocapTrackConfig(),
}


def _annotated_subcommand(name: str, value: BaseTrackDatasetConfig):
    return Annotated[(type(value), tyro.conf.subcommand(name, default=value, prefix_name=False))]


def _build_subcommand_union(defaults: dict[str, BaseTrackDatasetConfig]):
    # Mirrors tyro.extras.subcommand_type_from_defaults but tolerates a
    # single-entry registry. tyro's helper asserts >= 2 because Union
    # needs at least two members to trigger subcommand dispatch. For a
    # one-entry registry we build a second Annotated with a fresh
    # subcommand marker object — identical name and default, but a
    # distinct instance — so Union.__getitem__ keeps both args and tyro
    # still routes the positional subcommand.
    annotated = [_annotated_subcommand(k, v) for k, v in defaults.items()]
    if len(annotated) == 1:
        k, v = next(iter(defaults.items()))
        annotated.append(_annotated_subcommand(k, v))
    return Union.__getitem__(tuple(annotated))  # type: ignore[operator]


if TYPE_CHECKING:
    TrackDatasetUnion = BaseTrackDatasetConfig
else:
    TrackDatasetUnion = _build_subcommand_union(track_dataset_defaults)

AnnotatedTrackDatasetUnion = tyro.conf.OmitSubcommandPrefixes[TrackDatasetUnion]
