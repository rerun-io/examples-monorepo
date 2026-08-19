"""Dataset registry: the keys become tyro subcommands on every dataforge verb."""

from __future__ import annotations

import tyro

from dataforge.datasets.base import DataforgeDatasetConfig
from dataforge.datasets.robocap import RobocapConfig

dataset_defaults: dict[str, DataforgeDatasetConfig] = {
    "robocap": RobocapConfig(),
}
"""Every dataset dataforge knows about, keyed by its CLI name."""


def _build_dataset_union() -> type[DataforgeDatasetConfig]:
    """Build the tyro subcommand union, tolerating a single-entry registry.

    ``subcommand_type_from_defaults`` asserts ``len(defaults) >= 2`` in tyro
    0.9.x (same workaround as ``robocap_slam.configs.track_dataset_configs``),
    so with one dataset we hand tyro the concrete config type instead — no
    placeholder subcommand shows up in ``--help``.
    """
    if len(dataset_defaults) >= 2:
        return tyro.extras.subcommand_type_from_defaults(dataset_defaults, prefix_names=False)
    only_default: DataforgeDatasetConfig = next(iter(dataset_defaults.values()))
    return type(only_default)


DatasetUnion: type[DataforgeDatasetConfig] = _build_dataset_union()
"""Config type every verb's ``Config.dataset`` field uses."""

AnnotatedDatasetUnion = tyro.conf.OmitSubcommandPrefixes[DatasetUnion]
"""``DatasetUnion`` without tyro's per-subcommand flag prefixes."""


def dataset_key(config: DataforgeDatasetConfig) -> str:
    """Registry key of a parsed dataset config (the catalog dataset name)."""
    for key, default in dataset_defaults.items():
        if isinstance(config, type(default)):
            return key
    raise ValueError(f"Config {type(config).__name__} is not in dataset_defaults")
