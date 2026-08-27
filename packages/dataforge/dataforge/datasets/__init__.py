"""Dataset registry: the keys become tyro subcommands on every dataforge verb."""

from __future__ import annotations

import tyro

from dataforge.datasets.base import DataforgeDatasetConfig
from dataforge.datasets.robocap import RobocapConfig
from dataforge.datasets.selfcap import SelfcapConfig
from dataforge.datasets.wildcap import WildcapConfig

dataset_defaults: dict[str, DataforgeDatasetConfig] = {config.name: config for config in (RobocapConfig(), SelfcapConfig(), WildcapConfig())}
"""Every dataset dataforge knows about, keyed by its own ``name`` (the single source of truth)."""

DatasetUnion = tyro.extras.subcommand_type_from_defaults(dataset_defaults, prefix_names=False)
"""Config type every verb's ``Config.dataset`` field uses (one tyro subcommand per registry key).

Deliberately unannotated. The helper returns an ``Annotated[...] |
Annotated[...]`` union, so a ``type[DataforgeDatasetConfig]`` annotation makes
beartype's claw reject the global, while a ``TypeAlias`` annotation makes
pyrefly reject the call in an annotation position.
"""

AnnotatedDatasetUnion = tyro.conf.OmitSubcommandPrefixes[DatasetUnion]
"""``DatasetUnion`` without tyro's per-subcommand flag prefixes."""
