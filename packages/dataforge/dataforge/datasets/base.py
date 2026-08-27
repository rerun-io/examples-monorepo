"""Dataset plugin surface: one config dataclass + one dataset class per dataset.

The config/``setup()`` split is simplecv's ``InstantiateConfig`` idiom (see
``simplecv/configs/base_config.py``): tyro parses the config, ``setup()``
instantiates the dataset that owns the ``download`` / ``sequences`` /
``convert`` verbs. dataforge keeps its own copy of the idiom so the dataset
surface does not inherit simplecv's sequence-loader machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import rerun.blueprint as rrb

from dataforge.identity import SequenceIdentity


@dataclass
class DataforgeDatasetConfig:
    """Base config for a dataforge dataset; ``setup()`` builds the dataset."""

    _target: type
    """Dataset class instantiated by ``setup()``."""

    def setup(self) -> DataforgeDataset[DataforgeDatasetConfig]:
        """Instantiate the dataset this config describes."""
        dataset: DataforgeDataset[DataforgeDatasetConfig] = self._target(self)
        return dataset


ConfigT = TypeVar("ConfigT", bound=DataforgeDatasetConfig)
"""Config type a concrete dataset is parameterized by (simplecv's sequence-loader idiom)."""


class DataforgeDataset(Generic[ConfigT], ABC):
    """A dataset dataforge can download, enumerate, and convert to base-layer rrds."""

    def __init__(self, config: ConfigT) -> None:
        self.config: ConfigT = config

    @abstractmethod
    def download(self) -> None:
        """Fetch (or verify) the raw corpus this dataset converts from."""

    @abstractmethod
    def sequences(self) -> list[SequenceIdentity]:
        """Enumerate every convertible sequence found on disk."""

    @abstractmethod
    def convert(self, identity: SequenceIdentity, *, force: bool) -> Path:
        """Write the base-layer rrd for one sequence and return its path."""

    def default_blueprint(self) -> rrb.Blueprint | None:
        """Dataset-wide default blueprint, registered on the catalog dataset.

        Per-recording blueprints are embedded at convert; this is the "dataset
        default at register" half of the design. ``None`` means the dataset has
        no sensible corpus-wide layout.
        """
        return None
