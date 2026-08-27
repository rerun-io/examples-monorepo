"""Dataset plugin surface: one config dataclass + one dataset class per dataset.

The config/``setup()`` split is simplecv's ``InstantiateConfig`` idiom (see
``simplecv/configs/base_config.py``): tyro parses the config, ``setup()``
instantiates the dataset that owns the ``download`` / ``discover`` /
``convert`` verbs. dataforge keeps its own copy of the idiom so the dataset
surface does not inherit simplecv's sequence-loader machinery.

``discover()`` is the single traversal of the raw tree: it returns identity **and**
the source location it was derived from, so ``convert`` never has to invert an
identity back into a path. ``SourceT`` is whatever a dataset needs to carry from
discovery to conversion (a directory, or a small record of parsed path parts).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import rerun.blueprint as rrb

from dataforge.identity import SequenceIdentity


@dataclass
class DataforgeDatasetConfig:
    """Base config for a dataforge dataset; ``setup()`` builds the dataset."""

    command: ClassVar[str]
    """CLI subcommand of the dataset, and its key in the registry."""

    _target: type
    """Dataset class instantiated by ``setup()``."""

    @property
    def name(self) -> str:
        """Catalog dataset name and the ``dataset`` part of every ``SequenceIdentity``.

        The command itself for a fixed dataset; a dataset that serves many
        corpora from one command (wildcap) derives a per-instance name.
        """
        return self.command

    def setup(self) -> DataforgeDataset:
        """Instantiate the dataset this config describes."""
        dataset: DataforgeDataset = self._target(self)
        return dataset


ConfigT = TypeVar("ConfigT", bound=DataforgeDatasetConfig)
"""Config type a concrete dataset is parameterized by (simplecv's sequence-loader idiom)."""

SourceT = TypeVar("SourceT")
"""Raw-tree handle ``discover()`` hands to ``convert()`` (an episode dir, a parsed segment record, …)."""


class DataforgeDataset(Generic[ConfigT, SourceT], ABC):
    """A dataset dataforge can download, enumerate, and convert to base-layer rrds."""

    def __init__(self, config: ConfigT) -> None:
        self.config: ConfigT = config

    @abstractmethod
    def download(self) -> None:
        """Fetch (or verify) the raw corpus this dataset converts from."""

    @abstractmethod
    def discover(self) -> list[tuple[SequenceIdentity, SourceT]]:
        """Walk the raw tree once, pairing every convertible sequence with its source."""

    def sequences(self) -> list[SequenceIdentity]:
        """Identities of every convertible sequence found on disk."""
        return [identity for identity, _ in self.discover()]

    @abstractmethod
    def convert(self, identity: SequenceIdentity, source: SourceT, *, force: bool) -> Path:
        """Write the base-layer rrd for one discovered sequence and return its path."""

    @abstractmethod
    def default_blueprint(self) -> rrb.Blueprint:
        """Dataset-wide default blueprint, registered on the catalog dataset.

        Per-recording blueprints are embedded at convert; this is the "dataset
        default at register" half of the design. A dataset that cannot produce
        one (e.g. a corpus-derived layout with no readable captures) raises
        rather than degrading to viewer heuristics.
        """

    @abstractmethod
    def table_blueprint(self) -> rrb.Blueprint:
        """Lightweight segment-table preview card, registered with ``segment_table=True``.

        The viewer renders every visible dataset row through this blueprint at
        once ("Table cards and blueprints" experimental setting), so it must
        stay cheap — cards decode exactly one stream (the front-stereo pane);
        everything else is excluded rather than hidden. Mandatory for the same
        reason: without it, table cards fall back to viewer heuristics that
        decode every stream of every visible row.
        """
