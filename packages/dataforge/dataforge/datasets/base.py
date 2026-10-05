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
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import rerun as rr
import rerun.blueprint as rrb
from serde import serde

from dataforge import paths, writing
from dataforge.identity import SequenceIdentity
from dataforge.timing import SequenceTimer
from dataforge.writing import TableFields


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


@dataclass
class FrameLimitedConfig(DataforgeDatasetConfig):
    """Config of a dataset that can convert a prefix of each sequence as a preview."""

    frame_limit: int | None = None
    """Convert only the first N frames, into output_root/preview-first<N>/, so a preview never overwrites or skips a full conversion."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RemoteSequence:
    """One sequence as the source lists it, before anything is downloaded (``dataforge-download --list-remote``).

    Lets a caller with a small disk fetch, convert and prune in batches: download and convert with
    ``--sequences`` set to a batch of keys, then delete that batch's ``files``.
    """

    key: str
    """Value the dataset config's ``sequences`` accepts to download or convert just this sequence."""
    size_bytes: int
    """Bytes ``download()`` fetches for ``files``; shared files (calibration, models, annotations) are not counted."""
    files: tuple[str, ...]
    """Raw-root-relative paths only this sequence uses, safe to delete once it is converted; empty when the
    source packs several sequences into one archive."""


ConfigT = TypeVar("ConfigT", bound=DataforgeDatasetConfig)
"""Config type a concrete dataset is parameterized by (simplecv's sequence-loader idiom)."""

SourceT = TypeVar("SourceT")
"""Raw-tree handle ``discover()`` hands to ``convert()`` (an episode dir, a parsed segment record, …)."""


class DataforgeDataset(Generic[ConfigT, SourceT], ABC):
    """A dataset dataforge can download, enumerate, and convert to layer rrds."""

    layers: tuple[str, ...] = paths.LAYERS
    """Layers this dataset publishes, in loading order."""

    def __init__(self, config: ConfigT) -> None:
        self.config: ConfigT = config
        self.timer: SequenceTimer = SequenceTimer()
        """Stage clock of the sequence being converted.

        dataforge-convert installs a fresh one before every sequence; converters
        time stages with self.timer.stage(name) and report self.timer.capture_s.
        """

    def targets(self, identity: SequenceIdentity) -> dict[str, Path]:
        """Layer destinations for one sequence; a ``FrameLimitedConfig`` preview lands in its own tree."""
        frame_limit: int | None = self.config.frame_limit if isinstance(self.config, FrameLimitedConfig) else None
        return paths.layer_targets(identity, self.layers, frame_limit=frame_limit)

    def pending_layers(self, identity: SequenceIdentity, *, force: bool, roots: Iterable[Path]) -> tuple[dict[str, Path], list[str]]:
        """First half of a conversion: the sequence's targets and, in order, the layers it still has to write.

        Refuses destinations (and the work directory) beneath ``roots``, the dataset's
        protected inputs, before anything is read. A layer already published is pending
        only under ``force``.
        """
        targets: dict[str, Path] = self.targets(identity)
        paths.require_outside([*targets.values(), paths.work_root()], roots=roots)
        pending: list[str] = [layer for layer, target in targets.items() if not writing.should_skip(target, force=force)]
        return targets, pending

    def write_layers(
        self,
        identity: SequenceIdentity,
        targets: Mapping[str, Path],
        pending: Sequence[str],
        writers: Mapping[str, Callable[[rr.RecordingStream], None]],
        *,
        together: Callable[[dict[str, rr.RecordingStream]], None] | None = None,
    ) -> None:
        """Second half of a conversion: open, write and publish every pending layer, timed as ``write:<layer>``.

        A layer with an entry in ``writers`` is written alone, in ``pending`` order. The
        pending layers without one are opened together and handed to ``together``, for
        layers fed by one pass over a shared read (EPFL's pose CSVs); it times its own work.
        """
        for layer in pending:
            if layer in writers:
                with self.timer.stage(f"write:{layer}"), self.layer_recording(identity, layer, targets[layer]) as recording:
                    writers[layer](recording)
        rest: list[str] = [layer for layer in pending if layer not in writers]
        if not rest:
            return
        if together is None:
            raise ValueError(f"no writer for pending layers {rest}")
        with ExitStack() as stack:
            together({layer: stack.enter_context(self.layer_recording(identity, layer, targets[layer])) for layer in rest})

    def layer_recording(self, identity: SequenceIdentity, layer: str, target: Path) -> AbstractContextManager[rr.RecordingStream]:
        """Open one layer's atomic recording; see ``writing.atomic_recording``.

        Only the base layer embeds the dataset's default blueprint and Rerun's own
        ``RecordingInfo``; a derived layer stacks onto it under the same recording id.
        """
        base: bool = layer == paths.BASE_LAYER
        return writing.atomic_recording(
            target,
            recording_id=identity.recording_id,
            default_blueprint=self.default_blueprint() if base else None,
            send_properties=base,
        )

    @abstractmethod
    def download(self) -> None:
        """Fetch (or verify) the raw corpus this dataset converts from."""

    def remote_sequences(self) -> list[RemoteSequence]:
        """Every sequence the source offers, without downloading it; see ``RemoteSequence``."""
        raise NotImplementedError(f"{type(self).__name__} cannot list its source")

    @abstractmethod
    def discover(self) -> list[tuple[SequenceIdentity, SourceT]]:
        """Walk the raw tree once, pairing every convertible sequence with its source."""

    def sequences(self) -> list[SequenceIdentity]:
        """Identities of every convertible sequence found on disk."""
        return [identity for identity, _ in self.discover()]

    def prefetch(self, identity: SequenceIdentity, source: SourceT, *, force: bool) -> None:
        """Optionally fetch raw inputs on a background thread.

        Must not touch self.timer, recordings, or print per-sequence progress.
        """

    @abstractmethod
    def convert(self, identity: SequenceIdentity, source: SourceT, *, force: bool) -> Path:
        """Write the base (and derived) layer rrds of one sequence; return the base path."""

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

    def table_fields(self) -> TableFields:
        """Segment-table columns the card and table layouts show by default.

        The segment table carries every property a layer writes, far more than a card
        can show legibly. Empty keeps the viewer default (every property column).
        """
        return TableFields()
