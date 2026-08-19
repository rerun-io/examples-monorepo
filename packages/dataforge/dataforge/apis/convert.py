"""``dataforge-convert``: write the base-layer rrd for one or every sequence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity


@dataclass
class Config:
    """Convert discovered sequences into base-layer recordings."""

    dataset: AnnotatedDatasetUnion = field(default_factory=RobocapConfig)
    """Dataset to convert; the raw-tree location lives on the dataset config."""
    sequence: str | None = None
    """Convert only this sequence (its ``sequence_key``, ``recording_id``, or any part)."""
    force: bool = False
    """Rewrite recordings that already exist instead of skipping them."""


def select(identities: list[SequenceIdentity], sequence: str | None) -> list[SequenceIdentity]:
    """Filter discovered identities by key, recording id, or a single part."""
    if sequence is None:
        return identities
    matches: list[SequenceIdentity] = [
        identity for identity in identities if sequence in (identity.sequence_key, identity.recording_id) or sequence in identity.parts
    ]
    if not matches:
        raise ValueError(f"No sequence matches {sequence!r} among {len(identities)} discovered sequences")
    return matches


def main(config: Config) -> None:
    """Convert every selected sequence serially."""
    dataset_config: DataforgeDatasetConfig = config.dataset
    dataset: DataforgeDataset = dataset_config.setup()
    selected: list[SequenceIdentity] = select(dataset.sequences(), config.sequence)
    print(f"converting {len(selected)} sequence(s)")
    for identity in selected:
        target: Path = dataset.convert(identity, force=config.force)
        if not target.is_file():
            raise RuntimeError(f"convert produced no recording for {identity.sequence_key}")
