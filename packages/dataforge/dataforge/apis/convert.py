"""``dataforge-convert``: write the base-layer rrd for one or every sequence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from beartype.roar import BeartypeException

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
    """Convert every selected sequence serially, surviving individual failures.

    One bad sequence must not throw away a batch that is hours in (a lesson from
    robocap's malformed ``s10``), so failures are reported and the run continues;
    the process still exits non-zero so a caller can tell a partial run apart
    from a clean one.
    """
    dataset_config: DataforgeDatasetConfig = config.dataset
    dataset: DataforgeDataset = dataset_config.setup()
    selected: list[SequenceIdentity] = select(dataset.sequences(), config.sequence)
    print(f"converting {len(selected)} sequence(s)")
    failed: list[str] = []
    for identity in selected:
        try:
            target: Path = dataset.convert(identity, force=config.force)
            if not target.is_file():
                raise RuntimeError(f"convert produced no recording for {identity.sequence_key}")
        except BeartypeException:
            raise
        except Exception as error:
            print(f"FAILED {identity.sequence_key}: {type(error).__name__}: {error}")
            failed.append(identity.sequence_key)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(selected)} sequence(s) failed: {', '.join(failed)}")
