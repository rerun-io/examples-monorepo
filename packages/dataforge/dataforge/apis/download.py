"""``dataforge-download``: fetch (or, for local corpora, verify) a dataset's raw tree."""

from __future__ import annotations

from dataclasses import dataclass, field

from serde.json import to_json

from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig


@dataclass
class Config:
    """Download/verify one dataset's raw corpus."""

    dataset: AnnotatedDatasetUnion = field(default_factory=RobocapConfig)
    """Dataset to download; the raw-tree location lives on the dataset config."""
    list_remote: bool = False
    """Print the source's sequences as JSON lines (key, size_bytes, files) instead of downloading."""


def main(config: Config) -> None:
    """Run the dataset's download verb."""
    dataset_config: DataforgeDatasetConfig = config.dataset
    dataset: DataforgeDataset = dataset_config.setup()
    if config.list_remote:
        for sequence in dataset.remote_sequences():
            print(to_json(sequence))
        return
    dataset.download()
