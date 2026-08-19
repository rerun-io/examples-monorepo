"""``dataforge-download``: fetch (or, for local corpora, verify) a dataset's raw tree."""

from __future__ import annotations

from dataclasses import dataclass, field

from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDatasetConfig


@dataclass
class Config:
    """Download/verify one dataset's raw corpus."""

    dataset: AnnotatedDatasetUnion = field(default_factory=RobocapConfig)
    """Dataset to download; the raw-tree location lives on the dataset config."""


def main(config: Config) -> None:
    """Run the dataset's download verb."""
    dataset_config: DataforgeDatasetConfig = config.dataset
    dataset_config.setup().download()
