"""``dataforge-publish-blueprints``: refresh a dataset's device assets and two default layouts from the layers it holds now."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry

from dataforge.apis.register import republish_blueprints
from dataforge.datasets import AnnotatedDatasetUnion
from dataforge.datasets.base import DataforgeDataset


@dataclass
class Config:
    """Register fresh device assets and default and segment-table blueprints of one existing dataset."""

    dataset: AnnotatedDatasetUnion
    """Dataset whose layouts and assets get published."""
    catalog_url: str
    """Catalog that holds the dataset; its server must read ``output_root``."""
    catalog_name: str
    """Existing dataset; it is never created here."""
    output_root: Path
    """Fresh assets and .rbl files go here; this dataset's older .rbl files under blueprints/ are deleted once retired."""


def main(config: Config) -> None:
    """Publish the assets and both layouts as the defaults and retire the older entries; no layer is written or changed."""
    entry: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.catalog_name)
    dataset: DataforgeDataset = config.dataset.setup()
    dataset.register_assets(entry, config.output_root)
    republish_blueprints(entry, dataset, config.catalog_name, config.output_root)
    print(f"published device assets and default and segment-table blueprints of {config.catalog_name}")
