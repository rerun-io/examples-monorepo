"""Publish SelfCap catalog layouts without registering or changing recording layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry

from dataforge.apis.register import republish_blueprints
from dataforge.datasets.selfcap import SelfcapConfig


@dataclass
class Config:
    """Refresh the layouts of an existing SelfCap dataset on the catalog host."""

    catalog_name: str
    """Existing SelfCap dataset, including a refined-only subset."""
    output_root: Path
    """Dataset directory on this host; fresh .rbl files go under blueprints/."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog whose server can read output_root on this same host."""


def main(config: Config) -> None:
    """Register the default and table blueprints; never create datasets or touch layers."""
    entry: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.catalog_name)
    republish_blueprints(entry, SelfcapConfig().setup(), config.catalog_name, config.output_root)
    print(f"Published SelfCap default and segment-table blueprints for {config.catalog_name}")
