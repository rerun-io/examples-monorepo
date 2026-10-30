"""Publish SelfCap catalog layouts and optional summary ranks without changing source layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry

from dataforge.apis.register import republish_blueprints
from dataforge.apis.selfcap_quality import publish_summary
from dataforge.datasets.selfcap import SelfcapConfig


@dataclass
class Config:
    """Refresh the layouts of an existing SelfCap dataset on the catalog host."""

    catalog_name: str
    """Existing SelfCap dataset, including a refined-only subset."""
    output_root: Path
    """Dataset directory on this host; fresh .rbl files go under blueprints/."""
    refresh_quality: bool = False
    """Recompute ranks from current catalog metadata and replace only the summary layer."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog whose server can read output_root on this same host."""


def main(config: Config) -> None:
    """Register the default and table blueprints; never create datasets or change source layers."""
    entry: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.catalog_name)
    if config.refresh_quality:
        publish_summary(entry, config.output_root)
    republish_blueprints(entry, SelfcapConfig().setup(), config.catalog_name, config.output_root)
    print(f"Published SelfCap default and segment-table blueprints for {config.catalog_name}")
