"""Create a laser-GT-clean sub-dataset from a registered ARKitScenes dataset.

Selection: captures carrying the CA-1M laser-GT layers whose largest interior
GT gap is at most a threshold (``property:gt:max_interior_gap_s``). The
sub-dataset references the source's registered RRD storage — nothing is
copied, so creation takes ~a second. The OSS ``rerun server`` catalog is
in-memory: re-run this after every server restart, after re-registering the
source dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rerun.catalog import CatalogClient, DatasetEntry

from arkitscenes_download.download_dataset import CONSOLE
from arkitscenes_download.ingest.catalog import DEFAULT_CATALOG_URL, register_default_blueprints


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for deriving the GT-clean sub-dataset."""

    source_dataset: str = "arkitscenes-v2"
    """Registered dataset to filter."""
    dataset_name: str = "arkitscenes-v2-gt-clean"
    """Sub-dataset to create."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the local Rerun catalog (``rerun server``)."""
    max_interior_gap_s: float = 1.0
    """Largest tolerated hole between consecutive GT frames (10 Hz grid = 0.1 s)."""
    blueprint_dir: Path = Path("/var/tmp/arkitscenes-blueprints")
    """Where the generated .rbl blueprint files live (the server may read them lazily)."""


def _first(cell: Any) -> Any:
    """Unwrap a segment-table property cell (a component batch) to its scalar."""
    return cell[0] if cell is not None and hasattr(cell, "__len__") and len(cell) else None


def gt_clean_mask(segment_table: pd.DataFrame, max_interior_gap_s: float) -> pd.Series:
    """Select segments with laser GT whose largest interior gap is within tolerance."""
    has_gt: pd.Series = segment_table["property:gt:provenance"].map(_first).notna()
    max_gap: pd.Series = segment_table["property:gt:max_interior_gap_s"].map(_first)
    return has_gt & (max_gap <= max_interior_gap_s).fillna(False)


def main(config: Config) -> None:
    """Create the sub-dataset by registering the selected segments' existing storage."""
    client = CatalogClient(config.catalog_url)
    source: DatasetEntry = client.get_dataset(name=config.source_dataset)
    segment_table: pd.DataFrame = source.segment_table().df.to_pandas()
    selected: pd.DataFrame = segment_table[gt_clean_mask(segment_table, config.max_interior_gap_s)]

    uris: list[str] = []
    layers: list[str] = []
    for _, row in selected.iterrows():
        uris.extend(row["rerun_storage_urls"])
        layers.extend(row["rerun_layer_names"])
    if not uris:
        raise SystemExit(f"no segments in {config.source_dataset!r} pass gt gap <= {config.max_interior_gap_s}s")

    if config.dataset_name in client.dataset_names():
        raise SystemExit(f"dataset {config.dataset_name!r} already exists; the in-memory catalog drops it on server restart")
    sub_dataset: DatasetEntry = client.create_dataset(config.dataset_name)
    sub_dataset.register(uris, layer_name=layers).wait()
    register_default_blueprints(sub_dataset, config.dataset_name, config.blueprint_dir)
    CONSOLE.print(
        f"sub-dataset '{config.dataset_name}' at {config.catalog_url}: {len(selected)} of {len(segment_table)} segments "
        f"(laser GT present, interior gaps <= {config.max_interior_gap_s}s)"
    )
