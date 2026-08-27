"""``dataforge-register``: register base-layer rrds into a local Rerun catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from dataforge import paths, writing
from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig


@dataclass
class Config:
    """Register a dataset's converted recordings into a catalog."""

    dataset: AnnotatedDatasetUnion = field(default_factory=RobocapConfig)
    """Dataset whose rrds get registered; its registry key is the catalog dataset name."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """gRPC URL of a locally running ``rerun server`` catalog."""


def main(config: Config) -> None:
    """Create the dataset if needed and register every base-layer rrd (idempotent)."""
    dataset_config: DataforgeDatasetConfig = config.dataset
    name: str = dataset_config.name
    layer_root: Path = paths.output_root() / paths.BASE_LAYER
    rrd_paths: list[Path] = sorted(layer_root.glob(f"{name}__*.rrd"))
    if not rrd_paths:
        raise FileNotFoundError(f"no {paths.BASE_LAYER}-layer rrds for {name} under {layer_root}")

    client: CatalogClient = CatalogClient(config.catalog_url)
    entry: DatasetEntry = client.create_dataset(name, exist_ok=True)
    entry.register(
        [path.resolve().as_uri() for path in rrd_paths],
        layer_name=paths.BASE_LAYER,
        on_duplicate=OnDuplicateSegmentLayer.SKIP,
    ).wait()

    dataset: DataforgeDataset = dataset_config.setup()
    # Blueprints register once: every register_blueprint call adds a NEW entry to the
    # catalog dataset's blueprint list (cluttering the viewer's selector), so an
    # incremental re-register skips a blueprint the catalog already has a default for.
    # To refresh a blueprint, delete the dataset and re-register. A re-register must
    # also never truncate an .rbl a live catalog server holds open.
    if entry.default_blueprint() is None:
        blueprint_path: Path = paths.blueprint_path(paths.output_root(), name)
        with writing.atomic_write(blueprint_path) as temp_path:
            dataset.default_blueprint().save(name, str(temp_path))
        entry.register_blueprint(blueprint_path.resolve().as_uri(), set_default=True)
    if entry.default_segment_table_blueprint() is None:
        table_path: Path = paths.blueprint_path(paths.output_root(), name, segment_table=True)
        with writing.atomic_write(table_path) as temp_path:
            dataset.table_blueprint().save(name, str(temp_path))
        entry.register_blueprint(table_path.resolve().as_uri(), segment_table=True)
    print(f"registered {len(rrd_paths)} {paths.BASE_LAYER}-layer rrds into '{name}' at {config.catalog_url}")
