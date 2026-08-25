"""``dataforge-register``: register base-layer rrds into a local Rerun catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import rerun.blueprint as rrb
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from dataforge import paths, writing
from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDatasetConfig


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
    dataset: DatasetEntry = client.create_dataset(name, exist_ok=True)
    dataset.register(
        [path.resolve().as_uri() for path in rrd_paths],
        layer_name=paths.BASE_LAYER,
        on_duplicate=OnDuplicateSegmentLayer.SKIP,
    ).wait()

    dataset_instance = dataset_config.setup()
    blueprint: rrb.Blueprint | None = dataset_instance.default_blueprint()
    if blueprint is not None:
        blueprint_path: Path = paths.output_root() / "blueprints" / f"{name}.rbl"
        # A re-register must never truncate the .rbl a live catalog server still holds open.
        with writing.atomic_write(blueprint_path) as temp_path:
            blueprint.save(name, str(temp_path))
        dataset.register_blueprint(blueprint_path.resolve().as_uri(), set_default=True)
    table_blueprint: rrb.Blueprint | None = dataset_instance.table_blueprint()
    if table_blueprint is not None:
        table_path: Path = paths.output_root() / "blueprints" / f"{name}-table.rbl"
        with writing.atomic_write(table_path) as temp_path:
            table_blueprint.save(name, str(temp_path))
        dataset.register_blueprint(table_path.resolve().as_uri(), segment_table=True)
    print(f"registered {len(rrd_paths)} {paths.BASE_LAYER}-layer rrds into '{name}' at {config.catalog_url}")
