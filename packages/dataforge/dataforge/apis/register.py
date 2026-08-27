"""``dataforge-register``: register base-layer rrds into a local Rerun catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import rerun.blueprint as rrb
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from dataforge import paths
from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig, dataset_key
from dataforge.datasets.base import DataforgeDatasetConfig

LAYER: str = "base"
"""Only layer dataforge v1 emits."""


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
    name: str = dataset_key(dataset_config)
    rrd_paths: list[Path] = sorted((paths.output_root() / LAYER).glob(f"{name}__*.rrd"))
    if not rrd_paths:
        raise FileNotFoundError(f"no {LAYER}-layer rrds for {name} under {paths.output_root() / LAYER}")

    client: CatalogClient = CatalogClient(config.catalog_url)
    dataset: DatasetEntry = client.create_dataset(name, exist_ok=True)
    dataset.register(
        [path.resolve().as_uri() for path in rrd_paths],
        layer_name=LAYER,
        on_duplicate=OnDuplicateSegmentLayer.SKIP,
    ).wait()

    blueprint: rrb.Blueprint | None = dataset_config.setup().default_blueprint()
    if blueprint is not None:
        blueprint_path: Path = paths.output_root() / "blueprints" / f"{name}.rbl"
        blueprint_path.parent.mkdir(parents=True, exist_ok=True)
        blueprint.save(name, str(blueprint_path))
        blueprint_path.chmod(0o644)  # uid-squashing NFS mounts can land fresh writes as mode 0000
        dataset.register_blueprint(blueprint_path.resolve().as_uri(), set_default=True)
    print(f"registered {len(rrd_paths)} {LAYER}-layer rrds into '{name}' at {config.catalog_url}")
