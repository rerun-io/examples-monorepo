"""Write the exocalib viewer blueprint (.rbl) for the current catalog dataset.

Catalog partitions use the dataset id as their application id, and the id
changes every time the dataset is recreated, so the blueprint must be
regenerated after re-registration and opened alongside the segment URL.
"""

from dataclasses import dataclass
from pathlib import Path

import rerun.blueprint as rrb

from exo_calib.catalog_io import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, EXO_CAMERA_NAMES, connect_dataset


@dataclass
class BlueprintConfig:
    """Config for the blueprint writer."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """Rerun catalog server URL."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset the blueprint targets (its id becomes the application id)."""
    output: Path = Path("data/exocalib.rbl")
    """Where to write the .rbl."""


def main(config: BlueprintConfig) -> None:
    """Write a world + per-camera-grid blueprint bound to the dataset's current id."""
    dataset = connect_dataset(config.catalog_url, config.dataset_name)
    cam_views: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin=f"/world/{name}/pinhole", name=name.split("/")[0]) for name in EXO_CAMERA_NAMES
    ]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="world"),
            rrb.Grid(*cam_views, grid_columns=4),
            column_shares=[1.0, 2.0],
        )
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    blueprint.save(str(dataset.id), str(config.output))
    print(f"wrote {config.output} for application id {dataset.id}")
