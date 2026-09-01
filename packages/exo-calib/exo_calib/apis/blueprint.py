"""Write the exocalib viewer blueprint (.rbl) for the current catalog dataset.

Catalog partitions use the dataset id as their application id, and the id
changes every time the dataset is recreated, so the blueprint must be
regenerated after re-registration and opened alongside the segment URL.
"""

from dataclasses import dataclass
from pathlib import Path

import rerun.blueprint as rrb

from exo_calib.catalog_io import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, EGO_CAMERA_NAMES, EXO_CAMERA_NAMES, connect_dataset


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
    """Recreate the exoego viewer layout, bound to the dataset's current id.

    Mirrors ``simplecv.apis.view_exoego.create_container`` (the base rrd's own
    blueprint, which the catalog does not serve): a 3D view with the ego cams
    tabbed in a right-hand column and the exo cams as a tabbed strip below.
    Not imported from simplecv because that module pulls in dataset configs
    with deps outside this env.
    """
    dataset = connect_dataset(config.catalog_url, config.dataset_name)
    exo_tabs: list[rrb.Tabs] = [rrb.Tabs(rrb.Spatial2DView(origin=f"/world/{name}/pinhole", name=name)) for name in EXO_CAMERA_NAMES]
    ego_tabs: list[rrb.Tabs] = [rrb.Tabs(rrb.Spatial2DView(origin=f"/world/{name}/pinhole", name=name)) for name in EGO_CAMERA_NAMES]
    main_view: rrb.Horizontal = rrb.Horizontal(
        contents=[
            rrb.Spatial3DView(
                origin="/",
                name="3D View",
                contents="/**",
                spatial_information=rrb.SpatialInformation.from_fields(show_axes=True),
            ),
            rrb.Vertical(contents=ego_tabs),
        ],
        column_shares=[4, 1],
    )
    container: rrb.Vertical = rrb.Vertical(
        main_view,
        rrb.Horizontal(contents=exo_tabs),
        row_shares=[4, 1],
    )
    blueprint: rrb.Blueprint = rrb.Blueprint(rrb.Horizontal(contents=[container], column_shares=[4, 1]), collapse_panels=True)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    blueprint.save(str(dataset.id), str(config.output))
    print(f"wrote {config.output} for application id {dataset.id}")
