"""Write and register the exocalib viewer layout as the catalog dataset's default blueprint.

Catalog datasets carry no layout of their own: the converters' base rrds embed a
blueprint the catalog does not serve, and only the dataforge converters register a
dataset default (a 2D grid for wildcap, whose bases have no calibration to place in
3D). This stage registers the exoego layout — a 3D scene, the ego cameras tabbed
beside it, the exo cameras as a strip below — plus the two visibility rules only a
blueprint can express: Stage B's queryable ``…_raw`` record stays out of the 2D
views, and an ego camera without calibration (no pose in any layer) stays out of
the 3D scene. Everything else about how exocalib data looks — the init / refined
frustum colours, class colours, radii — is logged with the data.

Catalog partitions use the dataset id as their application id, and the id changes
whenever the dataset is recreated, so rerun this tool after a dataset recreate.
"""

from dataclasses import dataclass, field
from pathlib import Path

import rerun.blueprint as rrb
import tyro

from exo_calib.catalog_io import RigLayout, StageConfig, StageContext, stage_context
from exo_calib.kp2d_layer import kp2d_raw_entity
from exo_calib.layer_io import pinhole_entity

BLUEPRINT_FILENAME: str = "exocalib.rbl"
"""Written under ``output_dir``; a ``-N`` suffix is added if it exists (the server caches file descriptors, so registered files are never overwritten)."""


@dataclass
class BlueprintConfig:
    """Config for the blueprint writer; ``register`` sets the .rbl as the dataset's default blueprint."""

    stage: tyro.conf.OmitArgPrefixes[StageConfig] = field(default_factory=StageConfig)
    """Shared stage flags (catalog, dataset, segment, rigs, output, register); the CLI shows them unprefixed."""
    loop_playback: bool = False
    """Loop the whole timeline instead of parking on the last frame; for short standalone exports of a segment."""


def main(config: BlueprintConfig) -> None:
    """Recreate the exoego viewer layout, bound to the dataset's current id.

    Mirrors ``simplecv.apis.view_exoego.create_container`` (the base rrd's own
    blueprint, which the catalog does not serve). Not imported from simplecv
    because that module pulls in dataset configs with deps outside this env.
    """
    context: StageContext = stage_context(config.stage)
    layout: RigLayout = context.layout
    exo_tabs: list[rrb.Tabs] = [
        rrb.Tabs(rrb.Spatial2DView(origin=pinhole_entity(name), name=name, overrides={kp2d_raw_entity(name): rrb.EntityBehavior(visible=False)}))
        for name in layout.exo_camera_names
    ]
    ego_tabs: list[rrb.Tabs] = [rrb.Tabs(rrb.Spatial2DView(origin=pinhole_entity(name), name=name)) for name in layout.ego_camera_names]
    main_view: rrb.Horizontal = rrb.Horizontal(
        contents=[
            rrb.Spatial3DView(
                origin="/",
                name="3D View",
                contents="/**",
                spatial_information=rrb.SpatialInformation.from_fields(show_axes=True),
                # An ego camera without dataset calibration has no pose in any layer, so keep it out of the 3D scene.
                overrides={
                    pinhole_entity(name): rrb.EntityBehavior(visible=False)
                    for name in layout.ego_camera_names
                    if name not in layout.calibrated_camera_names
                },
            ),
            rrb.Vertical(contents=ego_tabs) if ego_tabs else rrb.Tabs(),
        ],
        column_shares=[4, 1],
    )
    container: rrb.Vertical = rrb.Vertical(main_view, rrb.Horizontal(contents=exo_tabs), row_shares=[4, 1])
    # An explicit time panel opts out of ``collapse_panels``, so restate the collapsed state.
    panels: list[rrb.TimePanel] = [rrb.TimePanel(loop_mode="All", state=rrb.PanelState.Collapsed)] if config.loop_playback else []
    blueprint: rrb.Blueprint = rrb.Blueprint(container, *panels, collapse_panels=True)
    config.stage.output_dir.mkdir(parents=True, exist_ok=True)
    rbl_path: Path = config.stage.output_dir / BLUEPRINT_FILENAME
    counter: int = 1
    while rbl_path.exists():
        rbl_path = config.stage.output_dir / f"{Path(BLUEPRINT_FILENAME).stem}-{counter}.rbl"
        counter += 1
    blueprint.save(str(context.dataset.id), str(rbl_path))
    print(f"wrote {rbl_path} for application id {context.dataset.id}")
    if config.stage.register:
        context.dataset.register_blueprint(rbl_path.resolve().as_uri(), set_default=True)
        print(f"registered as default blueprint: {context.dataset.default_blueprint()}")
