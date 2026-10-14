"""Write and register the exocalib viewer layout as the catalog dataset's default blueprint.

Catalog datasets carry no layout of their own: the converters' base rrds embed a
blueprint the catalog does not serve, and only the dataforge converters register a
dataset default (a 2D grid for wildcap, whose bases have no calibration to place in
3D). This stage registers the exoego layout — a 3D scene, the ego cameras tabbed
beside it, the exo cameras as a strip below — plus the one visibility rule only a
blueprint can express: an ego camera without calibration (no pose in any layer)
stays out of the 3D scene. Everything else about how exocalib data looks — the init / refined
frustum colours, class colours, radii — is logged with the data.

Catalog partitions use the dataset id as their application id, and the id changes
whenever the dataset is recreated, so rerun this tool after a dataset recreate.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import rerun.blueprint as rrb
from dagkit.sources import SegmentRef

from exo_calib.sources.layout import discover_rig_layout
from exo_calib.types import RigLayout

BLUEPRINT_FILENAME: str = "exocalib.rbl"
"""Written under ``output_dir``; a ``-N`` suffix is added if it exists (the server caches file descriptors, so registered files are never overwritten)."""


@dataclass
class BlueprintConfig:
    """Config for the blueprint writer; ``register`` sets the .rbl as the dataset's default blueprint."""

    segment: SegmentRef
    """Any segment of the dataset (the layout is the same across a dataset's segments)."""
    exo_rigs: tuple[str, ...] | None = None
    """Explicit exo rig names; ``None`` discovers them from catalog metadata."""
    output_dir: Path = Path("data/outputs")
    register: bool = True
    """Register the .rbl as the dataset's default blueprint."""
    loop_playback: bool = False
    """Loop the whole timeline instead of parking on the last frame; for short standalone exports of a segment."""


def exoego_view(layout: RigLayout, scene_contents: Sequence[str] = ("/**",)) -> rrb.Vertical:
    """The exoego layout: a 3D scene, the ego cameras tabbed beside it (when there are any), the exo cameras as a strip below.

    Mirrors ``simplecv.apis.view_exoego.create_container`` (the base rrd's own
    blueprint, which the catalog does not serve). Not imported from simplecv
    because that module pulls in dataset configs with deps outside this env.
    """
    exo_tabs: list[rrb.Tabs] = [rrb.Tabs(rrb.Spatial2DView(origin=f"/world/{name}/pinhole", name=name)) for name in layout.exo_camera_names]
    ego_tabs: list[rrb.Tabs] = [rrb.Tabs(rrb.Spatial2DView(origin=f"/world/{name}/pinhole", name=name)) for name in layout.ego_camera_names]
    scene = rrb.Spatial3DView(
        origin="/",
        name="3D View",
        contents=list(scene_contents),
        spatial_information=rrb.SpatialInformation.from_fields(show_axes=True),
        # An ego camera without dataset calibration has no pose in any layer, so keep it out of the 3D scene.
        overrides={f"/world/{name}/pinhole": rrb.EntityBehavior(visible=False) for name in layout.ego_camera_names if name not in layout.calibrated_camera_names},
    )
    top: rrb.Horizontal | rrb.Spatial3DView = rrb.Horizontal(scene, rrb.Vertical(contents=ego_tabs), column_shares=[4, 1]) if ego_tabs else scene
    return rrb.Vertical(top, rrb.Horizontal(contents=exo_tabs), row_shares=[4, 1])


def main(config: BlueprintConfig) -> None:
    """Recreate the exoego viewer layout, bound to the dataset's current id."""
    layout: RigLayout = discover_rig_layout(config.segment, exo_rigs=config.exo_rigs)
    # An explicit time panel opts out of ``collapse_panels``, so restate the collapsed state.
    panels: list[rrb.TimePanel] = [rrb.TimePanel(loop_mode="All", state=rrb.PanelState.Collapsed)] if config.loop_playback else []
    blueprint: rrb.Blueprint = rrb.Blueprint(exoego_view(layout), *panels, collapse_panels=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rbl_path: Path = config.output_dir / BLUEPRINT_FILENAME
    counter: int = 1
    while rbl_path.exists():
        rbl_path = config.output_dir / f"{Path(BLUEPRINT_FILENAME).stem}-{counter}.rbl"
        counter += 1
    dataset = config.segment.entry()
    blueprint.save(str(dataset.id), str(rbl_path))
    print(f"wrote {rbl_path} for application id {dataset.id}")
    if config.register:
        dataset.register_blueprint(rbl_path.resolve().as_uri(), set_default=True)
        print(f"registered as default blueprint: {dataset.default_blueprint()}")
