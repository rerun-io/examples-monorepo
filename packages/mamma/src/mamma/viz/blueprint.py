"""Default Rerun blueprint for the streaming pipeline.

Layout ported from the original ``visualization/rerun_log.py``: a 3D world view
beside the per-camera 2D views (paged into tabs of 4 when the rig is large so
a 32-camera dome stays readable).
"""

from __future__ import annotations

import rerun.blueprint as rrb

WORLD_TAG: str = "world"
"""Root entity for everything in the shared 3D frame."""
CAM_TAG: str = "world/cameras"
"""Parent entity for per-camera transforms/pinholes."""

_CAMS_PER_TAB: int = 4
"""Camera 2D views per tab page; <=4 cameras keeps a single grid."""


def camera_entity(name: str) -> str:
    """Entity path holding a camera's world transform."""
    return f"{CAM_TAG}/{name}"


def pinhole_entity(name: str) -> str:
    """Entity path holding a camera's pinhole + image-space children."""
    return f"{CAM_TAG}/{name}/pinhole"


def default_blueprint(camera_names: list[str]) -> rrb.Blueprint:
    """Build the default layout: 3D world view | camera grid (or tabs of 4)."""
    world_view: rrb.Spatial3DView = rrb.Spatial3DView(origin=f"/{WORLD_TAG}", name="World")
    camera_views: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin=pinhole_entity(name), name=name) for name in camera_names
    ]

    if len(camera_views) > _CAMS_PER_TAB:
        tab_grids: list[rrb.Grid] = []
        for t in range(0, len(camera_views), _CAMS_PER_TAB):
            group: list[rrb.Spatial2DView] = camera_views[t : t + _CAMS_PER_TAB]
            lo: str = camera_names[t]
            hi: str = camera_names[min(t + _CAMS_PER_TAB - 1, len(camera_names) - 1)]
            tab_grids.append(rrb.Grid(*group, grid_columns=2, name=f"{lo}-{hi}"))
        camera_container: rrb.ContainerLike = rrb.Tabs(*tab_grids, name="Camera Images")
    else:
        camera_container = rrb.Grid(*camera_views, grid_columns=2, name="Camera Images")

    return rrb.Blueprint(
        rrb.Horizontal(world_view, camera_container, column_shares=[1, 1]),
        collapse_panels=True,
    )
