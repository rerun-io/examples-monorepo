"""Default Rerun blueprint of the capture App: the 3D world beside the camera views (tabs of four for a large rig), runner health below."""

import rerun.blueprint as rrb

CAMS_PER_TAB: int = 4


def default_blueprint(cameras: tuple[str, ...]) -> rrb.Blueprint:
    """``cameras`` are entity names under ``/world`` (``rig_01/cam_00``); each 2D view shows that camera's ``pinhole`` subtree."""
    views: list[rrb.Spatial2DView] = [rrb.Spatial2DView(origin=f"/world/{camera}/pinhole", name=camera) for camera in cameras]
    pages: list[rrb.Grid] = [
        rrb.Grid(*views[i : i + CAMS_PER_TAB], grid_columns=2, name=f"{cameras[i]} ..") for i in range(0, len(views), CAMS_PER_TAB)
    ]
    world = rrb.Spatial3DView(origin="/world", name="World", contents=["+ /world/**", *(f"- /world/{camera}/pinhole/**" for camera in cameras)])
    health = rrb.TimeSeriesView(origin="/dagkit/health", name="Runner health")
    return rrb.Blueprint(
        rrb.Vertical(rrb.Horizontal(world, pages[0] if len(pages) == 1 else rrb.Tabs(*pages), column_shares=[1, 1]), health, row_shares=[3, 1]),
        collapse_panels=True,
    )
