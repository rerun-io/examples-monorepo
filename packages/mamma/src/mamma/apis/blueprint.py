"""Rerun blueprint of the capture App: the 3D world beside the camera views (tabs of four for a large rig), runner health
below; then the secondary "pipeline (debug)" tab (graph picture, guide, one view per step's output)."""

import rerun.blueprint as rrb
from dagkit.app import with_pipeline

CAMS_PER_TAB: int = 4


def default_blueprint(cameras: tuple[str, ...]) -> rrb.Blueprint:
    """``cameras`` are entity names under ``/world`` (``rig_01/cam_00``); each 2D view shows that camera's ``pinhole`` subtree.

    The debug tab's output views are named after the graph's box numbers (``GRAPH.md``) and show the first camera.
    """
    views: list[rrb.Spatial2DView] = [rrb.Spatial2DView(origin=f"/world/{camera}/pinhole", name=camera) for camera in cameras]
    pages: list[rrb.Grid] = [
        rrb.Grid(*views[i : i + CAMS_PER_TAB], grid_columns=2, name=f"{cameras[i]} ..") for i in range(0, len(views), CAMS_PER_TAB)
    ]
    world = rrb.Spatial3DView(origin="/world", name="World", contents=["+ /world/**", *(f"- /world/{camera}/pinhole/**" for camera in cameras)])
    health = rrb.TimeSeriesView(origin="/dagkit/health", name="Runner health")
    normal = rrb.Blueprint(
        rrb.Vertical(rrb.Horizontal(world, pages[0] if len(pages) == 1 else rrb.Tabs(*pages), column_shares=[1, 1]), health, row_shares=[3, 1]),
        collapse_panels=True,
    )
    mamma: str = f"/world/{cameras[0]}/pinhole/mamma"
    outputs: list[rrb.View] = [
        rrb.Spatial2DView(origin=f"/world/{cameras[0]}/pinhole", name=f"2 track · masks + boxes ({cameras[0]})", contents=[f"+ {mamma}/**", f"- {mamma}/landmarks/**"]),
        rrb.Spatial2DView(origin=f"/world/{cameras[0]}/pinhole", name=f"3 landmarks · dense 2D ({cameras[0]})", contents=[f"+ {mamma}/landmarks/**"]),
        rrb.Spatial3DView(origin="/world", name="4 fit · SMPL-X bodies + 3D landmarks", contents=["+ /world/mamma/**"]),
    ]  # fmt: skip
    return with_pipeline(normal, outputs)
