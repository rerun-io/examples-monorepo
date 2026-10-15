"""Rerun blueprint of the capture App: the 3D world beside the camera views (tabs of four for a large rig), the stage
timings, fit health and SMPL-X parameters below; then the secondary "pipeline (debug)" tab (graph picture, guide, one view
per step's output)."""

import rerun as rr
import rerun.blueprint as rrb
from dagkit.app import with_pipeline
from dagkit.doc import HEALTH_ROOT

CAMS_PER_TAB: int = 4
FRUSTUM_M: float = 0.4
"""Image-plane distance of the frusta in the World view (the rig stores 0.1): each camera's video reads beside a person."""
STAGES: tuple[str, ...] = ("track", "landmarks", "fit", "viewer")
"""The nodes whose step times (the runner's, on ``wall_time``) are the stage timings."""


def default_blueprint(cameras: tuple[str, ...]) -> rrb.Blueprint:
    """``cameras`` are entity names under ``/world`` (``rig_01/cam_00``); each 2D view shows that camera's ``pinhole`` subtree.

    The debug tab's output views are named after the graph's box numbers (``GRAPH.md``) and show the first camera.
    """
    views: list[rrb.Spatial2DView] = [rrb.Spatial2DView(origin=f"/world/{camera}/pinhole", name=camera) for camera in cameras]
    pages: list[rrb.Grid] = [
        rrb.Grid(*views[i : i + CAMS_PER_TAB], grid_columns=2, name=f"{cameras[i]}-{cameras[min(i + CAMS_PER_TAB, len(cameras)) - 1]}")
        for i in range(0, len(views), CAMS_PER_TAB)
    ]
    images: rrb.Container = rrb.Tabs(*pages, name="Camera Images") if len(pages) > 1 else rrb.Grid(*views, grid_columns=2, name="Camera Images")
    world = rrb.Spatial3DView(
        origin="/world",
        name="World",
        overrides={f"/world/{camera}/pinhole": rr.Pinhole.from_fields(image_plane_distance=FRUSTUM_M).visualizer() for camera in cameras},
    )
    plots = rrb.Horizontal(
        rrb.TimeSeriesView(
            origin=HEALTH_ROOT.rstrip("/"), name="Stage timings (ms, wall_time)", contents=[f"+ {HEALTH_ROOT}{stage}/step_ms" for stage in STAGES]
        ),
        rrb.TimeSeriesView(origin="/metrics/fit", name="Fit health"),
        rrb.TimeSeriesView(origin="/metrics/params", name="SMPL-X params"),
    )
    normal = rrb.Blueprint(
        rrb.Vertical(rrb.Horizontal(world, images, column_shares=[1, 1]), plots, row_shares=[3, 1]),
        collapse_panels=True,
    )
    mamma: str = f"/world/{cameras[0]}/pinhole/mamma"
    outputs: list[rrb.View] = [
        rrb.Spatial2DView(origin=f"/world/{cameras[0]}/pinhole", name=f"2 track · masks + boxes ({cameras[0]})", contents=[f"+ {mamma}/**", f"- {mamma}/landmarks/**"]),
        rrb.Spatial2DView(origin=f"/world/{cameras[0]}/pinhole", name=f"3 landmarks · dense 2D ({cameras[0]})", contents=[f"+ {mamma}/landmarks/**"]),
        rrb.Spatial3DView(origin="/world", name="4 fit · SMPL-X bodies + 3D landmarks", contents=["+ /world/mamma/**"]),
    ]  # fmt: skip
    return with_pipeline(normal, outputs)
