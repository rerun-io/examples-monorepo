from __future__ import annotations

import rerun.blueprint as rrb


def create_sam3d_body_blueprint(camera_id: str = "sam3d_image") -> rrb.Blueprint:
    """Create a sam3d-body-rerun style layout for a converted SAM-3D-Body recording."""
    ego_pinhole_path: str = f"/world/ego/{camera_id}/pinhole"
    view_2d: rrb.Vertical = rrb.Vertical(
        contents=[
            rrb.Spatial2DView(name="image", origin="/world/cam/pinhole", contents=["/world/cam/pinhole/image"]),
            rrb.Spatial2DView(
                name="mhr",
                origin="/world/cam/pinhole",
                contents=["/world/cam/pinhole/image/**", f"{ego_pinhole_path}/image/**", f"{ego_pinhole_path}/subjects/**"],
            ),
        ]
    )
    view_3d: rrb.Spatial3DView = rrb.Spatial3DView(
        name="mhr_3d",
        origin="/world",
        contents=["/world/cam/**", "/world/pred/**", "/world/gt/**", f"/world/ego/{camera_id}/**"],
        line_grid=rrb.LineGrid3D(visible=False),
    )
    return rrb.Blueprint(
        rrb.Tabs(contents=[rrb.Horizontal(contents=[view_2d, view_3d], column_shares=[2, 3])], name="sam-3d-body-parquet"),
        collapse_panels=True,
    )
