"""The viewer layout of a tracked UmeTrack segment: the predicted and ground-truth hands in 3D, the four cameras, the plots.

One colour language across the panes: white is the ground truth (a translucent veil over the mesh in 3D, the box
and the skeleton in 2D); cyan and orange are the predicted left and right hand (opaque meshes, skeletons, series).
A predicted box is magenta when DetNet found it and green when the track projected it; DetNet-alone's boxes
(the ``detnet_v1`` layer, when stacked) are yellow. Entities of a layer that is not stacked are inert, so the
layout also serves a segment that carries only some of the layers.
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from dataforge import blueprints, schema
from jaxtyping import Float64
from numpy import ndarray

from handtrack import rerun_layers
from handtrack.rerun_layers import NUM_CAMERAS, RIG, SIDES

HANDS_IN_RIG: tuple[float, float, float] = (0.05, 0.45, 0.03)
"""Where the hands usually are in the rig (``cam_00``) frame, in metres: the mean ground-truth landmark over 8 UmeTrack
test segments (per-segment means lie within 0.2 m of it; the spread within a segment is about 0.1 m)."""
UP_IN_RIG: tuple[float, float, float] = (0.45, -0.8, 0.4)
"""The world's +Y in the rig frame, averaged over the same segments (it varies by about 0.15 per axis)."""
EYE_BACK_M: float = 0.25
"""How far behind the headset the eye sits, along the horizontal direction to the hands."""
EYE_UP_M: float = 0.4
"""How far above the headset the eye sits. With ``EYE_BACK_M`` chosen from screenshots of a hand_hand clip (hands
close together) and a separate_hand clip (hands 0.5 m apart): both keep both hands in shot."""
GT_MESH_ALBEDO: tuple[int, int, int, int] = (255, 255, 255, 90)
"""The ground-truth meshes repainted as a translucent white veil: white is the ground truth in every pane."""
PRED_MESH_ALPHA: int = 255
"""The predicted meshes are opaque in the side colours of the predicted keypoints, so a deviation shows as colour beside the veil."""
VIEW_ROW_SHARES: tuple[int, int] = (3, 1)
"""Height split between the 3D view with the cameras and the plots."""
VIEW_COLUMN_SHARES: tuple[int, int] = (2, 3)
"""Width split between the 3D view and the camera grid."""


def scene_eye() -> rrb.EyeControls3D:
    """An orbital eye that rides the headset: behind and above it, aimed at where the hands usually are.

    The 3D view is rooted at the rig, so one eye serves every segment and the hands stay in shot however the wearer turns.
    """
    up: Float64[ndarray, "3"] = np.asarray(UP_IN_RIG) / np.linalg.norm(UP_IN_RIG)
    target: Float64[ndarray, "3"] = np.asarray(HANDS_IN_RIG)
    level: Float64[ndarray, "3"] = target - (target @ up) * up
    forward: Float64[ndarray, "3"] = level / np.linalg.norm(level)
    position: Float64[ndarray, "3"] = -EYE_BACK_M * forward + EYE_UP_M * up
    return blueprints.eye_controls_from_pose(tuple(position.tolist()), HANDS_IN_RIG, tuple(up.tolist()))


def camera_pane(camera: int) -> rrb.Spatial2DView:
    """One camera: its video with every box and keypoint overlay of both layers, rooted at the pinhole (``dataforge.blueprints.camera_view``)."""
    pinhole: str = schema.pinhole_path(RIG, camera)
    return blueprints.camera_view(
        f"cam_{camera:02}", RIG, camera, contents=[f"+ {schema.video_path(RIG, camera)}", f"+ {pinhole}/boxes/**", f"+ {rerun_layers.camera_root(camera)}/**"]
    )


def scene_view() -> rrb.Spatial3DView:
    """Both hands' meshes and skeletons, predicted and ground truth; the camera frusta, videos and 2D overlays stay out."""
    excluded: list[str] = [f"- {schema.pinhole_path(RIG, camera)}/**" for camera in range(NUM_CAMERAS)]
    return rrb.Spatial3DView(
        name="Hands 3D",
        origin=schema.rig_path(RIG),
        contents=["+ /world/**", *excluded],
        # LineGrid3D draws on the origin frame's z = 0 plane, which in the rig frame is no ground.
        line_grid=False,
        eye_controls=scene_eye(),
        overrides={
            **{schema.hand_mesh_path(side): rr.Mesh3D.from_fields(albedo_factor=GT_MESH_ALBEDO) for side in SIDES},
            **{
                rerun_layers.pred_mesh_path(side): rr.Mesh3D.from_fields(albedo_factor=(*rerun_layers.PRED_COLORS[index], PRED_MESH_ALPHA))
                for index, side in enumerate(SIDES)
            },
        },
    )


def plots() -> list[rrb.TimeSeriesView]:
    """The per-frame 3D error, KeyNet's presence per camera, and the track state with DetNet's round-robin presence."""
    return [
        rrb.TimeSeriesView(name="3D keypoint error (mm)", origin=f"{rerun_layers.SERIES_ROOT}/error_mm", plot_legend=rrb.PlotLegend(visible=True)),
        rrb.TimeSeriesView(
            name="KeyNet presence",
            origin=f"{rerun_layers.SERIES_ROOT}/presence",
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(0.0, 1.05)),
        ),
        rrb.TimeSeriesView(
            name="Tracked · DetNet presence",
            origin=rerun_layers.SERIES_ROOT,
            contents=[f"+ {rerun_layers.SERIES_ROOT}/tracked/**", f"+ {rerun_layers.SERIES_ROOT}/detnet_presence/**"],
            plot_legend=rrb.PlotLegend(visible=True),
            axis_y=rrb.ScalarAxis(range=(0.0, 1.05)),
        ),
    ]


def handtrack_blueprint() -> rrb.Blueprint:
    """3D beside the 2x2 camera grid, over the plots, on ``video_time``."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                scene_view(),
                rrb.Grid(*(camera_pane(camera) for camera in range(NUM_CAMERAS)), grid_columns=2, name="Cameras"),
                column_shares=list(VIEW_COLUMN_SHARES),
            ),
            rrb.Horizontal(*plots()),
            row_shares=list(VIEW_ROW_SHARES),
        ),
        # An explicit TimePanel is not collapsed by collapse_panels, so it states its own state.
        rrb.TimePanel(timeline=schema.TIMELINE, state="collapsed"),
        collapse_panels=True,
    )
