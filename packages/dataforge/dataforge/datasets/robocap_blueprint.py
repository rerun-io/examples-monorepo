"""The RoboCap result layout: one segment in four tabs, SLAM / colsfm / Comparison / Map.

The Insta360 layout's skeleton (``insta360_blueprint``: an overview of the walk top left, the level follow view beside or
under it, the source's own pictures in the right column; slam-rs orange, colsfm blue) with RoboCap's six fisheye cameras
instead of two lenses and six cube faces, and a Map tab for a GPS track (``rrb.MapView`` on ``/world/gps``).

What differs from Insta360:

- slam-rs feeds four of the six cameras (left_front, right_front, left, right = rig positions 0, 1, 4, 5). Its own run
  rig numbers them by feed order (``/world/runs/slam_rs_vio/rig/cam_00..03``); the keypoints drawn on the base cameras
  are a layer of their own (``SLAM_KEYPOINTS`` under each fed camera's ``pinhole``).
- colsfm runs on all six cameras; its rig's frusta are named by camera (``/world/runs/colsfm/rig/<camera>/pinhole``).
- Every 3D view follows the raw slam-rs run (``slam_rs_vio``), colsfm's prior: the final run can carry a false loop.

The layout names entities of layers a segment may not have yet; an override on a missing entity is inert.
"""

from __future__ import annotations

from collections.abc import Sequence

import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from numpy import ndarray

from dataforge import blueprints, schema
from dataforge.datasets import insta360_blueprint as shared

RIG: int = 0
"""RoboCap is one rig."""
CAMERAS: tuple[str, ...] = ("left_front", "right_front", "left_eye", "right_eye", "left", "right")
"""The base's cameras, ``cam_00..cam_05`` (dataforge ``CAMERA_DISPLAY_ORDER``)."""
SLAM_CAMERAS: tuple[int, ...] = (0, 1, 4, 5)
"""Rig positions slam-rs's RoboCap rig feeds, in feed order."""
SLAM_KEYPOINTS: str = "slam_keypoints"
"""Leaf under a base camera's ``pinhole`` holding the keypoints slam-rs tracked on it."""
GPS_ROOT: str = "/world/gps"
"""Where the GPS layer logs its ``GeoLineStrings`` / ``GeoPoints``."""
ANALYSIS_ROOT: str = "/analysis"
"""Where the GPS layer logs its speed and heading-drift plots."""
FOLLOW_SOURCE: str = schema.SLAM_RS_VIO_SOURCE
"""The run every follow view and overview dot rides: the raw VIO, colsfm's prior."""


def camera_pane(position: int, *, keypoints: bool, mask: bool) -> rrb.Spatial2DView:
    """One base camera: its video, optionally slam-rs's keypoints and the people mask on its own stamps."""
    pinhole: str = schema.pinhole_path(RIG, position)
    contents: list[str] = [schema.video_path(RIG, position)]
    overrides: shared.Overrides = {}
    if keypoints:
        contents.append(f"{pinhole}/{SLAM_KEYPOINTS}")
        overrides[f"{pinhole}/{SLAM_KEYPOINTS}"] = rr.Points2D.from_fields(radii=rr.Radius.ui_points(shared.KEYPOINT_RADIUS_UI))
    if mask:
        # Masks exist at colsfm's keyframe stamps only: show each on its own stamp, not held over the frames after it.
        contents.append(schema.people_mask_path(RIG, position))
        overrides[schema.people_mask_path(RIG, position)] = blueprints.cursor_window(0.0)
    return blueprints.camera_view(CAMERAS[position], RIG, position, contents=contents, overrides=overrides or None)


def slam_rig_frusta(size_m: float) -> shared.Overrides:
    """The four fed cameras of slam-rs's own rig copy, in its colour."""
    return shared.frusta(f"{shared.VIO}/rig", [f"cam_{index:02d}/pinhole" for index in range(len(SLAM_CAMERAS))], shared.SLAM_COLOR, size_m)


def colsfm_rig_frusta(size_m: float) -> shared.Overrides:
    """colsfm's six cameras, in its colour."""
    return shared.frusta(f"{shared.COLSFM}/rig", [f"{name}/pinhole" for name in CAMERAS], shared.COLSFM_COLOR, size_m)


def base_cameras_out() -> list[str]:
    """Content rules dropping the base cameras' subtrees (videos, masks, keypoints) from a 3D view."""
    return [f"- {schema.cam_path(RIG, position)}/**" for position in range(len(CAMERAS))]


def slam_tab(eye: rrb.EyeControls3D | None) -> rrb.Vertical:
    """slam-rs: the map and the VIO path, the follow view with the window and landmarks, the four fed cameras."""
    scene: list[str] = ["/world/**", *base_cameras_out(), f"- {shared.COLSFM}/**", f"- {GPS_ROOT}/**", *shared.window_pictures()]
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.Vertical(
                shared.overview("Overview: SLAM map and VIO path", scene, shared.slam_overrides(close=False), eye, FOLLOW_SOURCE),
                shared.follow(
                    "Follow: level chase view, VIO window and landmarks",
                    scene,
                    {**shared.slam_overrides(close=True), **slam_rig_frusta(0.18)},
                    FOLLOW_SOURCE,
                ),
                row_shares=[5, 4],
            ),
            rrb.Grid(*(camera_pane(position, keypoints=True, mask=False) for position in SLAM_CAMERAS), grid_columns=2, name="SLAM cameras"),
            column_shares=[3, 2],
        ),
        shared.imu_strip(),
        row_shares=[7, 1],
        name="SLAM",
    )


def colsfm_overrides(*, close: bool) -> shared.Overrides:
    """colsfm's trail in its walk gradient, the cloud growing from the start of the walk, its six frusta."""
    return {
        schema.trail_path(schema.COLSFM_SOURCE): shared.trail(None, 3.0),
        schema.trajectory_path(schema.COLSFM_SOURCE): shared.stroke(None, 2.5),
        f"{shared.COLSFM}/points_growing/recent": [blueprints.cursor_window(shared.GROWING_BLOCK_S), shared.cloud(shared.growing_cloud_size(close=close))],
        f"{shared.COLSFM}/points_growing": [blueprints.cursor_window(None), shared.cloud(shared.growing_cloud_size(close=close))],
        **colsfm_rig_frusta(0.18 if close else 0.3),
    }


def colsfm_tab(eye: rrb.EyeControls3D | None) -> rrb.Horizontal:
    """colsfm: the cloud that grows with the walk, the follow view of its six-camera rig, the six cameras with masks."""
    scene: list[str] = [f"{shared.COLSFM}/**", f"- {shared.COLSFM}/points/**", f"- {shared.POSITION_DIFFERENCE}/**", schema.follow_path(FOLLOW_SOURCE)]
    return rrb.Horizontal(
        rrb.Vertical(
            shared.overview("Overview: colsfm cloud grows with the walk", scene, colsfm_overrides(close=False), eye, FOLLOW_SOURCE,
                            shared.colsfm_cloud_defaults(close=False)),
            shared.follow("Follow: level chase view, colsfm rig", scene, colsfm_overrides(close=True), FOLLOW_SOURCE, shared.colsfm_cloud_defaults(close=True)),
            row_shares=[5, 4],
        ),
        rrb.Grid(*(camera_pane(position, keypoints=False, mask=True) for position in range(len(CAMERAS))), grid_columns=2, name="colsfm cameras"),
        column_shares=[3, 2],
        name="colsfm",
    )


def comparison_overrides(*, close: bool) -> shared.Overrides:
    """The VIO path in flat grey, colsfm's in its walk gradient, both clouds in their sources' colours."""
    wide, narrow = (3.5, 3.5) if close else (6.0, 3.0)
    return {
        schema.trail_path(FOLLOW_SOURCE): shared.trail(shared.COMPARISON_SLAM_COLOR, wide),
        schema.trajectory_path(FOLLOW_SOURCE): shared.stroke(shared.COMPARISON_SLAM_COLOR, wide),
        schema.trail_path(schema.COLSFM_SOURCE): shared.trail(None, narrow),
        schema.trajectory_path(schema.COLSFM_SOURCE): shared.stroke(None, narrow),
        f"{shared.SLAM}/map/points": shared.cloud(1.5, shared.shade(shared.SLAM_COLOR, shared.CLOUD_SHADE)),
        f"{shared.COLSFM}/points": shared.cloud(1.5, shared.shade(shared.COLSFM_COLOR, shared.CLOUD_SHADE)),
        **shared.hidden(f"{shared.SLAM}/gba", f"{shared.SLAM}/pgo", f"{shared.SLAM}/pose_graph", f"{shared.SLAM}/keyframes", f"{shared.SLAM}/loops"),
        **colsfm_rig_frusta(0.125 if close else 0.25),
    }


def comparison_tab(eye: rrb.EyeControls3D | None) -> rrb.Vertical:
    """Both sources in one world, close up at the rig, and how far colsfm's positions are from its prior."""
    scene: list[str] = [
        "/world/**",
        *base_cameras_out(),
        f"- {GPS_ROOT}/**",
        f"- {shared.COLSFM}/points_growing/**",
        f"- {shared.POSITION_DIFFERENCE}/**",
        f"- {shared.VIO}/window/**",
        f"- {shared.VIO}/marginalized/**",
        f"- {shared.VIO}/landmarks",
        # The final slam-rs path can carry a false loop; colsfm's prior is the VIO run.
        f"- {shared.SLAM}/trajectory",
        f"- {shared.SLAM}/trail",
    ]
    return rrb.Vertical(
        rrb.Horizontal(
            shared.overview("Overview: VIO path (grey) and colsfm (gradient), both clouds", scene, comparison_overrides(close=False), eye, FOLLOW_SOURCE),
            shared.follow("Follow: colsfm's rig on the VIO path", scene, comparison_overrides(close=True), FOLLOW_SOURCE),
            column_shares=[3, 2],
        ),
        rrb.TimeSeriesView(
            name="Position difference, colsfm to its prior [m]",
            origin=shared.POSITION_DIFFERENCE,
            plot_legend=rrb.PlotLegend(rrb.Corner2D.LeftTop, visible=True),
            overrides={shared.POSITION_DIFFERENCE: rr.SeriesLines(colors=[shared.DIFFERENCE_COLOR], widths=[2.0])},
        ),
        row_shares=[3, 1],
        name="Comparison",
    )


def map_tab(camera_positions: Sequence[int] = SLAM_CAMERAS) -> rrb.Horizontal:
    """The GPS track on OpenStreetMap with the moving position and the loop ends, speed and heading drift, the cameras."""
    return rrb.Horizontal(
        rrb.Vertical(
            rrb.MapView(origin=GPS_ROOT, name="Map: GPS track, position, loop ends (magenta real, red false)", zoom=15.0,
                        background=rrb.MapProvider.OpenStreetMap),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="Speed [m/s]", origin=f"{ANALYSIS_ROOT}/speed"),
                rrb.TimeSeriesView(name="Heading drift vs GPS [deg]", origin=f"{ANALYSIS_ROOT}/heading_drift_deg"),
            ),
            row_shares=[3, 1],
        ),
        rrb.Grid(*(camera_pane(position, keypoints=False, mask=False) for position in camera_positions), grid_columns=2, name="Cameras"),
        column_shares=[3, 2],
        name="Map",
    )


def result_layout(walk: Float64[ndarray, "n 3"] | None = None) -> rrb.Blueprint:
    """The RoboCap result layout: SLAM, colsfm, Comparison and Map tabs.

    Args:
        walk: Rig positions of the walk (world frame, metres), when known: each overview then frames it (``walk_eye``).
    """
    eyes: list[rrb.EyeControls3D | None] = [None if walk is None else shared.walk_eye(walk, aspect=aspect) for aspect in shared.OVERVIEW_ASPECTS]
    return rrb.Blueprint(
        rrb.Tabs(slam_tab(eyes[0]), colsfm_tab(eyes[1]), comparison_tab(eyes[2]), map_tab(), active_tab=0),
        rrb.TimePanel(timeline=schema.TIMELINE, state="collapsed"),
        collapse_panels=True,
    )
