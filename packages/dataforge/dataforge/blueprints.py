"""The blueprint pieces every dataforge dataset builds its layout out of.

A capture rig's viewer layout is the same shape whatever device recorded it: the
rig in 3D beside a grid of camera panes, over a row of sensor plots, with a
follow camera that rides the rig once a pose layer animates it. Only the panes,
the plots, the eye and the run source of the derived trajectory change, so those
are the arguments and everything else lives here once — including the overrides
that make the pair of 3D views complementary. The overview shows the whole path
and hides the trail; the follow view keeps both, with the path repainted thin
and dim so it reads as context under the highlighted trail rather than as a
second stroke competing with it.

The overrides name entities a base-only recording does not have. That is
deliberate: an override on a missing entity is inert, so the layout is already
right the moment a pose or ground-truth layer stacks onto the same segment.
"""

from __future__ import annotations

from collections.abc import Sequence

import rerun as rr
import rerun.blueprint as rrb

from dataforge import schema

TRAIL_WINDOW_S: float = -10.0
"""How far back of the cursor the follow view's motion trail reaches, in seconds."""
DIM_TRAJECTORY_COLOR: tuple[int, int, int] = (90, 110, 140)
"""Tint the follow view repaints the full path in: cool and desaturated, so it sits behind the trail."""
DIM_TRAJECTORY_RADIUS_UI_POINTS: float = 1.0
"""How thin the follow view draws the full path, in ui points; the trail is three."""
CAMERA_GRID_COLUMNS: int = 2
"""Columns in the synchronized-camera grid; two keeps a stereo pair side by side."""
RIG_COLUMN_SHARES: tuple[int, int] = (3, 2)
"""Width split between the 3D column and the camera grid."""
PLOT_ROW_SHARES: tuple[int, int] = (3, 1)
"""Height split between the views and the sensor plots."""
FOLLOW_BACK_M: float = 0.9
"""How far behind the device the follow eye sits, along the device's own forward."""
FOLLOW_UP_M: float = 0.45
"""How far above the device the follow eye sits, along the device's own up."""
FOLLOW_AHEAD_M: float = 0.3
"""How far ahead of the device the follow eye aims, so the shot leads the motion.

The three distances were tuned together on LaMAria's R_01_easy and hold for the MSD
headsets as they stand: a shot that keeps the camera frusta and the last ten seconds
of trail in view at once without the ground filling it."""


def camera_view(name: str, rig: int, cam: int) -> rrb.Spatial2DView:
    """One camera's 2D pane.

    The origin is the ``pinhole`` node, not the camera node, so the pane *is*
    that camera's projection and anything logged beneath it (the video, later a
    keypoint overlay) lands in the same image space.

    Args:
        name: Pane label, whatever the dataset calls the stream.
        rig: Rig index the camera hangs off.
        cam: Camera index within the rig.

    Returns:
        The pane every dataset's camera grid is built from.
    """
    return rrb.Spatial2DView(name=name, origin=schema.pinhole_path(rig, cam), contents=f"{schema.pinhole_path(rig, cam)}/**")


def sensor_plot(name: str, origin: str, contents: str) -> rrb.TimeSeriesView:
    """One sensor channel's plot pane, with the legend visible.

    The legend is on because the channels are three unlabelled scalars each;
    without it a reader cannot tell the axes apart.

    Args:
        name: Pane label (``"Gyroscope"``, ``"Magnetometer"``, …).
        origin: Sensor node the pane is rooted at.
        contents: Channel entity to plot beneath it.

    Returns:
        The plot pane the bottom row of a rig layout is built from.
    """
    return rrb.TimeSeriesView(name=name, origin=origin, contents=contents, plot_legend=rrb.PlotLegend(visible=True))


def eye_controls_from_pose(
    position: tuple[float, float, float],
    look_target: tuple[float, float, float],
    eye_up: tuple[float, float, float],
) -> rrb.EyeControls3D:
    """An orbital eye placed by hand, in the coordinates of the view's origin.

    Orbital is the dataforge default for every 3D view: the eye orbits
    ``look_target`` when dragged, which is what a person inspecting a rig or a
    scene wants, whereas first-person controls fly the eye and lose the subject.
    The follow view's origin is the rig node, so an eye fixed in that frame rides
    the rig (the ``rerun-io/eye_control_example`` pattern).

    Args:
        position: Where the eye sits.
        look_target: What it aims at and orbits around.
        eye_up: Which way is up for the eye; it fixes the horizon's roll.

    Returns:
        The eye both 3D views of a rig layout can be given.
    """
    # EyeControls3D is marked unstable by the SDK; re-validate this factory on Rerun bumps.
    return rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=position,
        look_target=look_target,
        eye_up=eye_up,
        spin_speed=0.0,
    )


def follow_eye_controls(
    forward: tuple[float, float, float],
    up: tuple[float, float, float],
    *,
    back_m: float = FOLLOW_BACK_M,
    up_m: float = FOLLOW_UP_M,
    ahead_m: float = FOLLOW_AHEAD_M,
) -> rrb.EyeControls3D:
    """A chase camera derived from the device's own forward and up.

    It sits ``back_m`` behind the device and ``up_m`` above it and aims
    ``ahead_m`` in front of it, which keeps the device itself and the ground it
    moves over both in shot. Every direction comes from the device's frame
    because devices carry their rig frames at different orientations, so one
    hand-picked position cannot serve them all.

    Args:
        forward: Where the device looks, in the rig frame; unit length.
        up: The device's up, in the rig frame; unit length.
        back_m: Distance behind the device, along ``forward``.
        up_m: Distance above it, along ``up``.
        ahead_m: Distance in front of it that the eye aims at.

    Returns:
        The eye the Follow view of both blueprints uses.
    """
    position: tuple[float, float, float] = (
        -back_m * forward[0] + up_m * up[0],
        -back_m * forward[1] + up_m * up[1],
        -back_m * forward[2] + up_m * up[2],
    )
    look_target: tuple[float, float, float] = (ahead_m * forward[0], ahead_m * forward[1], ahead_m * forward[2])
    return eye_controls_from_pose(position, look_target, up)


def rig_blueprint(
    camera_panes: Sequence[rrb.Spatial2DView],
    *,
    rig: int,
    run_source: str,
    eye_controls: rrb.EyeControls3D,
    plots: Sequence[rrb.TimeSeriesView],
    extra_run_sources: Sequence[str] = (),
) -> rrb.Blueprint:
    """The default layout of a single-rig capture: 3D and cameras over the sensor plots.

    Two 3D views on purpose. The overview is rooted at the world and shows the
    whole derived path with its cursor trail hidden; the follow view is rooted at
    the rig and highlights the last ``TRAIL_WINDOW_S`` of motion over the same
    path drawn thin and dim, so the two together read as "where it went" and
    "where it is going".
    Only the overview draws the ground grid: the grid lives in the view origin's
    frame, and in the rig frame there is no fixed ground plane to draw it on.

    Args:
        camera_panes: One pane per camera, in display order.
        rig: Rig index the follow view is rooted at.
        run_source: Processing source whose trajectory and trail the overrides
            name (``"gt"``, ``"basalt"``, …).
        eye_controls: Eye of the follow view.
        plots: Sensor plot panes for the bottom row.
        extra_run_sources: Further run sources whose trajectory and trail get the
            same overrides, for a dataset that carries more than one pose layer.

    Returns:
        The blueprint embedded in the dataset's rrds and registered as the
        catalog dataset's default.
    """
    sources: tuple[str, ...] = (run_source, *extra_run_sources)
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="Rig",
                        origin="/",
                        line_grid=True,
                        overrides={
                            # The rig is centimetres on a path that can be kilometres, so the
                            # overview marks its current pose with a screen-space dot: a point
                            # at the rig's own origin rides the rig transform, and a UI-point
                            # radius keeps the same pixel size at any zoom. Display only: no
                            # recorded geometry or calibration changes, and Follow never sees it.
                            schema.rig_path(rig): rr.Points3D(
                                [[0.0, 0.0, 0.0]], radii=rr.Radius.ui_points(9.0), colors=[255, 220, 0], labels=["rig"], show_labels=True
                            ),
                            **{schema.trail_path(source): rrb.EntityBehavior(visible=False) for source in sources},
                        },
                    ),
                    rrb.Spatial3DView(
                        name="Follow",
                        origin=schema.rig_path(rig),
                        contents="/**",
                        # No grid: LineGrid3D draws on the origin frame's z = 0 plane,
                        # and this view's origin is the rig, whose z axis is nowhere
                        # near vertical on a worn device — the grid stood as a wall
                        # beside the trail. The world ground is a moving plane in the
                        # rig frame, so no fixed plane setting can represent it; the
                        # overview above keeps the real ground.
                        line_grid=False,
                        overrides={
                            # Not hidden: with the path gone the highlighted trail
                            # floated with nothing to place it against, so it stays
                            # as thin dim context underneath.
                            **{
                                schema.trajectory_path(source): rr.LineStrips3D.from_fields(
                                    colors=DIM_TRAJECTORY_COLOR, radii=rr.Radius.ui_points(DIM_TRAJECTORY_RADIUS_UI_POINTS)
                                )
                                for source in sources
                            },
                            **{
                                schema.trail_path(source): rrb.VisibleTimeRanges(
                                    rrb.VisibleTimeRange(
                                        schema.TIMELINE,
                                        start=rrb.TimeRangeBoundary.cursor_relative(seconds=TRAIL_WINDOW_S),
                                        end=rrb.TimeRangeBoundary.cursor_relative(),
                                    )
                                )
                                for source in sources
                            },
                        },
                        eye_controls=eye_controls,
                    ),
                ),
                rrb.Grid(*camera_panes, grid_columns=CAMERA_GRID_COLUMNS, name="Synchronized cameras"),
                column_shares=list(RIG_COLUMN_SHARES),
            ),
            rrb.Horizontal(*plots),
            row_shares=list(PLOT_ROW_SHARES),
        ),
        rrb.TimePanel(timeline=schema.TIMELINE),
        collapse_panels=True,
    )


def table_blueprint(
    num_cameras: int,
    *,
    rig: int,
    run_source: str,
    eye_controls: rrb.EyeControls3D,
    front_pane: rrb.Spatial2DView,
) -> rrb.Blueprint:
    """The segment-table preview card: the follow-framed rig beside one video pane.

    Every visible table row renders through this at once, so exactly one stream
    is decoded: the other cameras' videos and the motion trail are **excluded**
    rather than hidden, because a hidden entity is still decoded.

    Args:
        num_cameras: Cameras the rig carries; all their video is excluded, and
            the one pane that decodes is ``front_pane``.
        rig: Rig index the 3D view is rooted at.
        run_source: Processing source whose trail is excluded.
        eye_controls: Eye of the 3D view.
        front_pane: The single 2D pane the card decodes.

    Returns:
        The blueprint registered with ``segment_table=True``.
    """
    video_exclusions: list[str] = [f"- {schema.video_path(rig, index)}/**" for index in range(num_cameras)]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Follow",
                origin=schema.rig_path(rig),
                contents=["/**", *video_exclusions, f"- {schema.trail_path(run_source)}/**"],
                line_grid=False,  # rig-rooted, same reason as rig_blueprint's Follow view
                eye_controls=eye_controls,
            ),
            front_pane,
            column_shares=list(RIG_COLUMN_SHARES),
        ),
        rrb.TimePanel(timeline=schema.TIMELINE),
    )
