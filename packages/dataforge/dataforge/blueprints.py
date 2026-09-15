"""Shared exoego rig, camera, trajectory and IMU presentation."""

import rerun as rr
import rerun.blueprint as rrb
from rerun.encodings import EntityPathLike

from dataforge import schema


def build_rig_blueprint(
    camera_names: list[str],
    *,
    pose_sources: tuple[str, ...] = ("basalt", "slam_rs"),
    follow_eye: rrb.EyeControls3D | None = None,
) -> rrb.Blueprint:
    """Show rig_00 and imu_00; adapters may supply rig-specific follow-eye settings."""
    camera_views: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(name=name, origin=schema.pinhole_path(0, index), contents=f"{schema.pinhole_path(0, index)}/**")
        for index, name in enumerate(camera_names)
    ]
    overview_overrides: dict[EntityPathLike, rrb.EntityBehavior | rrb.VisibleTimeRanges | rr.Points3D] = {
        # The rig is centimetres on a path that can be kilometres, so the overview marks
        # its current pose with a screen-space dot: a point at the rig's own origin rides
        # the rig transform, and a UI-point radius keeps the same pixel size at any zoom.
        # Display only: no recorded geometry or calibration changes, and Follow never sees it.
        schema.rig_path(0): rr.Points3D([[0.0, 0.0, 0.0]], radii=rr.Radius.ui_points(9.0), colors=[255, 220, 0], labels=["rig"], show_labels=True),
    }
    follow_overrides: dict[EntityPathLike, rrb.EntityBehavior | rrb.VisibleTimeRanges] = {}
    for source in pose_sources:
        overview_overrides[schema.trail_path(source)] = rrb.EntityBehavior(visible=False)
        overview_overrides[schema.trajectory_path(source)] = rrb.VisibleTimeRanges(
            rrb.VisibleTimeRange("video_time", start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.cursor_relative()),
        )
        follow_overrides[schema.trajectory_path(source)] = rrb.EntityBehavior(visible=False)
        follow_overrides[schema.trail_path(source)] = rrb.VisibleTimeRanges(
            rrb.VisibleTimeRange(
                "video_time", start=rrb.TimeRangeBoundary.cursor_relative(seconds=-10.0), end=rrb.TimeRangeBoundary.cursor_relative(),
            ),
        )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="Rig",
                        origin="/",
                        line_grid=True,
                        # The overview shows the whole SLAM path, while its trail stays hidden.
                        # Overrides on entities a base-only recording lacks are simply inert.
                        overrides=overview_overrides,
                    ),
                    # Follow-cam (rerun-io/eye_control_example pattern): the view's
                    # origin IS the rig frame, so a fixed first-person eye in that
                    # frame rides the rig. Inert until a pose layer animates rig_00.
                    rrb.Spatial3DView(
                        name="Follow",
                        origin=schema.rig_path(0),
                        contents="/**",
                        line_grid=True,
                        # The follow view hides the full path and shows only a 10 s
                        # cursor-relative trail. The window is a viewer setting.
                        overrides=follow_overrides,
                        eye_controls=follow_eye,
                    ),
                ),
                rrb.Grid(*camera_views, grid_columns=2, name="Synchronized cameras"),
                column_shares=[3, 2],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Gyroscope",
                    origin=schema.imu_path(0, 0),
                    contents=schema.gyro_path(0, 0),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    name="Accelerometer",
                    origin=schema.imu_path(0, 0),
                    contents=schema.accel_path(0, 0),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
            ),
            row_shares=[3, 1],
        ),
        rrb.TimePanel(timeline="video_time"),
        collapse_panels=True,
    )
