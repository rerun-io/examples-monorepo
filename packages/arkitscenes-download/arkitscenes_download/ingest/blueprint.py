"""Default viewer layout for an ingested sequence."""

import rerun as rr
import rerun.blueprint as rrb

from arkitscenes_download.ingest.paths import IMU_ACCEL, IMU_GYRO, PINHOLE_ULTRAWIDE, PINHOLE_WIDE, PINHOLE_WIDE_LOWRES, WORLD


def _imu_axis() -> rrb.archetypes.TimeAxis:
    """Keep plot X axes to a cursor-centered window instead of the full recording."""
    window: rr.datatypes.TimeRange = rr.datatypes.TimeRange(
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-5.0),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=5.0),
    )
    return rrb.archetypes.TimeAxis(view_range=window)


def make_blueprint(portrait: bool = False) -> rrb.Blueprint:
    """Build the multi-camera, depth, and IMU layout.

    The layout is identical for both orientations; only the column widths
    change. Portrait camera views are tall and narrow, so the 3D world view
    takes a wider share of the row.
    """
    column_shares: list[float] = [2.6, 1.0, 1.0] if portrait else [2.0, 1.0, 1.0]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="world", origin=f"/{WORLD}", eye_controls=rrb.archetypes.EyeControls3D(spin_speed=0.25)),
                rrb.Vertical(
                    rrb.Spatial2DView(name="wide", origin=PINHOLE_WIDE, contents=["$origin/video", f"/{WORLD}/gt/boxes/**"]),
                    rrb.Spatial2DView(name="ultrawide", origin=PINHOLE_ULTRAWIDE, contents=["$origin/video", f"/{WORLD}/gt/boxes/**"]),
                ),
                rrb.Vertical(
                    rrb.Tabs(
                        rrb.Spatial2DView(name="depth ARKit", origin=PINHOLE_WIDE_LOWRES, contents=["$origin/depth"]),
                        rrb.Spatial2DView(name="depth GT (laser)", origin=PINHOLE_WIDE, contents=["$origin/depth_gt"]),
                    ),
                    rrb.Spatial2DView(name="confidence", origin=PINHOLE_WIDE_LOWRES, contents=["$origin/confidence"]),
                ),
                column_shares=column_shares,
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="gyro", origin=IMU_GYRO, axis_x=_imu_axis()),
                rrb.TimeSeriesView(name="accel", origin=IMU_ACCEL, axis_x=_imu_axis()),
            ),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
