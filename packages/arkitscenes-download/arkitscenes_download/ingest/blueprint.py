"""Default viewer layout for an ingested sequence."""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from simplecv.rerun_log_utils import orbit_eye_position

from arkitscenes_download.ingest.paths import IMU_ACCEL, IMU_GYRO, PINHOLE_ULTRAWIDE, PINHOLE_WIDE, PINHOLE_WIDE_LOWRES, WORLD

FIT_DISTANCE_FACTOR: float = 2.2
"""Eye distance per bounding-sphere radius that keeps the whole mesh in frame (tuned visually)."""
EXTRA_ZOOM_OUT: float = 1.25
"""Additional margin beyond the tight fit."""
SPIN_SPEED: float = 0.25
"""Slow orbit around the look target."""


def _eye_controls(mesh_center_xyz: Float64[np.ndarray, "3"] | None, bounding_radius_m: float | None) -> rrb.archetypes.EyeControls3D:
    """Frame the GT mesh when its geometry is known; otherwise keep the viewer's default framing."""
    if mesh_center_xyz is None or bounding_radius_m is None:
        return rrb.archetypes.EyeControls3D(spin_speed=SPIN_SPEED)
    return rrb.archetypes.EyeControls3D(
        kind=rrb.components.Eye3DKind.Orbital,
        position=orbit_eye_position(mesh_center_xyz, bounding_radius_m, FIT_DISTANCE_FACTOR * EXTRA_ZOOM_OUT),
        look_target=mesh_center_xyz,
        eye_up=(0.0, 0.0, 1.0),
        spin_speed=SPIN_SPEED,
    )


def _imu_axis() -> rrb.archetypes.TimeAxis:
    """Keep plot X axes to a cursor-centered window instead of the full recording."""
    window: rr.datatypes.TimeRange = rr.datatypes.TimeRange(
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-5.0),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=5.0),
    )
    return rrb.archetypes.TimeAxis(view_range=window)


def make_blueprint(
    portrait: bool = False,
    mesh_center_xyz: Float64[np.ndarray, "3"] | None = None,
    bounding_radius_m: float | None = None,
) -> rrb.Blueprint:
    """Build the multi-camera, depth, and IMU layout.

    The layout is identical for both orientations; only the column widths
    change. Portrait camera views are tall and narrow, so the 3D world view
    takes a wider share of the row. When the GT mesh geometry is provided
    (per-sequence embedded blueprints), the 3D eye orbits the mesh center
    with the whole mesh in frame; dataset-default blueprints omit it.
    """
    column_shares: list[float] = [2.6, 1.0, 1.0] if portrait else [2.0, 1.0, 1.0]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="world", origin=f"/{WORLD}", eye_controls=_eye_controls(mesh_center_xyz, bounding_radius_m)),
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
