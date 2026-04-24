"""Rerun blueprint shared across all slam-evals recordings.

Panels for entities that don't exist in a given recording are silently empty,
so one blueprint covers every modality (mono / stereo / rgbd, with or without
IMU) without dispatching on the recording's metadata.
"""

from __future__ import annotations

import rerun.blueprint as rrb


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="world", origin="/world"),
                rrb.Vertical(
                    rrb.Spatial2DView(name="cam_0", origin="/world/body/cam_0/pinhole"),
                    rrb.Spatial2DView(name="cam_1", origin="/world/body/cam_1/pinhole"),
                    rrb.Spatial2DView(name="depth", origin="/world/body/cam_depth"),
                ),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="gyro", origin="/imu/gyro"),
                rrb.TimeSeriesView(name="accel", origin="/imu/accel"),
            ),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
