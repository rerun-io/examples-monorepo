"""Rerun blueprint shared across all slam-evals recordings.

With RGB and depth sharing the same pinhole tree (see
``slam_evals/ingest/sequence.py`` module docstring), a single 2D view at
``/world/body/cam_<i>/pinhole`` shows the VideoStream with the DepthImage
automatically overlaid for rgbd modalities. Views whose origin entity
doesn't exist in a given recording are silently empty.
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
