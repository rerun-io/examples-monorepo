"""Rerun blueprint shared across all slam-evals recordings.

Entity paths follow the COLMAP-aligned schema documented in
``docs/schema.md``: rig at ``/world/rig_0``, sensors as peer children
(``cam_<i>``, ``imu_<i>``), and per-sensor data nested under each sensor's
pinhole or scalar entity.

With RGB and depth sharing the same pinhole tree, a single 2D view at
``/world/rig_0/cam_<i>/pinhole`` shows the VideoStream with the
EncodedDepthImage automatically overlaid for rgbd modalities. Views whose
origin entity doesn't exist in a given segment are silently empty.
"""

from __future__ import annotations

import rerun.blueprint as rrb


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="world", origin="/world"),
                rrb.Vertical(
                    rrb.Spatial2DView(name="cam_0", origin="/world/rig_0/cam_0/pinhole"),
                    rrb.Spatial2DView(name="cam_1", origin="/world/rig_0/cam_1/pinhole"),
                ),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="gyro", origin="/world/rig_0/imu_0/gyro"),
                rrb.TimeSeriesView(name="accel", origin="/world/rig_0/imu_0/accel"),
            ),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
