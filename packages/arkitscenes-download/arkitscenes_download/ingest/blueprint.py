"""Default viewer layout for an ingested sequence."""

import rerun.blueprint as rrb

from arkitscenes_download.ingest.paths import IMU_ACCEL, IMU_GYRO, PINHOLE_ULTRAWIDE, PINHOLE_WIDE, PINHOLE_WIDE_LOWRES, WORLD


def make_blueprint() -> rrb.Blueprint:
    """Build the multi-camera, depth, and IMU layout."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="world", origin=f"/{WORLD}"),
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
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="gyro", origin=IMU_GYRO),
                rrb.TimeSeriesView(name="accel", origin=IMU_ACCEL),
            ),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
