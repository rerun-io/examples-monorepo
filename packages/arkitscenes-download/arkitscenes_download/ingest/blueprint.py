"""Default viewer layout for an ingested sequence."""

from typing import TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from simplecv.rerun_log_utils import orbit_eye_position

from arkitscenes_download.ingest.paths import (
    CONFIDENCE,
    DEPTH,
    DEPTH_GT,
    DEPTH_PROMPTDA,
    GT_MESH,
    IMU_ACCEL,
    IMU_GYRO,
    PINHOLE_ULTRAWIDE,
    PINHOLE_WIDE,
    PINHOLE_WIDE_LOWRES,
    PROMPTDA_MESH,
    TIMELINE,
    VIDEO_ULTRAWIDE,
    VIDEO_WIDE,
    WORLD,
)

# TODO(rerun#upstream): remove once the viewer auto-ranges encoded depth from the
# decoded values instead of the PNG bit depth (broken in 0.34.1, no issue filed yet).
# Delete this constant, the per-view overrides below, and the `depth_range=` kwargs
# at every EncodedDepthImage logging site — each is marked with a TODO pointing here
# (ingest/cli.py, prompt_da_arkitscenes.py, prompt_da_arkitscenes_raw.py).
DEPTH_RANGE_MM: tuple[float, float] = (0.0, 4000.0)
"""Colormap range for the depth views, in native mm units.

Without it the viewer guesses the range from the PNG bit depth (0-65535 for
uint16), squashing real indoor depths into the bottom of the colormap — a
near-flat dark purple. Sampled data: lowres ARKit depth is mostly <=4 m
(rarely up to ~5 m), GT laser depth <=3.5 m; pixels beyond clamp to the
colormap max, which stays readable.
"""

FIT_DISTANCE_FACTOR: float = 2.2
"""Eye distance per bounding-sphere radius that keeps the whole mesh in frame (tuned visually)."""
EXTRA_ZOOM_OUT: float = 1.25
"""Additional margin beyond the tight fit."""
SPIN_SPEED: float = 0.25
"""Slow orbit around the look target."""


MeshFraming: TypeAlias = tuple[Float64[np.ndarray, "3"], float]
"""GT-mesh framing geometry: (world-frame AABB center, bounding-sphere radius in metres)."""


def _eye_controls(framing: MeshFraming | None) -> rrb.archetypes.EyeControls3D:
    """Frame the GT mesh when its geometry is known; otherwise keep the viewer's default framing."""
    if framing is None:
        return rrb.archetypes.EyeControls3D(spin_speed=SPIN_SPEED)
    mesh_center_xyz, bounding_radius_m = framing
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


def make_table_blueprint() -> rrb.Blueprint:
    """3D-only preview layout for the dataset's segment table (experimental table cards).

    Registered with ``DatasetEntry.register_blueprint(uri, segment_table=True)``; the
    viewer renders each segment-table row through this view when "Table cards and
    blueprints" is enabled under Settings > Experimental. Kept to a single 3D view
    showing only mesh, boxes, and camera frusta: with ~15 cards visible, the AV1
    streams and per-frame depth/confidence image uploads dominated the frame
    (profiled: dav1d decode saturated ~12 cores, ~7 FPS; excluding them restores
    interactive rates while the preview looks nearly identical).
    """
    return rrb.Blueprint(
        rrb.Spatial3DView(
            name="world",
            origin=f"/{WORLD}",
            contents=[
                "$origin/**",
                f"- /{VIDEO_WIDE}/**",
                f"- /{VIDEO_ULTRAWIDE}/**",
                f"- /{DEPTH_GT}/**",
                f"- /{DEPTH}/**",
                f"- /{CONFIDENCE}/**",
            ],
            eye_controls=_eye_controls(None),
        ),
        rrb.TimePanel(timeline=TIMELINE),
    )


def make_blueprint(portrait: bool = False, framing: MeshFraming | None = None, include_promptda: bool = False) -> rrb.Blueprint:
    """Build the multi-camera, depth, and IMU layout.

    The layout is identical for both orientations; only the column widths
    change. Portrait camera views are tall and narrow, so the 3D world view
    takes a wider share of the row. When the GT mesh framing is provided
    (per-sequence embedded blueprints), the 3D eye orbits the mesh center
    with the whole mesh in frame; dataset-default blueprints omit it.

    With ``include_promptda`` the depth tabs gain an active "depth PromptDA"
    tab and the 3D area becomes tabs: the stock "world" plus a
    "world PromptDA" that swaps the ARKit GT mesh (and the other depth
    backprojections) for the PromptDA depth + TSDF-fused mesh.
    """
    world_view = rrb.Spatial3DView(name="world", origin=f"/{WORLD}", eye_controls=_eye_controls(framing))
    world_area: rrb.Spatial3DView | rrb.Tabs = world_view
    depth_range = rr.EncodedDepthImage.from_fields(depth_range=DEPTH_RANGE_MM)
    depth_tabs = [
        rrb.Spatial2DView(name="depth ARKit", origin=PINHOLE_WIDE_LOWRES, contents=["$origin/depth"], overrides={f"/{DEPTH}": depth_range}),
        rrb.Spatial2DView(name="depth GT (laser)", origin=PINHOLE_WIDE, contents=["$origin/depth_gt"], overrides={f"/{DEPTH_GT}": depth_range}),
    ]
    active_depth_tab: int = 0
    if include_promptda:
        world_view = rrb.Spatial3DView(
            name="world",
            origin=f"/{WORLD}",
            contents=["$origin/**", f"- /{PROMPTDA_MESH}/**", f"- /{DEPTH_PROMPTDA}/**"],
            eye_controls=_eye_controls(framing),
        )
        depth_tabs.append(
            rrb.Spatial2DView(
                name="depth PromptDA", origin=PINHOLE_WIDE, contents=["$origin/depth_promptda"], overrides={f"/{DEPTH_PROMPTDA}": depth_range}
            )
        )
        active_depth_tab = len(depth_tabs) - 1
        world_area = rrb.Tabs(
            world_view,
            rrb.Spatial3DView(
                name="world PromptDA",
                origin=f"/{WORLD}",
                contents=["$origin/**", f"- /{GT_MESH}/**", f"- /{DEPTH_GT}/**", f"- /{PINHOLE_WIDE_LOWRES}/**"],
                eye_controls=_eye_controls(framing),
            ),
            active_tab=0,
        )
    column_shares: list[float] = [2.6, 1.0, 1.0] if portrait else [2.0, 1.0, 1.0]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                world_area,
                rrb.Vertical(
                    rrb.Spatial2DView(name="wide", origin=PINHOLE_WIDE, contents=["$origin/video", f"/{WORLD}/gt/boxes/**"]),
                    rrb.Spatial2DView(name="ultrawide", origin=PINHOLE_ULTRAWIDE, contents=["$origin/video", f"/{WORLD}/gt/boxes/**"]),
                ),
                rrb.Vertical(
                    rrb.Tabs(*depth_tabs, active_tab=active_depth_tab),
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
