"""Default viewer layout for an ingested sequence."""

from typing import TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from simplecv.rerun_log_utils import orbit_eye_position

from arkitscenes_download.ingest.paths import (
    ARKIT_MESH,
    CONFIDENCE,
    DEPTH,
    DEPTH_PROMPTDA,
    DEPTH_SPLAT_ULTRAWIDE,
    DEPTH_SPLAT_WIDE,
    DEPTH_ULTRAWIDE_RECT,
    ERROR_DEPTH_ULTRAWIDE_RECT,
    ERROR_NORMAL_ULTRAWIDE_RECT,
    ERROR_RGB_ULTRAWIDE_RECT,
    GT,
    GT_BOXES,
    GT_DEPTH,
    GT_PINHOLE_WIDE,
    IMU_ACCEL,
    IMU_GYRO,
    NORMALS_MOGE,
    NORMALS_SPLAT_ULTRAWIDE_RECT,
    NORMALS_SPLAT_WIDE,
    NORMALS_ULTRAWIDE_RECT,
    PINHOLE_MOGE,
    PINHOLE_ULTRAWIDE,
    PINHOLE_ULTRAWIDE_RECT,
    PINHOLE_WIDE,
    PINHOLE_WIDE_LOWRES,
    PROMPTDA,
    PROMPTDA_MESH,
    RGB_SPLAT_ULTRAWIDE_RECT,
    RGB_ULTRAWIDE_RECT,
    RIG,
    SHARPNESS_ULTRAWIDE,
    SHARPNESS_WIDE,
    SPLATS,
    TIMELINE,
    VIDEO_ULTRAWIDE,
    VIDEO_WIDE,
    WORLD,
)
from arkitscenes_download.schema import DEPTH_RANGE_MM

# TODO(rerun#upstream): remove once the viewer auto-ranges encoded depth from the
# decoded values instead of the PNG bit depth (broken in 0.34.1, no issue filed yet).
# Delete this constant, the per-view overrides below, and the `depth_range=` kwargs
# at every EncodedDepthImage logging site — each is marked with a TODO pointing here
# (ingest/cli.py, prompt_da_arkitscenes.py, prompt_da_arkitscenes_raw.py).
FIT_DISTANCE_FACTOR: float = 2.2
"""Eye distance per bounding-sphere radius that keeps the whole mesh in frame (tuned visually)."""
EXTRA_ZOOM_OUT: float = 1.25
"""Additional margin beyond the tight fit."""
SPIN_SPEED: float = 0.25
"""Slow orbit around the look target."""
INSIDE_DISTANCE_FACTOR: float = 0.12
"""Eye offset per bounding-sphere radius for the Fit-triage splat view, orbiting just off the scene center.

Mesh views render front faces only, so an outside orbit still sees into the room; a splat is
view-blocking from every side, so splat-containing views need an inside or chase eye instead."""


MeshFraming: TypeAlias = tuple[Float64[np.ndarray, "3"], float]
"""ARKit-mesh framing geometry: (world-frame AABB center, bounding-sphere radius in metres)."""


def _orbit_eye_controls(
    framing: MeshFraming | None,
    distance_factor: float,
    direction_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> rrb.archetypes.EyeControls3D:
    """Orbit the scene center at the given distance when framing is known; otherwise keep the viewer default."""
    if framing is None:
        return rrb.archetypes.EyeControls3D(spin_speed=SPIN_SPEED)
    mesh_center_xyz, bounding_radius_m = framing
    return rrb.archetypes.EyeControls3D(
        kind=rrb.components.Eye3DKind.Orbital,
        position=orbit_eye_position(mesh_center_xyz, bounding_radius_m, distance_factor, direction_xyz=direction_xyz),
        look_target=mesh_center_xyz,
        eye_up=(0.0, 0.0, 1.0),
        spin_speed=SPIN_SPEED,
    )


def _eye_controls(framing: MeshFraming | None) -> rrb.archetypes.EyeControls3D:
    """Frame the whole mesh from outside (front-face rendering lets this see into the room)."""
    return _orbit_eye_controls(framing, FIT_DISTANCE_FACTOR * EXTRA_ZOOM_OUT)


CHASE_OFFSET_RIG_XYZ: tuple[float, float, float] = (0.0, -0.6, -2.2)
"""Chase-eye offset in the rig's RDF camera frame: slightly above (-y) and behind (-z) the camera."""


def _follow_eye_controls() -> rrb.archetypes.EyeControls3D:
    """Third-person chase eye, fixed in the rig frame so it moves and turns with the camera.

    Pairs with a view whose ``origin`` is the moving rig entity: the eye pose is expressed
    in that frame, so as the rig's world transform updates each frame, the eye rides along
    behind it — no live blueprint logging needed (the NTU VIRAL pattern from
    rerun-io/eye_control_example).

    TODO(rerun#upstream): replace with ``EyeControls3D(tracking_entity=...)`` once its
    follow-at-offset modes engage from a static blueprint — in 0.36.0a1 they silently
    never latch (only FirstPerson on a camera entity works); no issue filed yet.
    """
    return rrb.archetypes.EyeControls3D(
        kind=rrb.components.Eye3DKind.Orbital,
        position=CHASE_OFFSET_RIG_XYZ,
        look_target=(0.0, 0.0, 0.0),
        eye_up=(0.0, -1.0, 0.0),
    )


def _inside_eye_controls(framing: MeshFraming | None) -> rrb.archetypes.EyeControls3D:
    """Orbit just off the scene center at eye height, so the Fit-triage splat spins around the viewer."""
    return _orbit_eye_controls(framing, INSIDE_DISTANCE_FACTOR, direction_xyz=(1.0, 1.0, 0.0))


def _depth_view(name: str, origin: str, entity_path: str) -> rrb.Spatial2DView:
    """One encoded-depth view with the explicit colormap range the viewer cannot infer (see ``DEPTH_RANGE_MM``)."""
    return rrb.Spatial2DView(
        name=name,
        origin=origin,
        contents=[f"/{entity_path}"],
        overrides={f"/{entity_path}": rr.EncodedDepthImage.from_fields(depth_range=DEPTH_RANGE_MM)},
    )


def _imu_axis() -> rrb.archetypes.TimeAxis:
    """Keep plot X axes to a cursor-centered window instead of the full recording."""
    window: rr.encodings.TimeRange = rr.encodings.TimeRange(
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
                f"- /{GT_DEPTH}/**",
                f"- /{DEPTH}/**",
                f"- /{CONFIDENCE}/**",
            ],
            eye_controls=_eye_controls(None),
        ),
        rrb.TimePanel(timeline=TIMELINE),
    )


def make_capture_page(portrait: bool = False, framing: MeshFraming | None = None) -> rrb.Vertical:
    """Build the Capture page with camera, depth, confidence, and IMU views."""
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="world",
                origin=f"/{WORLD}",
                contents=["$origin/**", f"- /{GT}/**", f"- /{SPLATS}/**", f"- /{PROMPTDA}/**"],
                eye_controls=_eye_controls(framing),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="wide", origin=PINHOLE_WIDE, contents=["$origin/video", f"/{GT_BOXES}/**"]),
                rrb.Spatial2DView(name="ultrawide", origin=PINHOLE_ULTRAWIDE, contents=["$origin/video", f"/{GT_BOXES}/**"]),
            ),
            rrb.Vertical(
                _depth_view("depth ARKit", PINHOLE_WIDE_LOWRES, DEPTH),
                rrb.Spatial2DView(name="confidence", origin=PINHOLE_WIDE_LOWRES, contents=["$origin/confidence"]),
            ),
            column_shares=[2.6, 1.0, 1.0] if portrait else [2.0, 1.0, 1.0],
        ),
        rrb.Horizontal(
            rrb.TimeSeriesView(name="gyro", origin=IMU_GYRO, axis_x=_imu_axis()),
            rrb.TimeSeriesView(name="accel", origin=IMU_ACCEL, axis_x=_imu_axis()),
        ),
        row_shares=[3, 1],
        name="Capture",
    )


def make_gt_page(portrait: bool = False, framing: MeshFraming | None = None) -> rrb.Horizontal:
    """Build the ground-truth page."""
    return rrb.Horizontal(
        rrb.Spatial3DView(
            name="world GT",
            origin=f"/{WORLD}",
            contents=["$origin/**", f"- /{ARKIT_MESH}/**", f"- /{GT_BOXES}/**", f"- /{DEPTH}/**", f"- /{CONFIDENCE}/**"],
            eye_controls=_eye_controls(framing),
        ),
        rrb.Vertical(
            rrb.Spatial2DView(name="wide", origin=PINHOLE_WIDE, contents=["$origin/video", f"/{GT_BOXES}/**"]),
            _depth_view("depth GT (laser)", GT_PINHOLE_WIDE, GT_DEPTH),
            rrb.Spatial2DView(name="ultrawide", origin=PINHOLE_ULTRAWIDE, contents=["$origin/video", f"/{GT_BOXES}/**"]),
        ),
        column_shares=[2.6, 1.0] if portrait else [2.0, 1.0],
        name="GT",
    )


def make_promptda_page(portrait: bool = False, framing: MeshFraming | None = None) -> rrb.Horizontal:
    """Build the PromptDA comparison page."""
    return rrb.Horizontal(
        rrb.Spatial3DView(
            name="world PromptDA",
            origin=f"/{WORLD}",
            contents=["$origin/**", f"- /{ARKIT_MESH}/**", f"- /{GT_DEPTH}/**", f"- /{PINHOLE_WIDE_LOWRES}/**"],
            eye_controls=_eye_controls(framing),
        ),
        rrb.Vertical(
            rrb.Spatial2DView(name="wide video", origin=PINHOLE_WIDE, contents=["$origin/video"]),
            _depth_view("depth PromptDA", PINHOLE_WIDE, DEPTH_PROMPTDA),
            _depth_view("depth ARKit", PINHOLE_WIDE_LOWRES, DEPTH),
        ),
        column_shares=[2.6, 1.0] if portrait else [2.0, 1.0],
        name="PromptDA",
    )


def make_splat_page(portrait: bool = False) -> rrb.Vertical:
    """Build the rendered-splat inspection page."""
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="world Splat",
                # Rooting the view at the moving rig makes the chase eye ride along with it;
                # the wide frustum + its video plane keep the followed camera visible.
                origin=f"/{RIG}",
                contents=[f"/{SPLATS}/**", f"/{PROMPTDA_MESH}", f"/{RIG}", f"/{PINHOLE_WIDE}", f"/{VIDEO_WIDE}"],
                eye_controls=_follow_eye_controls(),
                # The ghost mesh stays in the view (toggle it on via the eye icon) but is hidden by
                # default: seen from inside, the splat is constantly viewed *through* it.
                overrides={
                    f"/{PROMPTDA_MESH}": [
                        rrb.EntityBehavior(visible=False),
                        rr.Mesh3D.from_fields(albedo_factor=(0, 255, 0, 96)),
                    ]
                },
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="wide video", origin=PINHOLE_WIDE, contents=["$origin/video"]),
                rrb.Tabs(
                    rrb.Spatial2DView(name="video", origin=PINHOLE_ULTRAWIDE, contents=["$origin/video"]),
                    rrb.Spatial2DView(name="rect RGB", origin=PINHOLE_ULTRAWIDE_RECT, contents=[f"/{RGB_ULTRAWIDE_RECT}"]),
                    rrb.Spatial2DView(name="splat RGB", origin=PINHOLE_ULTRAWIDE_RECT, contents=[f"/{RGB_SPLAT_ULTRAWIDE_RECT}"]),
                    active_tab=0,
                ),
            ),
            rrb.Vertical(
                rrb.Tabs(
                    _depth_view("splat depth", PINHOLE_WIDE, DEPTH_SPLAT_WIDE),
                    _depth_view("PromptDA depth", PINHOLE_WIDE, DEPTH_PROMPTDA),
                    active_tab=0,
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(
                        name="splat normals",
                        origin=PINHOLE_WIDE,
                        contents=[f"/{NORMALS_SPLAT_WIDE}"],
                    ),
                    rrb.Spatial2DView(name="MoGe normals", origin=PINHOLE_MOGE, contents=[f"/{NORMALS_MOGE}"]),
                    active_tab=0,
                ),
                rrb.Tabs(
                    _depth_view("splat depth", PINHOLE_ULTRAWIDE_RECT, DEPTH_SPLAT_ULTRAWIDE),
                    _depth_view("raycast depth", PINHOLE_ULTRAWIDE_RECT, DEPTH_ULTRAWIDE_RECT),
                    active_tab=0,
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(
                        name="splat normals",
                        origin=PINHOLE_ULTRAWIDE_RECT,
                        contents=[f"/{NORMALS_SPLAT_ULTRAWIDE_RECT}"],
                    ),
                    rrb.Spatial2DView(
                        name="MoGe rect normals",
                        origin=PINHOLE_ULTRAWIDE_RECT,
                        contents=[f"/{NORMALS_ULTRAWIDE_RECT}"],
                    ),
                    active_tab=0,
                ),
            ),
            column_shares=[3.4, 2.0, 1.0] if portrait else [4.8, 2.2, 1.0],
        ),
        rrb.TimeSeriesView(
            name="sharpness",
            origin=f"/{WORLD}",
            contents=[f"/{SHARPNESS_WIDE}", f"/{SHARPNESS_ULTRAWIDE}"],
            axis_x=_imu_axis(),
        ),
        row_shares=[6, 1],
        name="Splat",
    )


def make_fit_triage_page(framing: MeshFraming | None = None) -> rrb.Horizontal:
    """Build the 3D and three-by-three fit-triage page."""
    triage_grid = rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Source RGB", origin=PINHOLE_ULTRAWIDE_RECT, contents=[f"/{RGB_ULTRAWIDE_RECT}"]),
            rrb.Spatial2DView(name="Fitted RGB", origin=PINHOLE_ULTRAWIDE_RECT, contents=[f"/{RGB_SPLAT_ULTRAWIDE_RECT}"]),
            rrb.Spatial2DView(name="RGB error [0, 0.5]", origin=PINHOLE_ULTRAWIDE_RECT, contents=[f"/{ERROR_RGB_ULTRAWIDE_RECT}"]),
        ),
        rrb.Horizontal(
            _depth_view("Source depth", PINHOLE_ULTRAWIDE_RECT, DEPTH_ULTRAWIDE_RECT),
            _depth_view("Fitted depth", PINHOLE_ULTRAWIDE_RECT, DEPTH_SPLAT_ULTRAWIDE),
            rrb.Spatial2DView(
                name="Depth error [0, 0.5 m]",
                origin=PINHOLE_ULTRAWIDE_RECT,
                contents=[f"/{ERROR_DEPTH_ULTRAWIDE_RECT}"],
            ),
        ),
        rrb.Horizontal(
            rrb.Spatial2DView(name="Source normal", origin=PINHOLE_ULTRAWIDE_RECT, contents=[f"/{NORMALS_ULTRAWIDE_RECT}"]),
            rrb.Spatial2DView(name="Fitted normal", origin=PINHOLE_ULTRAWIDE_RECT, contents=[f"/{NORMALS_SPLAT_ULTRAWIDE_RECT}"]),
            rrb.Spatial2DView(
                name="Normal error [0, 45 deg]",
                origin=PINHOLE_ULTRAWIDE_RECT,
                contents=[f"/{ERROR_NORMAL_ULTRAWIDE_RECT}"],
            ),
        ),
        row_shares=[1.0, 1.0, 1.0],
        name="Triage grid",
    )
    return rrb.Horizontal(
        rrb.Spatial3DView(
            name="Triage 3D splat",
            origin=f"/{WORLD}",
            contents=[f"/{SPLATS}/**", f"/{PROMPTDA_MESH}"],
            eye_controls=_inside_eye_controls(framing),
            overrides={f"/{PROMPTDA_MESH}": rr.Mesh3D.from_fields(albedo_factor=(0, 255, 0, 96))},
        ),
        triage_grid,
        column_shares=[1.0, 2.0],
        name="Fit triage",
    )


def make_blueprint(
    portrait: bool = False,
    framing: MeshFraming | None = None,
    include_promptda: bool = False,
    include_gauss_surf: bool = False,
) -> rrb.Blueprint:
    """Build per-source Capture, GT, PromptDA, Splat, and Fit-triage tabs.

    Capture and GT are always present. PromptDA is optional, while Splat and
    Fit triage are added together for the gauss-surf extension. Portrait mode
    gives the 3D views more width. Per-sequence framing keeps every 3D page on
    the same mesh-centered orbital eye controls. When present, Splat opens as
    the active source page.
    """
    pages: list[rrb.Container] = [make_capture_page(portrait, framing), make_gt_page(portrait, framing)]
    if include_promptda:
        pages.append(make_promptda_page(portrait, framing))

    splat_tab_index: int | None = None
    if include_gauss_surf:
        splat_tab_index = len(pages)
        pages.extend((make_splat_page(portrait), make_fit_triage_page(framing)))

    return rrb.Blueprint(
        rrb.Tabs(*pages, active_tab=splat_tab_index if splat_tab_index is not None else 0, name="ARKitScenes"),
        collapse_panels=True,
    )
