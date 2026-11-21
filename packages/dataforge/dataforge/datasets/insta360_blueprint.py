"""The Insta360 viewer layout: one segment in three tabs, SLAM / colsfm / Comparison.

Every tab has the same skeleton, so a tab switch changes the content and not the places: the overview of the walk top
left, the follow view beside or under it, and the source's own pictures in the right column (the two lenses for
slam-rs, the six cube faces for colsfm). One colour per source in every tab: slam-rs orange, colsfm blue.

The layout names entities of layers a segment may not have yet. An override on a missing entity is inert, so the same
blueprint serves a base-only segment, a SLAM run and a colsfm run.

What the 3D views lean on, beside the runs themselves:

- ``schema.follow_path(source)``: the level follow frame of the run a tab shows, where its follow view is rooted: the
  causal VIO in the SLAM tab, the final slam-rs path in colsfm and Comparison (colsfm's prior; clip 007: 16 m from
  the VIO at 65 s). A view rooted at a rig rolls and
  pitches with the stick, and the viewer's ``tracking_entity`` does not move the eye (0.42);
- ``schema.floor_path(source)``: the floor under that run's walk, each edge at the time the walk passed it. The built-in grid
  lies on the view origin's plane: it stands still under a follow view, and in the world it sits at the rig's first
  height, through the cloud. An overview shows every floor of the walk; a follow view the floor of the
  ``FOLLOW_FLOOR_WINDOW_S`` around the cursor, so on a walk over several storeys it draws the one under the rig.

slam-rs writes both with the rig (``slam_rs.follow_frame``).

What accumulates is logged once at its own time and shown through a cursor window: the VIO path (a segment per
frameset) over the last ``OVERVIEW_PATH_WINDOW_S`` in the overview and ``FOLLOW_PATH_WINDOW_S`` in the follow view, and
the colsfm cloud by first view (``points_growing``) since the start, so it grows along the walk.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.encodings import EntityPathLike

from dataforge import blueprints, schema
from dataforge.datasets import insta360_cube as cube

RIG: int = 0
"""One rig per clip."""
PANO_ENTITY: str = f"{schema.rig_path(RIG)}/pano"
"""Where the panorama layer puts its video."""
FIRST_FACE_CAM: int = 2
"""Camera index of the first cube face; face ``k`` of ``insta360_cube.FACES`` is ``2 + k``."""
FACE_CAMS: range = range(FIRST_FACE_CAM, FIRST_FACE_CAM + len(cube.FACES))
"""``cam_02 .. cam_07``: the cube faces."""

Overrides: TypeAlias = dict[EntityPathLike, blueprints.ViewOverride]
"""A view's per-entity overrides."""

SLAM: str = schema.run_path(schema.SLAM_RS_SOURCE)
"""slam-rs's final run and its map (``map/points``, ``keyframes``, ``loops``)."""
VIO: str = schema.run_path(schema.SLAM_RS_VIO_SOURCE)
"""slam-rs's raw causal VIO run: the replay draws its path a segment a frameset, with the window and its landmarks."""
COLSFM: str = schema.run_path(schema.COLSFM_SOURCE)
"""The colsfm layer: refined rig, frusta, the static sparse cloud (``points``) and the same cloud by first view (``points_growing``)."""
POSITION_DIFFERENCE: str = f"{COLSFM}/position_difference"
"""How far each colsfm rig frame lies from the prior pose it started from, metres (``colsfm.rerun_layer``)."""

SLAM_COLOR: tuple[int, int, int] = (255, 140, 0)
"""slam-rs in every tab: paths and frusta."""
COLSFM_COLOR: tuple[int, int, int] = (64, 170, 255)
"""colsfm in every tab."""
WINDOW_COLOR: tuple[int, int, int] = (255, 205, 130)
"""The estimator's sliding-window frames: a light tint of the slam-rs colour (the layer's own blue is colsfm's colour here)."""
DIFFERENCE_COLOR: tuple[int, int, int] = (230, 230, 235)
"""The position difference: it belongs to both sources, so it takes neither colour."""
RIG_MARK_COLOR: tuple[int, int, int] = (255, 255, 255)
"""The rig's position in an overview."""
KEYPOINT_COLOR: tuple[int, int, int] = (60, 220, 120)
"""Tracked keypoints on the lenses: a green no wall or floor has."""
AXIS_COLORS: list[tuple[int, int, int]] = [(240, 90, 90), (110, 210, 110), (90, 150, 255)]
"""x, y, z of an IMU plot: the red, green and blue the viewer draws axes in."""
WHOLE_PATH_SHADE: float = 0.38
"""The whole walk under its travelled part: the source colour, darkened by this factor."""
CLOUD_SHADE: float = 0.5
"""A cloud tinted with its source colour in the Comparison tab: dark, so both paths stay readable on it."""

KEYPOINT_RADIUS_UI: float = 2.5
"""Keypoint radius in ui points: a 2 px image radius shrinks below one screen pixel on a 1920 px lens in a small pane."""
FRUSTUM_LINE_WIDTH: float = -1.5
"""Stroke of a frustum: 1.5 ui points (Rerun reads a negative width as ui points)."""
WINDOW_SLOTS: int = 16
"""Window frames the replay may log (``window/NN``); an override on a slot that is not there is inert."""
MARGINALIZED_SLOTS: int = 4
"""The same for the frames a marginalization removed (``marginalized/NN``)."""
LENSES: tuple[str, str] = ("cam_00", "cam_01")
"""The two lens cameras, as slam-rs names them under its rig and under each window frame."""
FOLLOW_PATH_WINDOW_S: float = 30.0
"""How far back of the cursor the follow view draws a path logged a segment a pose, in seconds."""
OVERVIEW_PATH_WINDOW_S: float = 120.0
"""The same for the overview. Bounded on purpose: on clip 007 (1,069 s, 32 k segments) a window from the start played at
7 fps in the native viewer, 120 s at 16.5 fps; the darkened whole path still shows where the walk went."""
OVERVIEW_FLOOR_COLOR: tuple[int, int, int] = (30, 31, 36)
"""The floor in an overview: a step over the background, so the walk's floor is a hint under the clouds and paths."""
FOLLOW_FLOOR_WINDOW_S: float = 20.0
"""How far either side of the cursor a follow view shows the floor, in seconds: about the grid's own reach at a walk."""
IMU_WINDOW_S: float = 5.0
"""Half width of the IMU strip's time axis: a window that rides the cursor reads as a waveform, a whole clip as noise."""
FOV_Y_RAD: float = float(np.radians(55.0))
"""Vertical field of view of the viewer's 3D eye."""
EYE_ELEVATION_DEG: float = 48.0
"""How far above the ground plane an overview eye looks from."""
WALK_PAD_M: float = 2.0
"""Room an overview leaves around the walk for what stands beside it (walls a few metres off)."""
OVERVIEW_ASPECTS: tuple[float, float, float] = (2.7, 2.0, 1.4)
"""Width over height of the overview pane in the SLAM, colsfm and Comparison tabs, on a 16:10 window."""

FOLLOW_EYE: rrb.EyeControls3D = rrb.EyeControls3D(
    kind=rrb.Eye3DKind.Orbital,
    position=(-2.5, -1.1, 1.1),
    look_target=(1.0, 0.0, -0.4),
    eye_up=(0.0, 0.0, 1.0),
    spin_speed=0.0,
)
"""Chase eye in the follow frame (x along the walk, z up): behind and above the rig, a little to one side."""


def walk_eye(walk: Float64[ndarray, "n 3"], *, aspect: float) -> rrb.EyeControls3D:
    """An overview eye that frames a walk: its long axis across the pane, seen from above one long side.

    The viewer's own framing takes the bounds of everything in the view, and a few landmarks a hundred metres off
    push the eye far from a walk of twenty.

    Args:
        walk: Rig positions of the walk in the world (z up), metres.
        aspect: Width over height of the pane the eye is for.
    """
    centre: Float64[ndarray, " 3"] = 0.5 * (walk.min(axis=0) + walk.max(axis=0))
    ground: Float64[ndarray, "n 2"] = walk[:, :2] - centre[:2]
    axes: Float64[ndarray, "2 2"] = np.linalg.svd(ground, full_matrices=False)[2]
    half_long: float = float(np.abs(ground @ axes[0]).max()) + WALK_PAD_M
    half_short: float = float(np.abs(ground @ axes[1]).max()) + WALK_PAD_M
    elevation: float = float(np.radians(EYE_ELEVATION_DEG))
    tan_half_fov: float = float(np.tan(FOV_Y_RAD / 2))
    # Far enough for the long axis to fit the width and for the foreshortened short axis to fit the height; the near
    # edge of the walk lies half_short * cos(elevation) closer than its middle.
    near_edge: float = half_short * float(np.cos(elevation))
    distance: float = max(half_long / (aspect * tan_half_fov), half_short * float(np.sin(elevation)) / tan_half_fov) + near_edge
    back: Float64[ndarray, " 3"] = np.array([*(-axes[1] * np.cos(elevation)), np.sin(elevation)])
    return rrb.EyeControls3D(
        kind=rrb.Eye3DKind.Orbital,
        position=(centre + distance * back).tolist(),
        look_target=centre.tolist(),
        eye_up=(0.0, 0.0, 1.0),
        spin_speed=0.0,
    )


def segment_walk(entry: DatasetEntry) -> Float64[ndarray, "n 3"] | None:
    """The walk of a dataset's one walked segment, read off the final slam-rs run's follow frame, or None.

    A catalog dataset has one segment blueprint. Its overviews can frame a walk only when one segment has one: a
    dataset of several walked clips gets the viewer's own framing.

    Args:
        entry: The catalog dataset, its layers registered.
    """
    follow_frame: str = schema.follow_path(schema.SLAM_RS_SOURCE)
    column: str = f"{follow_frame}:Transform3D:translation"
    if column not in {found.name for found in entry.schema().component_columns()}:
        return None
    table: pa.Table = entry.filter_contents([follow_frame]).reader(index=schema.TIMELINE).select("rerun_segment_id", column).to_arrow_table()
    if len(set(table["rerun_segment_id"].to_pylist())) != 1:
        return None
    walk: Float64[ndarray, "n 3"] = np.asarray([row[0] for row in table[column].to_pylist() if row], dtype=np.float64).reshape(-1, 3)
    return walk if len(walk) >= 2 else None


def shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """A colour darkened towards the viewer's background (lines and points do not blend, so alpha would not fade them)."""
    return (round(color[0] * factor), round(color[1] * factor), round(color[2] * factor))


MAP_BLOCK_UI_OVERVIEW: float = 2.0
"""Size, ui points, of the clouds an overview draws without an override of their own: slam-rs's map blocks."""
MAP_BLOCK_UI_FOLLOW: float = 1.5
"""The same in a follow view."""


def cloud(radius_ui: float, color: tuple[int, int, int] | None = None) -> rr.Points3D:
    """A cloud's display size in ui points and, with ``color``, one tint in place of its own colours."""
    return rr.Points3D.from_fields(radii=rr.Radius.ui_points(radius_ui), colors=color)


def stroke(color: tuple[int, int, int], radius_ui: float) -> rr.LineStrips3D:
    """A path's colour and width in ui points."""
    return rr.LineStrips3D.from_fields(colors=color, radii=rr.Radius.ui_points(radius_ui))


def pictures() -> list[str]:
    """Content rules that drop every camera's subtree from a 3D view: lens and face videos, masks, keypoints."""
    return [f"- {schema.cam_path(RIG, cam)}/**" for cam in (0, 1, *FACE_CAMS)]


def frusta(root: str, cameras: Sequence[str], color: tuple[int, int, int], size_m: float) -> Overrides:
    """One colour and size for the frusta of ``cameras`` under ``root``."""
    return {f"{root}/{camera}": rr.Pinhole.from_fields(color=color, image_plane_distance=size_m, line_width=FRUSTUM_LINE_WIDTH) for camera in cameras}


def lens_frusta(size_m: float) -> Overrides:
    """The two lenses of slam-rs's rig, in its colour."""
    return frusta(f"{VIO}/rig", [f"{lens}/pinhole" for lens in LENSES], SLAM_COLOR, size_m)


def window_frusta(size_m: float) -> Overrides:
    """Both lenses of every frame in the estimator's window, and of the frames its last marginalization removed."""
    slots: list[str] = [f"window/{slot:02d}" for slot in range(WINDOW_SLOTS)] + [f"marginalized/{slot:02d}" for slot in range(MARGINALIZED_SLOTS)]
    return frusta(VIO, [f"{slot}/{lens}/pinhole" for slot in slots for lens in LENSES], WINDOW_COLOR, size_m)


def run_pictures() -> list[str]:
    """Content rules that drop the pictures the two runs show in their frusta, and keep the frusta."""
    lens: list[str] = [f"{VIO}/rig/{camera}/pinhole/video" for camera in LENSES]
    faces: list[str] = [f"{COLSFM}/rig/{name}/pinhole/video" for name, _ in cube.FACES]
    return [f"- {picture}/**" for picture in (*lens, *faces)]


def face_frusta(size_m: float) -> Overrides:
    """The six cube faces of colsfm's rig, in its colour."""
    return frusta(f"{COLSFM}/rig", [f"{name}/pinhole" for name, _ in cube.FACES], COLSFM_COLOR, size_m)


def hidden(*entities: str) -> Overrides:
    """Entities a view holds but does not show until the reader turns them on."""
    return {entity: rrb.EntityBehavior(visible=False) for entity in entities}


def slam_overrides(*, close: bool) -> Overrides:
    """How a SLAM-tab 3D view draws slam-rs: the travelled path bright over the darkened whole walk, the map in its image colours.

    The ``close`` (follow) view adds what happens at the rig: the two lenses with their pictures, the window's frames
    (a keyframe's frusta hold the lens pictures of its own time) and its landmarks.
    """
    window_s: float = FOLLOW_PATH_WINDOW_S if close else OVERVIEW_PATH_WINDOW_S
    at_the_rig: Overrides = {
        f"{VIO}/landmarks": cloud(3.0),
        **lens_frusta(0.35),
        **window_frusta(0.15),
    }
    return {
        f"{VIO}/path": stroke(shade(SLAM_COLOR, WHOLE_PATH_SHADE), 1.5),
        f"{VIO}/trajectory": [blueprints.cursor_window(window_s), stroke(SLAM_COLOR, 3.0)],
        # The catalog layer's per-pose trail, where a segment has that layer and not the replay.
        schema.trail_path(schema.SLAM_RS_SOURCE): [blueprints.cursor_window(window_s), stroke(SLAM_COLOR, 3.0)],
        # The final path lands on the VIO path at the last frameset; the darkened stroke keeps the VIO path readable.
        f"{SLAM}/trajectory": stroke(shade(SLAM_COLOR, WHOLE_PATH_SHADE), 1.5),
        f"{SLAM}/map/points": cloud(1.5 if close else 2.0),
        **hidden(schema.trail_path(schema.SLAM_RS_VIO_SOURCE), f"{SLAM}/gba", f"{SLAM}/pgo", f"{SLAM}/pose_graph", f"{SLAM}/keyframes"),
        **(at_the_rig if close else hidden(f"{VIO}/landmarks", f"{VIO}/window", f"{VIO}/marginalized", f"{VIO}/rig")),
    }


def colsfm_overrides(*, close: bool) -> Overrides:
    """How a colsfm-tab 3D view draws colsfm: the cloud growing from the start of the walk, the path and the cube rig in
    blue, each face's picture in its frustum."""
    return {
        f"{COLSFM}/trajectory": stroke(COLSFM_COLOR, 2.0),
        f"{COLSFM}/points_growing": [blueprints.cursor_window(None), cloud(1.5 if close else 2.0)],
        **face_frusta(0.35 if close else 0.6),
    }


def comparison_overrides(*, close: bool) -> Overrides:
    """How a Comparison-tab 3D view draws both sources: the final slam-rs path and colsfm's, colsfm's rig, two clouds, each in its source's colour.

    The paths lie centimetres apart. In the overview slam-rs is the wide stroke and colsfm the narrow one on it, so
    both show where they agree and two lines where they part; close up they have one width.
    """
    wide, narrow = (3.5, 3.5) if close else (6.0, 2.0)
    size_m: float = 0.25 if close else 0.5
    return {
        f"{SLAM}/path": stroke(SLAM_COLOR, wide),
        f"{SLAM}/trajectory": stroke(SLAM_COLOR, wide),
        f"{COLSFM}/trajectory": stroke(COLSFM_COLOR, narrow),
        f"{SLAM}/map/points": cloud(1.5, shade(SLAM_COLOR, CLOUD_SHADE)),
        f"{COLSFM}/points": cloud(1.5, shade(COLSFM_COLOR, CLOUD_SHADE)),
        **hidden(f"{SLAM}/gba", f"{SLAM}/pgo", f"{SLAM}/pose_graph", f"{SLAM}/keyframes", f"{SLAM}/loops"),
        **face_frusta(size_m),
    }


def floor_window(half_s: float | None) -> rrb.VisibleTimeRanges:
    """The floor a view shows: the edges the walk passed within ``half_s`` of the cursor, or with None all of them."""
    if half_s is None:
        return rrb.VisibleTimeRanges(rrb.VisibleTimeRange(schema.TIMELINE, start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()))
    return rrb.VisibleTimeRanges(
        rrb.VisibleTimeRange(schema.TIMELINE, start=rrb.TimeRangeBoundary.cursor_relative(seconds=-half_s), end=rrb.TimeRangeBoundary.cursor_relative(seconds=half_s))
    )


def overview(name: str, contents: list[str], overrides: Overrides, eye: rrb.EyeControls3D | None, source: str) -> rrb.Spatial3DView:
    """A tab's world view of the whole walk of ``source``, with that run's position as a dot.

    The rig is centimetres on a walk of tens of metres, so a point at the follow frame's origin with a ui-point radius
    marks it at any zoom. No built-in grid: the floor is ``schema.floor_path(source)``, all of it.
    """
    return rrb.Spatial3DView(
        name=name,
        origin="/world",
        contents=contents,
        overrides={
            schema.follow_path(source): rr.Points3D([[0.0, 0.0, 0.0]], radii=rr.Radius.ui_points(6.0), colors=[RIG_MARK_COLOR]),
            # From the overview distance the floor's edges merge into a haze over the walk (clip 007: 41,687 edges, six
            # storeys), so it takes one faint colour there; the follow views keep the layer's own fade.
            schema.floor_path(source): [floor_window(None), rr.LineStrips3D.from_fields(colors=OVERVIEW_FLOOR_COLOR)],
            **overrides,
        },
        # slam-rs draws its map in blocks under ``map/points/blocks`` while the run goes on (no radii logged); an override
        # reaches only one exact entity, so the view sizes them. The final map on ``map/points`` keeps its override.
        defaults=[rr.Points3D.from_fields(radii=rr.Radius.ui_points(MAP_BLOCK_UI_OVERVIEW))],
        eye_controls=eye,
        line_grid=False,
    )


def follow(name: str, contents: list[str], overrides: Overrides, source: str) -> rrb.Spatial3DView:
    """A tab's chase view, rooted at the level follow frame of ``source``, over that run's floor around the cursor."""
    return rrb.Spatial3DView(
        name=name,
        origin=schema.follow_path(source),
        contents=contents,
        overrides={schema.floor_path(source): floor_window(FOLLOW_FLOOR_WINDOW_S), **overrides},
        defaults=[rr.Points3D.from_fields(radii=rr.Radius.ui_points(MAP_BLOCK_UI_FOLLOW))],
        eye_controls=FOLLOW_EYE,
        line_grid=False,
    )


def lens_pane(cam: int) -> rrb.Spatial2DView:
    """One lens: its video, the keypoints slam-rs tracked on it, and the people mask the trackers ignored.

    The replay's own frames (``pinhole/image``, gray) are left out: the base video is the same picture in colour.
    """
    pinhole: str = schema.pinhole_path(RIG, cam)
    return blueprints.camera_view(
        f"Lens {cam}: tracked keypoints, people mask",
        RIG,
        cam,
        contents=[schema.video_path(RIG, cam), f"{pinhole}/keypoints", f"{pinhole}/people_mask"],
        overrides={f"{pinhole}/keypoints": rr.Points2D.from_fields(colors=KEYPOINT_COLOR, radii=rr.Radius.ui_points(KEYPOINT_RADIUS_UI))},
    )


def face_pane(cam: int, name: str) -> rrb.Spatial2DView:
    """One cube face, colsfm's input: its video and the people mask that keeps keypoints off the operator."""
    title: str = name.removeprefix("cube6_")
    if title[-1].isdigit():
        title = f"{title[:-1]} {title[-1]}"
    return blueprints.camera_view(f"Face {title}", RIG, cam, contents=[schema.video_path(RIG, cam), f"{schema.pinhole_path(RIG, cam)}/people_mask"])


def imu_strip() -> rrb.Horizontal:
    """Gyroscope and accelerometer over a few seconds around the cursor, x, y, z in red, green, blue."""
    axis: rrb.TimeAxis = rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.cursor_relative(seconds=-IMU_WINDOW_S), end=rrb.TimeRangeBoundary.cursor_relative(seconds=IMU_WINDOW_S))
    )
    return rrb.Horizontal(
        *(
            rrb.TimeSeriesView(
                name=name,
                origin=schema.imu_path(RIG, 0),
                contents=channel,
                axis_x=axis,
                plot_legend=rrb.PlotLegend(rrb.Corner2D.LeftTop, visible=True),
                overrides={channel: rr.SeriesLines(names=["x", "y", "z"], colors=AXIS_COLORS, widths=[1.5, 1.5, 1.5])},
            )
            for name, channel in (("Gyroscope [rad/s]", schema.gyro_path(RIG, 0)), ("Accelerometer [m/s²]", schema.accel_path(RIG, 0)))
        )
    )


def slam_tab(eye: rrb.EyeControls3D | None) -> rrb.Vertical:
    """slam-rs: the map and the VIO path, the follow view of the window, both lenses, over the IMU strip."""
    # The causal VIO is what this tab shows: its window, landmarks and lens frusta live in that run's frame.
    scene: list[str] = ["/world/**", *pictures(), f"- {COLSFM}/**", f"- {schema.floor_path(schema.SLAM_RS_SOURCE)}/**"]
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.Vertical(
                overview("Overview: SLAM map and VIO path", scene, slam_overrides(close=False), eye, schema.SLAM_RS_VIO_SOURCE),
                follow("Follow: level chase view, VIO window and landmarks", scene, slam_overrides(close=True), schema.SLAM_RS_VIO_SOURCE),
                row_shares=[5, 4],
            ),
            rrb.Vertical(lens_pane(0), lens_pane(1)),
            column_shares=[3, 1],
        ),
        imu_strip(),
        row_shares=[7, 1],
        name="SLAM",
    )


def colsfm_tab(eye: rrb.EyeControls3D | None) -> rrb.Horizontal:
    """colsfm: the cloud that grows with the walk, the follow view of the cube rig, and the six faces it reads."""
    # Followed on the final slam-rs path, the prior colsfm refined: its rig stays within a metre or so of it.
    scene: list[str] = [
        f"{COLSFM}/**",
        f"- {COLSFM}/points/**",
        f"- {POSITION_DIFFERENCE}/**",
        schema.follow_path(schema.SLAM_RS_SOURCE),
        schema.floor_path(schema.SLAM_RS_SOURCE),
    ]
    return rrb.Horizontal(
        rrb.Vertical(
            overview("Overview: colsfm cloud grows with the walk", scene, colsfm_overrides(close=False), eye, schema.SLAM_RS_SOURCE),
            follow("Follow: level chase view, cube rig", scene, colsfm_overrides(close=True), schema.SLAM_RS_SOURCE),
            row_shares=[5, 4],
        ),
        rrb.Grid(*(face_pane(cam, name) for cam, (name, _) in zip(FACE_CAMS, cube.FACES, strict=True)), grid_columns=2, name="Cube faces"),
        column_shares=[8, 5],
        name="colsfm",
    )


def comparison_tab(eye: rrb.EyeControls3D | None) -> rrb.Vertical:
    """Both sources in one world, close up at the rig, and how far apart their positions are over the walk."""
    scene: list[str] = [
        "/world/**",
        *pictures(),
        *run_pictures(),
        f"- {COLSFM}/points_growing/**",
        f"- {POSITION_DIFFERENCE}/**",
        # The final path is what colsfm started from; the causal VIO stands metres off it early in a long walk.
        f"- {VIO}/**",
        f"- {schema.trail_path(schema.SLAM_RS_SOURCE)}/**",
    ]
    return rrb.Vertical(
        rrb.Horizontal(
            overview("Overview: slam-rs final (orange) and colsfm (blue), paths and clouds", scene, comparison_overrides(close=False), eye, schema.SLAM_RS_SOURCE),
            follow("Follow: colsfm's rig on the final slam-rs path", scene, comparison_overrides(close=True), schema.SLAM_RS_SOURCE),
            column_shares=[3, 2],
        ),
        rrb.TimeSeriesView(
            name="Position difference, colsfm to its slam-rs prior [m]",
            origin=POSITION_DIFFERENCE,
            plot_legend=rrb.PlotLegend(rrb.Corner2D.LeftTop, visible=True),
            overrides={POSITION_DIFFERENCE: rr.SeriesLines(colors=[DIFFERENCE_COLOR], widths=[2.0])},
        ),
        row_shares=[3, 1],
        name="Comparison",
    )


def segment_layout(*, panorama: bool, walk: Float64[ndarray, "n 3"] | None = None) -> rrb.Blueprint:
    """The Insta360 layout: SLAM, colsfm and Comparison tabs, and with ``panorama`` a tab for the panorama.

    Args:
        panorama: Whether a segment holds the panorama layer.
        walk: Rig positions of the segment's walk (world frame, metres), when the caller knows them: each overview
            then frames the walk for its own pane (``walk_eye``). Without them the viewer frames the view's bounds.
    """
    eyes: list[rrb.EyeControls3D | None] = [None if walk is None else walk_eye(walk, aspect=aspect) for aspect in OVERVIEW_ASPECTS]
    return rrb.Blueprint(
        rrb.Tabs(
            slam_tab(eyes[0]),
            colsfm_tab(eyes[1]),
            comparison_tab(eyes[2]),
            *([rrb.Spatial2DView(name="Panorama", origin=PANO_ENTITY)] if panorama else []),
            active_tab=0,
        ),
        # An explicit TimePanel is not collapsed by collapse_panels, so it states its own state.
        rrb.TimePanel(timeline=schema.TIMELINE, state="collapsed"),
        collapse_panels=True,
    )
