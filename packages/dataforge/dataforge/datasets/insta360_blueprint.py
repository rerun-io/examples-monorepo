"""The Insta360 viewer layout: one segment in three tabs, SLAM / colsfm / Comparison.

Every tab has the same skeleton, so a tab switch changes the content and not the places: the overview of the walk top
left, the follow view beside or under it, and the source's own pictures in the right column (the two lenses for
slam-rs, the six cube faces for colsfm). One colour per source in every tab: slam-rs orange, colsfm blue.

The layout names entities of layers a segment may not have yet. An override on a missing entity is inert, so the same
blueprint serves a base-only segment, a SLAM run and a colsfm run.

What the 3D views lean on, beside the runs themselves:

- ``schema.follow_path(source)``: the level follow frame of the run a tab shows, where its follow view is rooted: the
  causal VIO in the SLAM tab, the final slam-rs path in colsfm and Comparison (colsfm's prior; clip 007: 16 m from
  the VIO at 65 s). A view rooted at a rig rolls and pitches with the stick, and the viewer's ``tracking_entity`` does
  not move the eye (0.42). slam-rs writes it with the rig (``slam_rs.follow_frame``). No grid: the built-in one lies on
  the view origin's plane, which is not the floor.

The paths are logged a segment per pose (``schema.trail_path``) and shown as a trail over the last ``TRAIL_S`` of the
cursor; each run logs its whole path (``schema.trajectory_path``) and its loops once, at the last frame, so a cursor halfway through a clip shows no
path ahead of it. The raw slam-rs run and colsfm carry their walk gradient (green at the start, red at the end) and
start and end dots in their data; in the Comparison tab slam-rs is one flat grey (``COMPARISON_SLAM_COLOR``) so the two
runs stay apart. The colsfm cloud by first view grows along the walk: its 30 s blocks (``points_growing/blocks/NNNN``,
sized by the view's defaults) show as they are and its rows per rig frame (``points_growing/recent``) over the last block
(``simplecv.rerun_growing_points.GROWING_BLOCK_S``, the writer's block span). A range query from the start would change on every frame of playback, and the viewer processes all
of its rows again each time (clip 007: 3,096 rows, 2.32 M points, in each view).
The overviews turn slowly about the walk (``OVERVIEW_SPIN_RAD_S``).

With the ``cube_view`` layer (``ColsfmRig``) the colsfm tab shows the viewing faces (``pinhole/video_view``, every
frame). The colsfm views draw ``rig_view`` (a pose at every viewing frame referencing them) for ``rig`` (the registered
rig frames) when the colsfm layer holds it (``colsfm_view_rig``): a layer written without it keeps ``rig``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.encodings import EntityPathLike
from simplecv.rerun_growing_points import GROWING_BLOCK_S

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

Color: TypeAlias = tuple[int, int, int] | tuple[int, int, int, int]
"""An RGB or RGBA override colour."""

SLAM_COLOR: tuple[int, int, int] = (255, 140, 0)
"""slam-rs's frusta, its catalog-layer trail and its cloud tint (the raw run's path keeps its walk gradient)."""
COLSFM_COLOR: tuple[int, int, int] = (64, 170, 255)
"""colsfm's frusta and cloud tint (its path keeps its walk gradient)."""
COMPARISON_SLAM_COLOR: tuple[int, int, int, int] = (215, 215, 215, 153)
"""slam-rs in the Comparison tab: light grey at 60 % opacity, so colsfm keeps the walk gradient and the two runs stay apart."""
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
TRAIL_S: float = 8.0
"""How far back of the cursor a path logged a segment a pose is drawn, in seconds: about 4 to 5 m of walk."""
OVERVIEW_SPIN_RAD_S: float = 2.0 * np.pi / 75.0
"""How fast an overview turns about the walk's centre: one turn in 75 s (0.0838 rad/s). The follow views stay still."""
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
        spin_speed=OVERVIEW_SPIN_RAD_S,
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


def cloud(radius_ui: float, color: tuple[int, int, int] | None = None) -> rr.Points3D:
    """A cloud's display size in ui points and, with ``color``, one tint in place of its own colours."""
    return rr.Points3D.from_fields(radii=rr.Radius.ui_points(radius_ui), colors=color)


def stroke(color: Color | None, radius_ui: float) -> rr.LineStrips3D:
    """A path's width in ui points and, with ``color``, one colour in place of its own (the walk gradient)."""
    return rr.LineStrips3D.from_fields(colors=color, radii=rr.Radius.ui_points(radius_ui))


def trail(color: Color | None, radius_ui: float) -> list[blueprints.VisualizerLike]:
    """A path logged a segment a pose, drawn over the last ``TRAIL_S`` of the cursor (``stroke``)."""
    return [blueprints.cursor_window(TRAIL_S), stroke(color, radius_ui)]


def pictures() -> list[str]:
    """Content rules that drop every camera's subtree from a 3D view: lens and face videos, masks, keypoints."""
    return [f"- {schema.cam_path(RIG, cam)}/**" for cam in (0, 1, *FACE_CAMS)]


def frusta(root: str, cameras: Sequence[str], color: Color | None, size_m: float) -> Overrides:
    """One size for the frusta of ``cameras`` under ``root`` and, with ``color``, one colour in place of their own."""
    return {f"{root}/{camera}": rr.Pinhole.from_fields(color=color, image_plane_distance=size_m, line_width=FRUSTUM_LINE_WIDTH) for camera in cameras}


def lens_frusta(size_m: float) -> Overrides:
    """The two lenses of slam-rs's rig, in its colour."""
    return frusta(f"{VIO}/rig", [f"{lens}/pinhole" for lens in LENSES], SLAM_COLOR, size_m)


def window_frusta(size_m: float) -> Overrides:
    """Both lenses of every frame in the estimator's window, and of the frames its last marginalization removed, in the
    role colour slam-rs logs (keyframe, long-term keyframe, pose)."""
    slots: list[str] = [f"window/{slot:02d}" for slot in range(WINDOW_SLOTS)] + [f"marginalized/{slot:02d}" for slot in range(MARGINALIZED_SLOTS)]
    return frusta(VIO, [f"{slot}/{lens}/pinhole" for slot in slots for lens in LENSES], None, size_m)


def window_pictures() -> list[str]:
    """Content rules that drop the window keyframes' lens pictures and keep their frusta: each slot's pictures are a
    reference at its own time, so drawing them decodes up to two frames per slot and step, and playback halves."""
    return [f"- {VIO}/window/{slot:02d}/{lens}/pinhole/video" for slot in range(WINDOW_SLOTS) for lens in LENSES]


@dataclass(frozen=True, slots=True)
class ColsfmRig:
    """What a segment's layers let the colsfm views draw: the rig with its face frusta, the rig they leave out, and the
    face stream the face panes play."""

    drawn: str
    """``rig_view`` (every viewing frame) when the colsfm layer holds it, else ``rig`` (the registered rig frames)."""
    dropped: str
    """The other one."""
    face_video: Callable[[int, int], str]
    """``schema.video_view_path`` with the ``cube_view`` layer, else ``schema.video_path``."""


def colsfm_rig(view_faces: bool, view_rig: bool) -> ColsfmRig:
    """The colsfm rig for a colsfm layer with (``view_rig``) or without ``rig_view``, and the face stream for a segment
    with (``view_faces``) or without the ``cube_view`` layer."""
    drawn, dropped = ("rig_view", "rig") if view_rig else ("rig", "rig_view")
    return ColsfmRig(drawn=f"{COLSFM}/{drawn}", dropped=f"{COLSFM}/{dropped}", face_video=schema.video_view_path if view_faces else schema.video_path)


def colsfm_view_rig(entry: DatasetEntry, view_faces: bool) -> bool:
    """Whether the colsfm views draw ``rig_view``: when a registered colsfm layer holds it. A colsfm layer without it
    (``colsfm.apis.align_to_prior``, ``run --no-view-rig``, a run made before the viewing faces) keeps ``rig``. Before
    any colsfm layer is registered the answer is ``view_faces``: a default colsfm run on viewing faces writes it.

    Args:
        entry: The catalog dataset, its layers registered.
        view_faces: Whether a segment holds the ``cube_view`` layer.
    """

    def written(rig: str) -> bool:
        return any(entity == f"{COLSFM}/{rig}" or entity.startswith(f"{COLSFM}/{rig}/") for entity in entities)

    entities: set[str] = {str(found.entity_path) for found in entry.schema().component_columns()}
    return written("rig_view") or (view_faces and not written("rig"))


def run_pictures() -> list[str]:
    """Content rules that drop the pictures the two runs show in their frusta, and keep the frusta."""
    lens: list[str] = [f"{VIO}/rig/{camera}/pinhole/video" for camera in LENSES]
    faces: list[str] = [f"{COLSFM}/{rig}/{name}/pinhole/video" for rig in ("rig", "rig_view") for name, _ in cube.FACES]
    return [f"- {picture}/**" for picture in (*lens, *faces)]


def face_frusta(rig: str, size_m: float) -> Overrides:
    """The six cube faces of a colsfm rig, in its colour."""
    return frusta(rig, [f"{name}/pinhole" for name, _ in cube.FACES], COLSFM_COLOR, size_m)


def hidden(*entities: str) -> Overrides:
    """Entities a view holds but does not show until the reader turns them on."""
    return {entity: rrb.EntityBehavior(visible=False) for entity in entities}


def slam_overrides(*, close: bool) -> Overrides:
    """How a SLAM-tab 3D view draws slam-rs: the raw run's trail in its walk gradient over the darkened final trail, the
    map in its image colours, and at the last frame both whole paths and the final loops.

    The ``close`` (follow) view adds what happens at the rig: the two lenses with their pictures, the window's frames in
    their role colours (without their pictures, ``window_pictures``) and its landmarks. The frames a marginalization
    removed stay hidden in every view (a reader can turn them on).
    """
    at_the_rig: Overrides = {
        f"{VIO}/landmarks": cloud(3.0),
        **lens_frusta(0.18),
        **window_frusta(0.075),
    }
    return {
        schema.trail_path(schema.SLAM_RS_VIO_SOURCE): trail(None, 3.0),
        schema.trajectory_path(schema.SLAM_RS_VIO_SOURCE): stroke(None, 2.5),
        # The final path lies on the VIO path; the darkened stroke keeps the VIO path readable.
        schema.trail_path(schema.SLAM_RS_SOURCE): trail(shade(SLAM_COLOR, WHOLE_PATH_SHADE), 1.5),
        schema.trajectory_path(schema.SLAM_RS_SOURCE): stroke(shade(SLAM_COLOR, WHOLE_PATH_SHADE), 1.5),
        f"{SLAM}/map/points": cloud(1.5 if close else 2.0),
        **hidden(f"{SLAM}/gba", f"{SLAM}/pgo", f"{SLAM}/pose_graph", f"{SLAM}/keyframes", f"{VIO}/marginalized"),
        **(at_the_rig if close else hidden(f"{VIO}/landmarks", f"{VIO}/window", f"{VIO}/rig")),
    }


def growing_cloud_size(*, close: bool) -> float:
    """Point radius of colsfm's growing cloud in a colsfm-tab view, ui points."""
    return 1.5 if close else 2.0


def colsfm_cloud_defaults(*, close: bool) -> list[rr.Points3D]:
    """A colsfm-tab view's point size for the blocks of colsfm's growing cloud (``points_growing/blocks/NNNN``): an
    override takes an exact path and the number of blocks comes with the clip, so they log no radius and take this."""
    return [cloud(growing_cloud_size(close=close))]


def colsfm_overrides(*, close: bool, rig: ColsfmRig) -> Overrides:
    """How a colsfm-tab 3D view draws colsfm: the cloud growing from the start of the walk (its blocks as they are, the
    block the cursor is in from the recent rows), the trail in its walk gradient (the whole path at the end), and the
    cube rig in blue, each face's picture in its frustum."""
    return {
        schema.trail_path(schema.COLSFM_SOURCE): trail(None, 3.0),
        schema.trajectory_path(schema.COLSFM_SOURCE): stroke(None, 2.5),
        f"{COLSFM}/points_growing/recent": [blueprints.cursor_window(GROWING_BLOCK_S), cloud(growing_cloud_size(close=close))],
        **face_frusta(rig.drawn, 0.18 if close else 0.3),
    }


def comparison_overrides(*, close: bool, rig: ColsfmRig) -> Overrides:
    """How a Comparison-tab 3D view draws both sources: the final slam-rs path in flat grey and colsfm's in its walk
    gradient, colsfm's rig, two clouds in their sources' colours.

    The paths lie centimetres apart. In the overview slam-rs is the wide stroke and colsfm the narrow one on it, so
    both show where they agree and two lines where they part; close up they have one width.
    """
    wide, narrow = (3.5, 3.5) if close else (6.0, 3.0)
    size_m: float = 0.125 if close else 0.25
    return {
        schema.trail_path(schema.SLAM_RS_SOURCE): trail(COMPARISON_SLAM_COLOR, wide),
        schema.trajectory_path(schema.SLAM_RS_SOURCE): stroke(COMPARISON_SLAM_COLOR, wide),
        schema.trail_path(schema.COLSFM_SOURCE): trail(None, narrow),
        schema.trajectory_path(schema.COLSFM_SOURCE): stroke(None, narrow),
        # The final map only: while the run goes on slam-rs draws its map as blocks under ``map/points/blocks``, which
        # no override can reach (exact paths only), so they keep their image colours until the final draw replaces them.
        f"{SLAM}/map/points": cloud(1.5, shade(SLAM_COLOR, CLOUD_SHADE)),
        f"{COLSFM}/points": cloud(1.5, shade(COLSFM_COLOR, CLOUD_SHADE)),
        **hidden(f"{SLAM}/gba", f"{SLAM}/pgo", f"{SLAM}/pose_graph", f"{SLAM}/keyframes", f"{SLAM}/loops"),
        **face_frusta(rig.drawn, size_m),
    }


def overview(
    name: str, contents: list[str], overrides: Overrides, eye: rrb.EyeControls3D | None, source: str, defaults: list[rr.Points3D] | None = None
) -> rrb.Spatial3DView:
    """A tab's world view of the whole walk of ``source``, with that run's position as a dot.

    The rig is centimetres on a walk of tens of metres, so a point at the follow frame's origin with a ui-point radius
    marks it at any zoom. No grid. ``defaults`` style what no override reaches (``colsfm_cloud_defaults``).
    """
    return rrb.Spatial3DView(
        name=name,
        origin="/world",
        contents=contents,
        overrides={
            schema.follow_path(source): rr.Points3D([[0.0, 0.0, 0.0]], radii=rr.Radius.ui_points(6.0), colors=[RIG_MARK_COLOR]),
            **overrides,
        },
        eye_controls=eye,
        line_grid=False,
        defaults=defaults,
    )


def follow(name: str, contents: list[str], overrides: Overrides, source: str, defaults: list[rr.Points3D] | None = None) -> rrb.Spatial3DView:
    """A tab's chase view, rooted at the level follow frame of ``source``; ``defaults`` as in ``overview``."""
    return rrb.Spatial3DView(
        name=name,
        origin=schema.follow_path(source),
        contents=contents,
        overrides=overrides,
        defaults=defaults,
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
        contents=[schema.video_path(RIG, cam), f"{pinhole}/keypoints", schema.people_mask_path(RIG, cam)],
        overrides={f"{pinhole}/keypoints": rr.Points2D.from_fields(colors=KEYPOINT_COLOR, radii=rr.Radius.ui_points(KEYPOINT_RADIUS_UI))},
    )


def face_pane(cam: int, name: str, video: str) -> rrb.Spatial2DView:
    """One cube face, colsfm's input: the face ``video`` (the viewing or the analysis stream) and the people mask that
    keeps keypoints off the operator.

    The masks are on the analysis stamps only: over the viewing stream a mask shows on its own stamp alone (an exact
    cursor range), as latest-at would hold it over the viewing frames up to the next.
    """
    title: str = name.removeprefix("cube6_")
    if title[-1].isdigit():
        title = f"{title[:-1]} {title[-1]}"
    mask: str = schema.people_mask_path(RIG, cam)
    overrides: Overrides = {mask: blueprints.cursor_window(0.0)} if video == schema.video_view_path(RIG, cam) else {}
    return blueprints.camera_view(f"Face {title}", RIG, cam, contents=[video, mask], overrides=overrides or None)


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
    scene: list[str] = ["/world/**", *pictures(), *window_pictures(), f"- {COLSFM}/**"]
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


def colsfm_tab(eye: rrb.EyeControls3D | None, rig: ColsfmRig) -> rrb.Horizontal:
    """colsfm: the cloud that grows with the walk, the follow view of the cube rig, and the six faces it reads."""
    # Followed on the final slam-rs path, the prior colsfm refined: its rig stays within a metre or so of it.
    scene: list[str] = [
        f"{COLSFM}/**",
        f"- {COLSFM}/points/**",
        f"- {POSITION_DIFFERENCE}/**",
        f"- {rig.dropped}/**",
        schema.follow_path(schema.SLAM_RS_SOURCE),
    ]
    return rrb.Horizontal(
        rrb.Vertical(
            overview(
                "Overview: colsfm cloud grows with the walk",
                scene,
                colsfm_overrides(close=False, rig=rig),
                eye,
                schema.SLAM_RS_SOURCE,
                colsfm_cloud_defaults(close=False),
            ),
            follow(
                "Follow: level chase view, cube rig",
                scene,
                colsfm_overrides(close=True, rig=rig),
                schema.SLAM_RS_SOURCE,
                colsfm_cloud_defaults(close=True),
            ),
            row_shares=[5, 4],
        ),
        rrb.Grid(*(face_pane(cam, name, rig.face_video(RIG, cam)) for cam, (name, _) in zip(FACE_CAMS, cube.FACES, strict=True)), grid_columns=2, name="Cube faces"),
        column_shares=[8, 5],
        name="colsfm",
    )


def comparison_tab(eye: rrb.EyeControls3D | None, rig: ColsfmRig) -> rrb.Vertical:
    """Both sources in one world, close up at the rig, and how far apart their positions are over the walk."""
    scene: list[str] = [
        "/world/**",
        *pictures(),
        *run_pictures(),
        f"- {COLSFM}/points_growing/**",
        f"- {POSITION_DIFFERENCE}/**",
        f"- {rig.dropped}/**",
        # The final path is what colsfm started from; the causal VIO stands metres off it early in a long walk.
        f"- {VIO}/**",
    ]
    return rrb.Vertical(
        rrb.Horizontal(
            overview(
                "Overview: slam-rs (grey path; live map in image colours, orange once final) and colsfm (blue)",
                scene,
                comparison_overrides(close=False, rig=rig),
                eye,
                schema.SLAM_RS_SOURCE,
            ),
            follow("Follow: colsfm's rig on the final slam-rs path", scene, comparison_overrides(close=True, rig=rig), schema.SLAM_RS_SOURCE),
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


def segment_layout(
    *, panorama: bool, view_faces: bool = False, view_rig: bool | None = None, walk: Float64[ndarray, "n 3"] | None = None
) -> rrb.Blueprint:
    """The Insta360 layout: SLAM, colsfm and Comparison tabs, and with ``panorama`` a tab for the panorama.

    Args:
        panorama: Whether a segment holds the panorama layer.
        view_faces: Whether a segment holds the ``cube_view`` layer: the colsfm tab then plays the viewing faces.
        view_rig: Whether the colsfm layer holds ``rig_view`` (``colsfm_view_rig``): the colsfm views then draw it for
            ``rig``. None, before a colsfm layer exists: as ``view_faces``.
        walk: Rig positions of the segment's walk (world frame, metres), when the caller knows them: each overview
            then frames the walk for its own pane (``walk_eye``). Without them the viewer frames the view's bounds.
    """
    eyes: list[rrb.EyeControls3D | None] = [None if walk is None else walk_eye(walk, aspect=aspect) for aspect in OVERVIEW_ASPECTS]
    rig: ColsfmRig = colsfm_rig(view_faces, view_faces if view_rig is None else view_rig)
    return rrb.Blueprint(
        rrb.Tabs(
            slam_tab(eyes[0]),
            colsfm_tab(eyes[1], rig),
            comparison_tab(eyes[2], rig),
            *([rrb.Spatial2DView(name="Panorama", origin=PANO_ENTITY)] if panorama else []),
            active_tab=0,
        ),
        # An explicit TimePanel is not collapsed by collapse_panels, so it states its own state.
        rrb.TimePanel(timeline=schema.TIMELINE, state="collapsed"),
        collapse_panels=True,
    )
