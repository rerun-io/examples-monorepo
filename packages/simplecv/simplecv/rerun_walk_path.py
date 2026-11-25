"""A run's walk path under ``/world/runs/<run>``, in dataforge's names; slam-rs, colsfm and dataforge all write it here.

- ``<run>/trail``: one two-point segment per pose on ``video_time``, row ``k`` at ``t_ns[k]`` from pose ``k - 1`` to
  pose ``k`` (the first a zero-length segment), so a view's cursor window over the trail holds exactly as many rows as
  the pose track it trails (dataforge's ``schema.trail_path``).
- ``<run>/trajectory``: the whole path as one row at a given stamp, usually the last pose, so no whole path stands over a
  cursor halfway through a clip (dataforge's ``schema.trajectory_path``).
- ``<run>/endpoints/start`` (a dot and a ring, static) and ``<run>/endpoints/end`` (a dot, from the given stamp on).

The walk gradient (``walk_colors``) runs green at the first pose through yellow to red at the last. Every stroke width is
in ui points: a walk can be hundreds of metres long, so a metric width is invisible in an overview of all of it.

Only numpy and rerun here: dataforge depends on simplecv, not back, so its leaf names are restated and a test in each
consumer pins them.
"""

import colorsys
from typing import Final

import numpy as np
import rerun as rr
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

TRAIL: Final[str] = "trail"
"""Per-pose segments under a run (dataforge's ``schema.trail_path``)."""
TRAJECTORY: Final[str] = "trajectory"
"""The whole path under a run (dataforge's ``schema.trajectory_path``)."""
ENDPOINTS: Final[str] = "endpoints"
"""The start and end dots under a run."""
TIMELINE: Final[str] = "video_time"
"""The base video's timeline every row is on."""
WALK_GO_HUE: Final[float] = 120.0
"""Hue of a path at its first pose: green."""
WALK_STOP_HUE: Final[float] = 0.0
"""Hue of a path at its last pose: red, through yellow and orange (HSV, so no muddy brown between)."""
WALK_SATURATION: Final[float] = 0.85
"""Saturation of the walk gradient."""
PATH_RADIUS_UI: Final[float] = 2.0
"""Stroke of a path, ui points."""
ENDPOINT_RADIUS_UI: Final[float] = 9.0
"""Radius of the start and end dots, ui points."""
START_RING_M: Final[float] = 0.25
"""Radius of the ring around the start dot, metres: a walk that passes its start again keeps the start readable."""
START_RING_RADIUS_UI: Final[float] = 2.5
"""Stroke of the ring, ui points."""
RING_POINTS: Final[int] = 49
"""Points on the ring (the first repeated last)."""


def walk_colors(t_ns: Int64[ndarray, " n"], first_ns: int, last_ns: int) -> UInt8[ndarray, "n 3"]:
    """The walk gradient at each stamp: green at ``first_ns``, through yellow, to red at ``last_ns``."""
    fraction: Float64[ndarray, " n"] = np.clip((t_ns - first_ns) / max(last_ns - first_ns, 1), 0.0, 1.0)
    rgb: list[tuple[float, float, float]] = [colorsys.hsv_to_rgb((WALK_GO_HUE + (WALK_STOP_HUE - WALK_GO_HUE) * f) / 360.0, WALK_SATURATION, 1.0) for f in fraction]
    return np.round(np.asarray(rgb, dtype=np.float64).reshape(-1, 3) * 255.0).astype(np.uint8)


def segments(positions: Float64[ndarray, "n 3"]) -> Float64[ndarray, "s 2 3"]:
    """Consecutive positions as two-point segments (``n - 1`` of them)."""
    return np.stack([positions[:-1], positions[1:]], axis=1)


def trail_segments(positions: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2 3"]:
    """One segment per position, the step that reached it; the first is zero-length."""
    previous: Int64[ndarray, " n"] = np.maximum(np.arange(len(positions), dtype=np.int64) - 1, 0)
    return np.stack([positions[previous], positions], axis=1)


def _at(timeline: str, t_ns: Int64[ndarray, " n"]) -> list[rr.TimeColumn]:
    return [rr.TimeColumn(timeline, duration=t_ns.astype("timedelta64[ns]"))]


def send_trail(
    entity: str,
    t_ns: Int64[ndarray, " n"],
    positions: Float64[ndarray, "n 3"],
    colors: UInt8[ndarray, "n 3"] | tuple[int, int, int],
    *,
    timeline: str = TIMELINE,
    radius_ui: float = PATH_RADIUS_UI,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Send a trail at ``entity``: row ``k`` at ``t_ns[k]`` is the step from pose ``k - 1`` to pose ``k`` (the first
    zero-length), in the colour of pose ``k``, or in one colour for every row (static, with the ui-point radius)."""
    flat: bool = isinstance(colors, tuple)
    strips: Float64[ndarray, "n 2 3"] = trail_segments(positions)
    columns = rr.LineStrips3D.columns(strips=list(strips), colors=None if flat else colors).partition([1] * len(strips))
    rr.send_columns(entity, indexes=_at(timeline, t_ns), columns=columns, recording=recording)
    rr.log(entity, rr.LineStrips3D.from_fields(colors=colors if flat else None, radii=rr.Radius.ui_points(radius_ui)), static=True, recording=recording)


def send_trajectory(
    entity: str,
    at_ns: int,
    positions: Float64[ndarray, "n 3"],
    colors: UInt8[ndarray, "n 3"] | tuple[int, int, int],
    *,
    timeline: str = TIMELINE,
    radius_ui: float = PATH_RADIUS_UI,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Send the whole path at ``entity`` as one row at ``at_ns``, each segment in the colour of its later pose, or in one
    colour (static, with the ui-point radius)."""
    flat: bool = isinstance(colors, tuple)
    strips = rr.LineStrips3D.columns(strips=list(segments(positions)), colors=None if flat else colors[1:])
    rr.send_columns(entity, indexes=_at(timeline, np.array([at_ns], dtype=np.int64)), columns=strips.partition([len(positions) - 1]), recording=recording)
    rr.log(entity, rr.LineStrips3D.from_fields(colors=colors if flat else None, radii=rr.Radius.ui_points(radius_ui)), static=True, recording=recording)


def send_endpoints(
    entity: str,
    at_ns: int,
    start: Float64[ndarray, " 3"],
    end: Float64[ndarray, " 3"],
    colors: tuple[UInt8[ndarray, " 3"], UInt8[ndarray, " 3"]],
    *,
    timeline: str = TIMELINE,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Send ``<entity>/start``, a dot, and ``<entity>/start_ring``, a ring around it (static: where the walk began shows
    from the first frame), and ``<entity>/end``, a dot at ``at_ns`` (from then on)."""
    radius = rr.Radius.ui_points(ENDPOINT_RADIUS_UI)
    rr.log(f"{entity}/start", rr.Points3D([start], colors=[colors[0]], radii=radius, labels=["start"], show_labels=True), static=True, recording=recording)
    angles: Float64[ndarray, " k"] = np.linspace(0.0, 2.0 * np.pi, RING_POINTS)
    ring: Float64[ndarray, "k 3"] = start + START_RING_M * np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    rr.log(f"{entity}/start_ring", rr.LineStrips3D([ring], colors=[colors[0]], radii=rr.Radius.ui_points(START_RING_RADIUS_UI)), static=True, recording=recording)
    columns = rr.Points3D.columns(positions=[end], colors=[colors[1]])
    rr.send_columns(f"{entity}/end", indexes=_at(timeline, np.array([at_ns], dtype=np.int64)), columns=columns, recording=recording)
    rr.log(f"{entity}/end", rr.Points3D.from_fields(radii=radius, labels=["end"], show_labels=True), static=True, recording=recording)


def send_walk_path(
    run: str,
    t_ns: Int64[ndarray, " n"],
    positions: Float64[ndarray, "n 3"],
    color: tuple[int, int, int] | None = None,
    *,
    timeline: str = TIMELINE,
    radius_ui: float = PATH_RADIUS_UI,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Send a run's whole walk path under ``run``: the trail, the trajectory and the end dot at the last pose, and the
    start dot and ring; in the walk gradient, or in one ``color``. Nothing with fewer than two poses."""
    if len(t_ns) < 2:
        return
    gradient: UInt8[ndarray, "n 3"] = walk_colors(t_ns, int(t_ns[0]), int(t_ns[-1])) if color is None else np.tile(np.asarray(color, dtype=np.uint8), (len(t_ns), 1))
    path_colors: UInt8[ndarray, "n 3"] | tuple[int, int, int] = gradient if color is None else color
    send_trail(f"{run}/{TRAIL}", t_ns, positions, path_colors, timeline=timeline, radius_ui=radius_ui, recording=recording)
    send_trajectory(f"{run}/{TRAJECTORY}", int(t_ns[-1]), positions, path_colors, timeline=timeline, radius_ui=radius_ui, recording=recording)
    send_endpoints(f"{run}/{ENDPOINTS}", int(t_ns[-1]), positions[0], positions[-1], (gradient[0], gradient[-1]), timeline=timeline, recording=recording)
