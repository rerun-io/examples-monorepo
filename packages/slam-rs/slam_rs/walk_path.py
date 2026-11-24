"""A run's path under ``/world/runs/<run>``, as the replay and the catalog layer write it, in dataforge's names.

- ``<run>/trail``: one two-point segment per pose, on ``video_time`` at the later pose: a view's visible time range
  makes the path so far (dataforge's ``schema.trail_path``).
- ``<run>/trajectory``: the whole path (dataforge's ``schema.trajectory_path``); the replay logs it once at the last
  pose, so no whole path stands over a cursor halfway through a clip.
- ``<run>/endpoints/start`` (a dot and a ring, static) and ``<run>/endpoints/end`` (a dot, from the last pose on).

The walk gradient (``walk_colors``) runs green at the first pose through yellow to red at the last. colsfm draws its
path with a copy of it (``colsfm.rerun_layer.walk_colors``; a test in each package pins the same colours).

Only numpy and rerun here: the replay's core runs without dataforge (macOS), so the names are restated and a test pins
them to dataforge's.
"""

import colorsys

import numpy as np
import rerun as rr
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

TRAIL: str = "trail"
"""Per-pose segments under a run (dataforge's ``schema.trail_path``)."""
TRAJECTORY: str = "trajectory"
"""The whole path under a run (dataforge's ``schema.trajectory_path``)."""
ENDPOINTS: str = "endpoints"
"""The start and end dots under a run."""
TIMELINE: str = "video_time"
"""The base video's timeline every row is on."""
WALK_GO_HUE: float = 120.0
"""Hue of a path at its first pose: green."""
WALK_STOP_HUE: float = 0.0
"""Hue of a path at its last pose: red, through yellow and orange (HSV, so no muddy brown between)."""
WALK_SATURATION: float = 0.85
"""Saturation of the walk gradient."""
PATH_RADIUS_M: float = 0.004
"""Stroke of a path, metres."""
ENDPOINT_RADIUS_UI: float = 9.0
"""Radius of the start and end dots, ui points."""
START_RING_M: float = 0.25
"""Radius of the ring around the start dot, metres: a walk that passes its start again keeps the start readable."""
START_RING_RADIUS_UI: float = 2.5
"""Stroke of the ring, ui points."""
RING_POINTS: int = 49
"""Points on the ring (the first repeated last)."""


def walk_colors(t_ns: Int64[ndarray, " n"], first_ns: int, last_ns: int) -> UInt8[ndarray, "n 3"]:
    """The walk gradient at each stamp: green at ``first_ns``, through yellow, to red at ``last_ns``."""
    fraction: Float64[ndarray, " n"] = np.clip((t_ns - first_ns) / max(last_ns - first_ns, 1), 0.0, 1.0)
    rgb: list[tuple[float, float, float]] = [colorsys.hsv_to_rgb((WALK_GO_HUE + (WALK_STOP_HUE - WALK_GO_HUE) * f) / 360.0, WALK_SATURATION, 1.0) for f in fraction]
    return np.round(np.asarray(rgb, dtype=np.float64).reshape(-1, 3) * 255.0).astype(np.uint8)


def segments(positions: Float64[ndarray, "n 3"]) -> Float64[ndarray, "s 2 3"]:
    """Consecutive positions as two-point segments."""
    return np.stack([positions[:-1], positions[1:]], axis=1)


def send_trail(
    run: str,
    t_ns: Int64[ndarray, " n"],
    positions: Float64[ndarray, "n 3"],
    colors: UInt8[ndarray, "n 3"] | tuple[int, int, int],
    *,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Send ``<run>/trail``: segment ``k`` (pose ``k`` to ``k + 1``) at ``t_ns[k + 1]`` (``video_time``, ns), in the colour
    of its later pose, or in one colour for every segment."""
    index = [rr.TimeColumn(TIMELINE, duration=t_ns[1:].astype("timedelta64[ns]"))]
    strips: Float64[ndarray, "s 2 3"] = segments(positions)
    if isinstance(colors, tuple):
        rr.send_columns(f"{run}/{TRAIL}", indexes=index, columns=rr.LineStrips3D.columns(strips=list(strips)).partition([1] * len(strips)), recording=recording)
        rr.log(f"{run}/{TRAIL}", rr.LineStrips3D.from_fields(colors=colors, radii=PATH_RADIUS_M), static=True, recording=recording)
        return
    columns = rr.LineStrips3D.columns(strips=list(strips), colors=colors[1:]).partition([1] * len(strips))
    rr.send_columns(f"{run}/{TRAIL}", indexes=index, columns=columns, recording=recording)
    rr.log(f"{run}/{TRAIL}", rr.LineStrips3D.from_fields(radii=PATH_RADIUS_M), static=True, recording=recording)


def send_trajectory(
    run: str, at_ns: int, positions: Float64[ndarray, "n 3"], colors: UInt8[ndarray, "n 3"] | tuple[int, int, int], *, recording: rr.RecordingStream | None = None
) -> None:
    """Send ``<run>/trajectory``: the whole path as one row at ``at_ns`` (``video_time``), each segment in the colour of
    its later pose, or in one colour."""
    flat: bool = isinstance(colors, tuple)
    strips = rr.LineStrips3D.columns(strips=list(segments(positions)), colors=None if flat else colors[1:], radii=[PATH_RADIUS_M] * (len(positions) - 1))
    rr.send_columns(f"{run}/{TRAJECTORY}", indexes=[rr.TimeColumn(TIMELINE, duration=[np.timedelta64(at_ns, "ns")])], columns=strips.partition([len(positions) - 1]), recording=recording)
    if flat:
        rr.log(f"{run}/{TRAJECTORY}", rr.LineStrips3D.from_fields(colors=colors), static=True, recording=recording)


def send_endpoints(
    run: str, at_ns: int, start: Float64[ndarray, " 3"], end: Float64[ndarray, " 3"], colors: tuple[UInt8[ndarray, " 3"], UInt8[ndarray, " 3"]], *, recording: rr.RecordingStream | None = None
) -> None:
    """Send ``<run>/endpoints``: the start dot and a ring around it (static: where the walk began shows from the first
    frame), and the end dot at ``at_ns`` (``video_time``, from the last pose on)."""
    radius = rr.Radius.ui_points(ENDPOINT_RADIUS_UI)
    rr.log(f"{run}/{ENDPOINTS}/start", rr.Points3D([start], colors=[colors[0]], radii=radius, labels=["start"], show_labels=True), static=True, recording=recording)
    angles: Float64[ndarray, " k"] = np.linspace(0.0, 2.0 * np.pi, RING_POINTS)
    ring: Float64[ndarray, "k 3"] = start + START_RING_M * np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    rr.log(f"{run}/{ENDPOINTS}/start_ring", rr.LineStrips3D([ring], colors=[colors[0]], radii=rr.Radius.ui_points(START_RING_RADIUS_UI)), static=True, recording=recording)
    index = [rr.TimeColumn(TIMELINE, duration=[np.timedelta64(at_ns, "ns")])]
    rr.send_columns(f"{run}/{ENDPOINTS}/end", indexes=index, columns=rr.Points3D.columns(positions=[end], colors=[colors[1]]), recording=recording)
    rr.log(f"{run}/{ENDPOINTS}/end", rr.Points3D.from_fields(radii=radius, labels=["end"], show_labels=True), static=True, recording=recording)
