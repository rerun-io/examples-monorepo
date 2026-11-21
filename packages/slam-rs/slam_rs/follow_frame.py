"""The level follow frame of a walked rig, and the floor grid under the walk.

A viewer's follow view rooted at the rig rolls and pitches with it: the eye lives in the origin's frame. The follow
frame is the rig's position with a rotation about the world's up axis only, so a view rooted there keeps the horizon
level. Its heading is the direction of travel on the ground plane, low-passed, and held while the rig stands still:
a lens on a stick points anywhere, the walk does not.

The floor grid is one-metre cells the rig's carrying height under the walk, fixed in the world. A view's built-in grid
lies on its origin's plane: it stands still under a view that follows the rig, and in a world view it sits at the
rig's first height, not on the floor. Each edge carries the time the walk passed it, so a view can show the floor
around the cursor: on a walk over several storeys a follow view then draws the floor under the rig and not the one
above it.

Both are per run, under ``/world/runs/<run>`` (:func:`follow_entity`, :func:`floor_entity`): the replay writes the raw
VIO run's as it goes and the final run's once it is known (:mod:`slam_rs.vio_log`); the catalog layer writes both
follow frames (:mod:`slam_rs.catalog_layer`).
"""

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from scipy.spatial import KDTree


def follow_entity(run_entity: str) -> str:
    """A run's level follow frame: ``Transform3D`` on ``video_time`` (dataforge's ``schema.follow_path``).

    Run-scoped: a view that shows one run centres on that run's path. The causal VIO and the final loop-closed path
    part by metres (clip 007: 16 m at 65 s), so one frame for both would put one of them off screen.
    """
    return f"{run_entity}/follow"


def floor_entity(run_entity: str) -> str:
    """The floor under a run's walk: ``LineStrips3D`` rows on ``video_time`` (dataforge's ``schema.floor_path``)."""
    return f"{run_entity}/floor_grid"

HEADING_TAU_S: float = 1.5
"""Time constant of the heading's low-pass, seconds: a turn is followed in a few seconds, a swing of the stick is not."""
MIN_SPEED_MPS: float = 0.15
"""Smoothed ground speed under which the rig counts as standing still and the heading is held."""
GRID_NEAR_M: float = 3.0
"""Distance from the path within which the floor grid is at full brightness."""
GRID_FAR_M: float = 13.0
"""Distance from the path at which the floor grid has faded out; no cell edge lies farther."""
GRID_COLOR: tuple[int, int, int] = (84, 86, 96)
"""A cell edge at the path: a cool grey under every source colour."""
BACKGROUND_COLOR: tuple[int, int, int] = (13, 13, 14)
"""What the grid fades into: the viewer's dark background."""
HEIGHT_RADIUS_M: float = 3.0
"""Ground-plane radius around the rig within which a window's landmarks count as its surroundings."""
HEIGHT_PERCENTILE: float = 95.0
"""Percentile of the landmarks' depths under the rig taken as the floor: over it lie only stray points. Clip B (one
level, 60 s) and clip 007 (several storeys, 1,069 s) both end their depths in one dense layer at 1.75 to 2.0 m, with
1.5 % of the landmarks under it; this percentile reads 1.90 m and 1.85 m."""
HEIGHT_MIN_LANDMARKS: int = 500
"""Fewest landmarks a walk must have shown around the rig before it gives a floor height."""
HEIGHT_BIN_M: float = 0.01
"""Resolution of the depth histogram."""
HEIGHT_RANGE_M: float = 10.0
"""Depths under (and heights over) the rig the histogram resolves; what lies beyond counts in its last bins."""
GRID_SAMPLE_M: float = 0.5
"""Walked ground between two poses the floor grid is laid under."""
LEVEL_M: float = 2.2
"""Two floors at one place on the ground are this far apart in height at least: under a storey, over a stair flight's step."""


@dataclass(slots=True)
class HeadingFilter:
    """The heading of travel so far, one pose at a time (causal, so a replay can log it as it goes)."""

    tau_s: float = HEADING_TAU_S
    """Time constant of the velocity low-pass, seconds."""
    min_speed_mps: float = MIN_SPEED_MPS
    """Smoothed ground speed under which the heading is held."""
    yaw_rad: float = 0.0
    """Heading about the world's z axis, from +x; zero until the rig first moves."""
    previous_t_ns: int | None = None
    """Stamp of the last pose taken; None before the first."""
    previous_xy_m: Float64[ndarray, " 2"] | None = None
    """Ground-plane position of the last pose taken."""
    velocity_mps: Float64[ndarray, " 2"] | None = None
    """Low-passed ground-plane velocity; None until two poses give the first one."""

    def update(self, t_ns: int, position_m: Float64[ndarray, " 3"]) -> float:
        """Take the next pose and return the heading at it, radians.

        Args:
            t_ns: The pose's stamp, nanoseconds; later than the previous one.
            position_m: The rig's position in a z-up world, metres.
        """
        xy_m: Float64[ndarray, " 2"] = np.asarray(position_m[:2], dtype=np.float64).copy()
        if self.previous_t_ns is not None and self.previous_xy_m is not None and t_ns > self.previous_t_ns:
            dt_s: float = (t_ns - self.previous_t_ns) * 1e-9
            measured_mps: Float64[ndarray, " 2"] = (xy_m - self.previous_xy_m) / dt_s
            if self.velocity_mps is None:
                self.velocity_mps = measured_mps
            else:
                self.velocity_mps = self.velocity_mps + (1.0 - float(np.exp(-dt_s / self.tau_s))) * (measured_mps - self.velocity_mps)
            if float(np.linalg.norm(self.velocity_mps)) > self.min_speed_mps:
                self.yaw_rad = float(np.arctan2(self.velocity_mps[1], self.velocity_mps[0]))
        self.previous_t_ns = t_ns
        self.previous_xy_m = xy_m
        return self.yaw_rad


def follow_yaw(t_ns: Int64[ndarray, " n"], position_m: Float64[ndarray, "n 3"], *, tau_s: float = HEADING_TAU_S) -> Float64[ndarray, " n"]:
    """The follow frame's heading at every pose of a finished run.

    Args:
        t_ns: Pose stamps, nanoseconds, increasing.
        position_m: Rig positions in a z-up world, metres.
        tau_s: Time constant of the low-pass, seconds.

    Returns:
        Yaw about the world's z axis at each pose, radians from +x, unwrapped.
    """
    heading: HeadingFilter = HeadingFilter(tau_s=tau_s)
    yaw: Float64[ndarray, " n"] = np.array([heading.update(int(t), position) for t, position in zip(t_ns, position_m, strict=True)], dtype=np.float64)
    return np.unwrap(yaw)


def yaw_quaternion_xyzw(yaw_rad: Float64[ndarray, " n"]) -> Float64[ndarray, "n 4"]:
    """Rotations about the world's z axis as Rerun's XYZW quaternions."""
    half: Float64[ndarray, " n"] = 0.5 * yaw_rad
    zeros: Float64[ndarray, " n"] = np.zeros_like(half)
    return np.column_stack([zeros, zeros, np.sin(half), np.cos(half)])


def path_samples(position_m: Float64[ndarray, "n 3"], spacing_m: float) -> Int64[ndarray, " k"]:
    """Indices of poses about ``spacing_m`` of walked ground apart, the first one included."""
    walked_m: Float64[ndarray, " n"] = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(position_m[:, :2], axis=0), axis=1))])
    cell: Int64[ndarray, " n"] = np.floor(walked_m / spacing_m).astype(np.int64)
    return np.flatnonzero(np.concatenate([[True], cell[1:] != cell[:-1]]))


@dataclass(slots=True)
class FloorHeight:
    """How high over the floor the rig is carried, measured as the walk goes from the landmarks around it.

    Each pose brings the landmarks its window holds; those within ``radius_m`` on the ground plane add their depth
    under the rig to one histogram, and the floor is its high percentile. Time-local on purpose: the final map of a
    walk over several storeys has, under one pose, the floors of every storey (clip 007: the map gave 4.8 m).
    """

    radius_m: float = HEIGHT_RADIUS_M
    """Ground-plane radius of the rig's surroundings."""
    percentile: float = HEIGHT_PERCENTILE
    """Percentile of the depths taken as the floor."""
    counts: Int64[ndarray, " bins"] = field(default_factory=lambda: np.zeros(round(2 * HEIGHT_RANGE_M / HEIGHT_BIN_M), dtype=np.int64))
    """Landmarks by depth under the rig, from ``-HEIGHT_RANGE_M`` (over it) to ``HEIGHT_RANGE_M``."""

    def add(self, position_m: Float64[ndarray, " 3"], landmarks_m: Float64[ndarray, "p 3"]) -> None:
        """Count the landmarks around one pose.

        Args:
            position_m: The rig's position in a z-up world, metres.
            landmarks_m: The landmarks its window holds, same world.
        """
        around = np.linalg.norm(landmarks_m[:, :2] - position_m[:2], axis=1) < self.radius_m
        depth_m: Float64[ndarray, " k"] = position_m[2] - landmarks_m[around, 2]
        bins: Int64[ndarray, " k"] = np.floor((depth_m[np.isfinite(depth_m)] + HEIGHT_RANGE_M) / HEIGHT_BIN_M).astype(np.int64)
        np.add.at(self.counts, np.clip(bins, 0, len(self.counts) - 1), 1)

    def height_m(self) -> float | None:
        """The carrying height in metres, or None while too few landmarks were seen around the rig."""
        total: int = int(self.counts.sum())
        if total < HEIGHT_MIN_LANDMARKS:
            return None
        at: int = int(np.searchsorted(np.cumsum(self.counts), self.percentile / 100.0 * total))
        return (at + 0.5) * HEIGHT_BIN_M - HEIGHT_RANGE_M


@dataclass(frozen=True, slots=True)
class FloorGrid:
    """The floor grid as ``LineStrips3D`` columns, in time order."""

    strips: Float32[ndarray, "s 2 3"]
    """One cell edge per strip, world frame, metres."""
    colors: UInt8[ndarray, "s 3"]
    """Each edge's colour: :data:`GRID_COLOR` at the path, faded into the background away from it."""
    t_ns: Int64[ndarray, " s"]
    """When the walk passed each edge: the stamp of the pose it lies under."""


def floor_grid(
    t_ns: Int64[ndarray, " n"], position_m: Float64[ndarray, "n 3"], rig_height: float, *, near_m: float = GRID_NEAR_M, far_m: float = GRID_FAR_M
) -> FloorGrid:
    """One-metre cells on the floor under the walk, bright at the path and faded out ``far_m`` from it.

    Each edge lies ``rig_height`` under the pose nearest to it on the ground plane and carries that pose's stamp. Where
    the walk passes one place at heights :data:`LEVEL_M` or more apart (two storeys, a staircase over a hall), the
    place gets an edge for each: the nearest pose first, then the nearest of those on another level, and so on.

    Args:
        t_ns: Pose stamps, nanoseconds.
        position_m: Rig positions of the walk, z-up world, metres.
        rig_height: How high over the floor the rig is carried (:class:`FloorHeight`), metres.
        near_m: Distance from the path up to which an edge has the full colour.
        far_m: Distance from the path at which the edges end.

    Returns:
        The edges in time order.
    """
    sampled: Int64[ndarray, " k"] = path_samples(position_m, GRID_SAMPLE_M)
    under_m: Float64[ndarray, "k 3"] = position_m[sampled] - np.array([0.0, 0.0, rig_height])
    low: Float64[ndarray, " 2"] = np.floor(under_m[:, :2].min(axis=0) - far_m)
    high: Float64[ndarray, " 2"] = np.ceil(under_m[:, :2].max(axis=0) + far_m)
    grid_x, grid_y = np.meshgrid(np.arange(low[0], high[0] + 1.0), np.arange(low[1], high[1] + 1.0), indexing="ij")
    corners: Float64[ndarray, "c 2"] = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    starts: Float64[ndarray, "e 2"] = np.concatenate([corners, corners])
    ends: Float64[ndarray, "e 2"] = np.concatenate([corners + np.array([1.0, 0.0]), corners + np.array([0.0, 1.0])])
    middles: Float64[ndarray, "e 2"] = 0.5 * (starts + ends)
    edge: list[int] = []
    pose: list[int] = []
    distance: list[float] = []
    for index, around in enumerate(KDTree(under_m[:, :2]).query_ball_point(middles, far_m)):
        if not around:
            continue
        candidates: Int64[ndarray, " a"] = np.asarray(around, dtype=np.int64)
        away_m: Float64[ndarray, " a"] = np.linalg.norm(under_m[candidates, :2] - middles[index], axis=1)
        order: Int64[ndarray, " a"] = np.argsort(away_m, kind="stable")
        order = order[away_m[order] < far_m]
        candidates, away_m = candidates[order], away_m[order]
        while len(candidates):
            edge.append(index)
            pose.append(int(candidates[0]))
            distance.append(float(away_m[0]))
            other_level = np.abs(under_m[candidates, 2] - under_m[candidates[0], 2]) >= LEVEL_M
            candidates, away_m = candidates[other_level], away_m[other_level]
    edges: Int64[ndarray, " s"] = np.asarray(edge, dtype=np.int64)
    poses: Int64[ndarray, " s"] = np.asarray(pose, dtype=np.int64)
    stamps: Int64[ndarray, " s"] = t_ns[sampled][poses]
    in_time: Int64[ndarray, " s"] = np.argsort(stamps, kind="stable")
    edges, poses, stamps = edges[in_time], poses[in_time], stamps[in_time]
    fade: Float64[ndarray, " s"] = np.clip((far_m - np.asarray(distance, dtype=np.float64)[in_time]) / (far_m - near_m), 0.0, 1.0) ** 1.5
    floor_z: Float64[ndarray, " s"] = under_m[poses, 2]
    strips: Float64[ndarray, "s 2 3"] = np.stack([np.column_stack([starts[edges], floor_z]), np.column_stack([ends[edges], floor_z])], axis=1)
    colors: Float64[ndarray, "s 3"] = np.outer(fade, np.array(GRID_COLOR, dtype=np.float64)) + np.outer(1.0 - fade, np.array(BACKGROUND_COLOR, dtype=np.float64))
    return FloorGrid(strips=strips.astype(np.float32), colors=np.rint(colors).astype(np.uint8), t_ns=stamps)
