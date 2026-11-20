"""The level follow frame of a walked rig, and the floor grid under the walk.

A viewer's follow view rooted at the rig rolls and pitches with it: the eye lives in the origin's frame. The follow
frame is the rig's position with a rotation about the world's up axis only, so a view rooted there keeps the horizon
level. Its heading is the direction of travel on the ground plane, low-passed, and held while the rig stands still:
a lens on a stick points anywhere, the walk does not.

The floor grid is one-metre cells the rig's carrying height under the walk, fixed in the world. A view's built-in grid
lies on its origin's plane: it stands still under a view that follows the rig, and in a world view it sits at the
rig's first height, not on the floor.

Both belong to the layer that moves ``/world/rig_00`` (:mod:`slam_rs.catalog_layer`, :mod:`slam_rs.vio_log`).
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from scipy.spatial import KDTree

from slam_rs.rig import RIG_ENTITY

FOLLOW_ENTITY: str = f"{RIG_ENTITY}_follow"
"""The follow frame: ``Transform3D`` on ``video_time`` (dataforge's ``schema.follow_path``)."""
FLOOR_GRID_ENTITY: str = "/world/floor_grid"
"""The floor grid: one static ``LineStrips3D`` (dataforge's ``schema.FLOOR_GRID_PATH``)."""
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
"""Ground-plane radius around a path sample within which map points count as its surroundings."""
HEIGHT_PERCENTILE: float = 5.0
"""Percentile of the surrounding points' heights taken as the floor: under it lie only stray points."""
HEIGHT_MIN_POINTS: int = 30
"""Fewest surrounding points a path sample needs to give a floor height."""
HEIGHT_SAMPLE_M: float = 1.0
"""Path length between two samples of the rig height."""


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


def rig_height_m(
    position_m: Float64[ndarray, "n 3"],
    points_m: Float64[ndarray, "p 3"],
    *,
    radius_m: float = HEIGHT_RADIUS_M,
    percentile: float = HEIGHT_PERCENTILE,
) -> float | None:
    """How high over the floor the rig is carried, measured from the map around the walk.

    Each map point's height is taken against the pose nearest to it on the ground plane, so a floor that rises with
    the walk (a ramp, a staircase) stays one height under it. At a sample every metre of the walk, the floor is the
    low percentile of those heights within ``radius_m``; the result is the median over the samples, so a stretch
    beside a drop (a balcony over a hall) does not move it.

    Args:
        position_m: Rig positions of the walk, z-up world, metres.
        points_m: Map points in the same world.
        radius_m: Ground-plane radius of a sample's surroundings.
        percentile: Percentile of the surrounding heights taken as the floor.

    Returns:
        The carrying height in metres, or None when no sample has enough points around it.
    """
    finite: Float64[ndarray, "q 3"] = points_m[np.isfinite(points_m).all(axis=1)]
    if len(finite) < HEIGHT_MIN_POINTS or len(position_m) == 0:
        return None
    _, nearest_pose = KDTree(position_m[:, :2]).query(finite[:, :2])
    under_rig_m: Float64[ndarray, " q"] = position_m[nearest_pose, 2] - finite[:, 2]
    samples: Float64[ndarray, "k 3"] = position_m[path_samples(position_m, HEIGHT_SAMPLE_M)]
    heights: list[float] = []
    for around in KDTree(finite[:, :2]).query_ball_point(samples[:, :2], radius_m):
        if len(around) >= HEIGHT_MIN_POINTS:
            heights.append(float(np.percentile(under_rig_m[around], 100.0 - percentile)))
    return float(np.median(heights)) if heights else None


@dataclass(frozen=True, slots=True)
class FloorGrid:
    """The floor grid as ``LineStrips3D`` columns."""

    strips: Float32[ndarray, "s 2 3"]
    """One cell edge per strip, world frame, metres."""
    colors: UInt8[ndarray, "s 3"]
    """Each edge's colour: :data:`GRID_COLOR` at the path, faded into the background away from it."""


def floor_grid(position_m: Float64[ndarray, "n 3"], rig_height: float, *, near_m: float = GRID_NEAR_M, far_m: float = GRID_FAR_M) -> FloorGrid:
    """One-metre cells on the floor under the walk, bright at the path and faded out ``far_m`` from it.

    Each edge lies ``rig_height`` under the nearest pose, so the grid climbs a staircase with the walk. The fade is in
    the colour, towards the viewer's background.

    Args:
        position_m: Rig positions of the walk, z-up world, metres.
        rig_height: How high over the floor the rig is carried (:func:`rig_height_m`), metres.
        near_m: Distance from the path up to which an edge has the full colour.
        far_m: Distance from the path at which the edges end.
    """
    tree: KDTree = KDTree(position_m[:, :2])
    low: Float64[ndarray, " 2"] = np.floor(position_m[:, :2].min(axis=0) - far_m)
    high: Float64[ndarray, " 2"] = np.ceil(position_m[:, :2].max(axis=0) + far_m)
    grid_x, grid_y = np.meshgrid(np.arange(low[0], high[0] + 1.0), np.arange(low[1], high[1] + 1.0), indexing="ij")
    corners: Float64[ndarray, "c 2"] = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    starts: Float64[ndarray, "e 2"] = np.concatenate([corners, corners])
    ends: Float64[ndarray, "e 2"] = np.concatenate([corners + np.array([1.0, 0.0]), corners + np.array([0.0, 1.0])])
    distance_m, nearest = tree.query(0.5 * (starts + ends))
    keep = distance_m < far_m
    fade: Float64[ndarray, " s"] = np.clip((far_m - distance_m[keep]) / (far_m - near_m), 0.0, 1.0) ** 1.5
    floor_z: Float64[ndarray, " s"] = position_m[nearest[keep], 2] - rig_height
    strips: Float64[ndarray, "s 2 3"] = np.stack([np.column_stack([starts[keep], floor_z]), np.column_stack([ends[keep], floor_z])], axis=1)
    colors: Float64[ndarray, "s 3"] = np.outer(fade, np.array(GRID_COLOR, dtype=np.float64)) + np.outer(1.0 - fade, np.array(BACKGROUND_COLOR, dtype=np.float64))
    return FloorGrid(strips=strips.astype(np.float32), colors=np.rint(colors).astype(np.uint8))
