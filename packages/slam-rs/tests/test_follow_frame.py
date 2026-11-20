"""The follow frame keeps the walk's heading and nothing else of the rig's rotation; the floor grid lies under the walk."""

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs.follow_frame import FloorGrid, floor_grid, follow_yaw, rig_height_m

RATE_HZ: float = 30.0


def times(seconds: float) -> Int64[ndarray, " n"]:
    """Frameset stamps at 30 Hz over ``seconds``."""
    return (np.arange(round(seconds * RATE_HZ)) / RATE_HZ * 1e9).astype(np.int64)


def walk(t_ns: Int64[ndarray, " n"], velocity_mps: tuple[float, float, float]) -> Float64[ndarray, "n 3"]:
    """A straight walk from the origin."""
    return np.outer(t_ns * 1e-9, np.array(velocity_mps, dtype=np.float64))


def test_a_straight_walk_faces_its_direction_from_the_second_pose_on() -> None:
    """Walking along +y at 1 m/s gives a quarter turn of yaw, whatever the rig does about its own axes."""
    t_ns: Int64[ndarray, " n"] = times(4.0)
    yaw: Float64[ndarray, " n"] = follow_yaw(t_ns, walk(t_ns, (0.0, 1.0, 0.0)))
    np.testing.assert_allclose(yaw[1:], np.pi / 2, atol=1e-9)


def test_a_climb_does_not_turn_the_frame() -> None:
    """Only the ground-plane motion sets the heading: stairs along +x keep yaw at zero."""
    t_ns: Int64[ndarray, " n"] = times(4.0)
    yaw: Float64[ndarray, " n"] = follow_yaw(t_ns, walk(t_ns, (1.0, 0.0, 0.6)))
    np.testing.assert_allclose(yaw[1:], 0.0, atol=1e-9)


def test_the_heading_is_held_while_the_rig_stands_still() -> None:
    """A walker who stops and shakes the stick by millimetres keeps the heading of the walk."""
    t_ns: Int64[ndarray, " n"] = times(14.0)
    moving: int = round(4.0 * RATE_HZ)
    position_m: Float64[ndarray, "n 3"] = walk(t_ns, (0.0, 1.0, 0.0))
    position_m[moving:] = position_m[moving]
    position_m[moving:] += np.random.default_rng(0).normal(scale=0.003, size=(len(t_ns) - moving, 3))
    yaw: Float64[ndarray, " n"] = follow_yaw(t_ns, position_m)
    np.testing.assert_allclose(yaw[-round(5.0 * RATE_HZ) :], np.pi / 2, atol=np.radians(3.0))


def test_a_turn_is_followed_smoothly() -> None:
    """After a sharp quarter turn the heading is part of the way one time constant later and there after five."""
    t_ns: Int64[ndarray, " n"] = times(14.0)
    turn: int = round(4.0 * RATE_HZ)
    position_m: Float64[ndarray, "n 3"] = walk(t_ns, (1.0, 0.0, 0.0))
    position_m[turn:] = position_m[turn] + walk(t_ns[turn:] - t_ns[turn], (0.0, 1.0, 0.0))
    yaw: Float64[ndarray, " n"] = follow_yaw(t_ns, position_m, tau_s=1.5)
    one_tau: float = float(yaw[turn + round(1.5 * RATE_HZ)])
    assert np.radians(40.0) < one_tau < np.radians(75.0), np.degrees(one_tau)
    np.testing.assert_allclose(yaw[turn + round(7.5 * RATE_HZ)], np.pi / 2, atol=np.radians(1.0))
    assert np.all(np.abs(np.diff(yaw)) < np.radians(3.0)), "no jump between two framesets"


def room(path_m: Float64[ndarray, "n 3"], rig_height: float, *, climb_per_m: float = 0.0, seed: int = 0) -> Float64[ndarray, "p 3"]:
    """Points of a corridor around a walk along +x that starts ``rig_height`` over the floor.

    The floor climbs ``climb_per_m`` along x (as the walk must); a ceiling lies 3 m over it, walls between, and a few
    points far below.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    xy_m: Float64[ndarray, "p 2"] = path_m[rng.integers(0, len(path_m), size=6000), :2] + rng.uniform(-2.5, 2.5, size=(6000, 2))
    floor_z: Float64[ndarray, " p"] = path_m[0, 2] - rig_height + climb_per_m * (xy_m[:, 0] - path_m[0, 0])
    kind: Int64[ndarray, " p"] = rng.integers(0, 3, size=6000)
    z: Float64[ndarray, " p"] = np.where(kind == 0, floor_z, np.where(kind == 1, floor_z + 3.0, floor_z + rng.uniform(0.0, 3.0, size=6000)))
    points_m: Float64[ndarray, "p 3"] = np.column_stack([xy_m, z + rng.normal(scale=0.02, size=6000)])
    points_m[:40, 2] -= 12.0  # stray points under the floor
    return points_m


def test_the_rig_height_is_measured_from_the_map_under_the_walk() -> None:
    """A rig carried 1.6 m over a level floor: the map's low points under the path give that height."""
    t_ns: Int64[ndarray, " n"] = times(20.0)
    path_m: Float64[ndarray, "n 3"] = walk(t_ns, (1.0, 0.0, 0.0)) + np.array([0.0, 0.0, 0.4])
    height: float | None = rig_height_m(path_m, room(path_m, 1.6))
    assert height is not None and abs(height - 1.6) < 0.1, height


def test_the_rig_height_follows_a_staircase() -> None:
    """On stairs the floor rises with the walk, so the height stays the carrying height and not the climb."""
    t_ns: Int64[ndarray, " n"] = times(20.0)
    path_m: Float64[ndarray, "n 3"] = walk(t_ns, (1.0, 0.0, 0.15))
    height: float | None = rig_height_m(path_m, room(path_m, 1.6, climb_per_m=0.15))
    assert height is not None and abs(height - 1.6) < 0.1, height


def test_no_map_gives_no_height() -> None:
    """Without points near the walk there is nothing to measure the floor from."""
    t_ns: Int64[ndarray, " n"] = times(5.0)
    assert rig_height_m(walk(t_ns, (1.0, 0.0, 0.0)), np.zeros((0, 3))) is None


def test_the_floor_grid_is_metre_cells_under_the_walk_that_fade_away_from_it() -> None:
    """Every edge is 1 m long, lies the rig height under the walk, and is brightest at the path and gone 13 m away."""
    t_ns: Int64[ndarray, " n"] = times(20.0)
    path_m: Float64[ndarray, "n 3"] = walk(t_ns, (1.0, 0.0, 0.0)) + np.array([0.0, 0.0, 0.4])
    grid: FloorGrid = floor_grid(path_m, 1.6)
    edges: Float64[ndarray, "s 3"] = (grid.strips[:, 1] - grid.strips[:, 0]).astype(np.float64)
    np.testing.assert_allclose(np.linalg.norm(edges, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(grid.strips[:, :, 2], 0.4 - 1.6, atol=1e-5)
    middle: Float64[ndarray, "s 3"] = grid.strips.mean(axis=1).astype(np.float64)
    on_the_path: bool = bool(np.any((np.abs(middle[:, 1]) < 0.6) & (np.abs(middle[:, 0] - 10.0) < 0.6)))
    assert on_the_path, "the grid covers the walk"
    assert np.abs(middle[:, 1]).max() < 13.0 and middle[:, 0].min() > -13.0, "nothing 13 m from the path"
    brightness: Float64[ndarray, " s"] = grid.colors.astype(np.float64).sum(axis=1)
    beside: Float64[ndarray, "s 3"] = middle[(middle[:, 0] > 0.0) & (middle[:, 0] < 20.0)]
    beside_brightness: Float64[ndarray, " s"] = brightness[(middle[:, 0] > 0.0) & (middle[:, 0] < 20.0)]
    near: Float64[ndarray, " s"] = beside_brightness[np.abs(beside[:, 1]) < 2.0]
    far: Float64[ndarray, " s"] = beside_brightness[np.abs(beside[:, 1]) > 10.0]
    assert near.min() > far.max(), "bright near the path, dim far from it"
