"""The follow frame keeps the walk's heading and nothing else of the rig's rotation; the floor grid lies under the walk."""

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs.follow_frame import FloorGrid, FloorHeight, floor_grid, follow_yaw

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


def surroundings(position_m: Float64[ndarray, " 3"], floor_z: float, rng: np.random.Generator) -> Float64[ndarray, "p 3"]:
    """What a window holds around one pose: floor points, a ceiling 3 m over the floor, walls between, and a few far below."""
    xy_m: Float64[ndarray, "p 2"] = position_m[:2] + rng.uniform(-4.0, 4.0, size=(120, 2))
    kind: Int64[ndarray, " p"] = rng.integers(0, 3, size=120)
    z: Float64[ndarray, " p"] = np.where(kind == 0, floor_z, np.where(kind == 1, floor_z + 3.0, floor_z + rng.uniform(0.0, 3.0, size=120)))
    points_m: Float64[ndarray, "p 3"] = np.column_stack([xy_m, z + rng.normal(scale=0.02, size=120)])
    points_m[0, 2] -= 12.0  # a stray point under the floor
    return points_m


def measured(path_m: Float64[ndarray, "n 3"], floor_z: Float64[ndarray, " n"]) -> float | None:
    """The carrying height after every pose of a walk has shown the landmarks around it."""
    rng: np.random.Generator = np.random.default_rng(0)
    height: FloorHeight = FloorHeight()
    for position_m, z in zip(path_m, floor_z, strict=True):
        height.add(position_m, surroundings(position_m, float(z), rng))
    return height.height_m()


def test_the_rig_height_is_measured_from_the_landmarks_around_the_rig() -> None:
    """A rig carried 1.6 m over a level floor: the low landmarks around it give that height."""
    path_m: Float64[ndarray, "n 3"] = walk(times(20.0), (1.0, 0.0, 0.0)) + np.array([0.0, 0.0, 0.4])
    height: float | None = measured(path_m, np.full(len(path_m), 0.4 - 1.6))
    assert height is not None and abs(height - 1.6) < 0.1, height


def test_the_rig_height_holds_on_a_walk_over_several_floors() -> None:
    """A walk that comes back over its own ground one storey up: each pose is measured against the landmarks of its
    own time, so the storey below does not count as a floor 5 m under the rig."""
    first: Float64[ndarray, "n 3"] = walk(times(20.0), (1.0, 0.0, 0.0))
    path_m: Float64[ndarray, "m 3"] = np.concatenate([first, first + np.array([0.0, 0.0, 3.5])])
    height: float | None = measured(path_m, path_m[:, 2] - 1.6)
    assert height is not None and abs(height - 1.6) < 0.1, height


def test_too_few_landmarks_give_no_height() -> None:
    """Without landmarks around the rig there is nothing to measure the floor from."""
    height: FloorHeight = FloorHeight()
    assert height.height_m() is None
    height.add(np.zeros(3), np.array([[50.0, 0.0, -1.6]]))  # one landmark, far from the rig
    assert height.height_m() is None


def test_the_floor_grid_is_metre_cells_under_the_walk_that_fade_away_from_it() -> None:
    """Every edge is 1 m long, lies the rig height under the walk, and is brightest at the path and gone 13 m away."""
    t_ns: Int64[ndarray, " n"] = times(20.0)
    path_m: Float64[ndarray, "n 3"] = walk(t_ns, (1.0, 0.0, 0.0)) + np.array([0.0, 0.0, 0.4])
    grid: FloorGrid = floor_grid(t_ns, path_m, 1.6)
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


def test_each_floor_edge_carries_the_time_the_walk_passed_it() -> None:
    """An edge is stamped with the time of the pose it lies under, in time order, so a view can show the floor around
    the cursor: the edge under the walk's 5th metre has the 5th second."""
    t_ns: Int64[ndarray, " n"] = times(20.0)
    grid: FloorGrid = floor_grid(t_ns, walk(t_ns, (1.0, 0.0, 0.0)), 1.6)
    assert len(grid.t_ns) == len(grid.strips) and np.all(np.diff(grid.t_ns) >= 0)
    middle: Float64[ndarray, "s 3"] = grid.strips.mean(axis=1).astype(np.float64)
    under_the_fifth_metre: Int64[ndarray, " k"] = grid.t_ns[(np.abs(middle[:, 0] - 5.0) < 0.6) & (np.abs(middle[:, 1]) < 0.6)]
    assert len(under_the_fifth_metre) > 0
    np.testing.assert_allclose(under_the_fifth_metre * 1e-9, 5.0, atol=1.0)


def test_a_walk_over_two_storeys_gets_a_floor_under_each() -> None:
    """Where the walk passes the same ground one storey up, both floors are laid there, each at its own height and time."""
    first_ns: Int64[ndarray, " n"] = times(20.0)
    first: Float64[ndarray, "n 3"] = walk(first_ns, (1.0, 0.0, 0.0))
    t_ns: Int64[ndarray, " m"] = np.concatenate([first_ns, first_ns + 100_000_000_000])
    path_m: Float64[ndarray, "m 3"] = np.concatenate([first, first + np.array([0.0, 0.0, 3.5])])
    grid: FloorGrid = floor_grid(t_ns, path_m, 1.6)
    middle: Float64[ndarray, "s 3"] = grid.strips.mean(axis=1).astype(np.float64)
    here = (np.abs(middle[:, 0] - 10.0) < 0.6) & (np.abs(middle[:, 1]) < 0.6)
    heights: list[float] = sorted({round(float(z), 2) for z in middle[here, 2]})
    assert heights == [-1.6, 1.9], heights
    assert np.all(grid.t_ns[here & (middle[:, 2] < 0.0)] < 30_000_000_000), "the lower floor belongs to the first pass"
    assert np.all(grid.t_ns[here & (middle[:, 2] > 0.0)] > 100_000_000_000), "the upper floor to the second"
