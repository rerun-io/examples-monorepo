"""The follow frame keeps the walk's heading and nothing else of the rig's rotation."""

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs.follow_frame import follow_yaw

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
