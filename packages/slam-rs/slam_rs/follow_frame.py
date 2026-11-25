"""The level follow frame of a walked rig.

A viewer's follow view rooted at the rig rolls and pitches with it: the eye lives in the origin's frame. The follow
frame is the rig's position with a rotation about the world's up axis only, so a view rooted there keeps the horizon
level. Its heading is the direction of travel on the ground plane, low-passed, and held while the rig stands still:
a lens on a stick points anywhere, the walk does not.

It is per run, under ``/world/runs/<run>`` (:func:`follow_entity`): the replay writes the raw VIO run's as it goes and
the final run's once it is known (:mod:`slam_rs.vio_log`); the catalog layer writes both (:mod:`slam_rs.catalog_layer`).
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray


def follow_entity(run_entity: str) -> str:
    """A run's level follow frame: ``Transform3D`` on ``video_time`` (dataforge's ``schema.follow_path``).

    Run-scoped: a view that shows one run centres on that run's path. The causal VIO and the final loop-closed path
    part by metres (clip 007: 16 m at 65 s), so one frame for both would put one of them off screen.
    """
    return f"{run_entity}/follow"


HEADING_TAU_S: float = 1.5
"""Time constant of the heading's low-pass, seconds: a turn is followed in a few seconds, a swing of the stick is not."""
MIN_SPEED_MPS: float = 0.15
"""Smoothed ground speed under which the rig counts as standing still and the heading is held."""


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
