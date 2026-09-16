"""Which way is up in a tracking world, measured from gravity.

A corpus that documents no world axes (MSD) and one that publishes them (LaMAria)
both need the same check: an accelerometer at rest measures the *reaction* to
gravity, so its reading points **up**, and rotating each sample into the world
with the ground truth's own orientation (``world_R_rig @ a_rig``) and averaging
yields a vector along the world's up axis. What each dataset then *does* with the
answer — declare it per device, or compare it with a published axis — stays with
the dataset; this module only measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge.logging_toolkit import ImuChannel

WorldUpAxis: TypeAlias = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
"""Signed axis of a tracking world that gravity points *away* from."""
POSITIVE_WORLD_AXES: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = ("+x", "+y", "+z")
"""Axis names by column index, for a positive mean; the negative row is below."""
NEGATIVE_WORLD_AXES: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = ("-x", "-y", "-z")
"""Axis names by column index, for a negative mean."""
STANDARD_GRAVITY_MS2: float = 9.80665
"""Standard gravity; ``measured_world_up`` reports its result as a fraction of this."""
MEASURED_UP_WINDOW_NS: int = 2_000_000_000
"""How much of a sequence's start ``measured_world_up`` averages over: a wearer has
usually not started moving yet, so the mean there is nearly pure gravity and gets
noisier the longer the window."""

WORLD_UP_VIEW_COORDINATES: dict[WorldUpAxis, rr.components.ViewCoordinates] = {
    "+x": rr.ViewCoordinates.RIGHT_HAND_X_UP,
    "-x": rr.ViewCoordinates.RIGHT_HAND_X_DOWN,
    "+y": rr.ViewCoordinates.RIGHT_HAND_Y_UP,
    "-y": rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
    "+z": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    "-z": rr.ViewCoordinates.RIGHT_HAND_Z_DOWN,
}
"""Root ``ViewCoordinates`` per world up axis, right-handed throughout.

Rerun's ``RIGHT_HAND_*`` aliases are exactly this table (``RIGHT_HAND_Y_UP`` is
``RUB``, ``RIGHT_HAND_Z_UP`` is ``RFU``), so naming the up axis is the whole
decision — the remaining two axes are then fixed by handedness.
"""


@dataclass(frozen=True, slots=True)
class MeasuredUp:
    """What one sequence's own gravity measurement found.

    A dataclass and not a ``NamedTuple``: beartype resolves a NamedTuple's field
    annotations as forward references, which a ``Literal`` alias is not.
    """

    axis: WorldUpAxis
    """The dominant signed world axis the mean upward acceleration points along."""
    fraction: float
    """That component as a fraction of standard gravity. A health check, not a
    calibration: near 1 means the window really was near rest and the axis is
    unambiguous, much less means the mean is not gravity."""


def measured_world_up(
    pose_times_ns: Int64[ndarray, "n_poses"],
    world_quaternions_xyzw: Float64[ndarray, "n_poses 4"],
    accel: ImuChannel,
    *,
    window_ns: int = MEASURED_UP_WINDOW_NS,
) -> MeasuredUp:
    """Measure which world axis is up, from gravity as the rig's accelerometer sees it.

    Each accelerometer sample inside the window is paired with the nearest ground-truth
    pose, rotated into the world with that pose's ``world_R_rig``, and the samples are
    averaged; the dominant signed axis of the mean is the answer.

    Args:
        pose_times_ns: Ground-truth pose times, ascending, on the accelerometer's clock.
        world_quaternions_xyzw: ``world_R_rig`` per pose, scalar last.
        accel: Accelerometer samples in m/s^2, in the rig frame, on the same clock.
        window_ns: Length of the averaging window, from the first time both streams cover.

    Returns:
        The axis and how much of |g| it carried.

    Raises:
        ValueError: Either stream is empty, or they do not overlap inside the window.
    """
    if pose_times_ns.size == 0 or accel.times_ns.size == 0:
        raise ValueError("measuring the world up axis needs both a gt pose and an accelerometer sample")
    start_ns: int = max(int(pose_times_ns[0]), int(accel.times_ns[0]))
    inside: Bool[ndarray, "n_samples"] = (accel.times_ns >= start_ns) & (accel.times_ns < start_ns + window_ns)
    if not inside.any():
        raise ValueError(f"no accelerometer sample within {window_ns / 1e9:g} s of {start_ns}, where the gt starts")

    window_times_ns: Int64[ndarray, "n_window"] = accel.times_ns[inside]
    after: Int64[ndarray, "n_window"] = np.clip(np.searchsorted(pose_times_ns, window_times_ns), 0, pose_times_ns.size - 1)
    before: Int64[ndarray, "n_window"] = np.clip(after - 1, 0, pose_times_ns.size - 1)
    nearest: Int64[ndarray, "n_window"] = np.where(
        np.abs(pose_times_ns[before] - window_times_ns) <= np.abs(pose_times_ns[after] - window_times_ns), before, after
    )
    world_accel_xyz: Float64[ndarray, "n_window 3"] = Rotation.from_quat(world_quaternions_xyzw[nearest]).apply(accel.values_xyz[inside])
    mean_xyz: Float64[ndarray, "3"] = world_accel_xyz.mean(axis=0)

    axis_index: int = int(np.argmax(np.abs(mean_xyz)))
    names: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = POSITIVE_WORLD_AXES if mean_xyz[axis_index] >= 0.0 else NEGATIVE_WORLD_AXES
    return MeasuredUp(axis=names[axis_index], fraction=float(abs(mean_xyz[axis_index]) / STANDARD_GRAVITY_MS2))
