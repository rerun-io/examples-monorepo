"""Group several cameras' frames into framesets as a live capture adapter does (SPEC 5.6: the one home of frameset matching).

A frameset opens at the earliest frame not yet taken and holds, per camera, that camera's next frame if it lies within
``tolerance_ns`` of the opening frame; its time is the opening frame's. A camera without a frame there is absent from it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Int64
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class FramesetIndex:
    """Which frame of each camera belongs to which frameset."""

    t_ns: Int64[ndarray, " f"]
    """The opening (earliest) frame's time."""
    frame: Int64[ndarray, "f c"]
    """Frame number within its camera, -1 where the camera has no frame in the frameset."""
    cam_t_ns: Int64[ndarray, "f c"]
    """Each camera's frame time, 0 where absent."""


def group_framesets(camera_t_ns: Sequence[Int64[ndarray, " n"]], tolerance_ns: int) -> FramesetIndex:
    """Every frame lands in exactly one frameset, in each camera's order; openings strictly increase.

    Raises:
        ValueError: If the tolerance is negative or a camera's frame times are not strictly increasing.
    """
    if tolerance_ns < 0:
        raise ValueError(f"group tolerance {tolerance_ns} ns is negative")
    for camera, times in enumerate(camera_t_ns):
        if times.size and not bool(np.all(np.diff(times) > 0)):
            raise ValueError(f"camera {camera}: frame times are not strictly increasing")
    cameras: int = len(camera_t_ns)
    cursor: list[int] = [0] * cameras
    rows: list[tuple[int, list[int], list[int]]] = []
    while heads := [int(times[cursor[c]]) for c, times in enumerate(camera_t_ns) if cursor[c] < len(times)]:
        opening: int = min(heads)
        frames: list[int] = [-1] * cameras
        stamps: list[int] = [0] * cameras
        for c, times in enumerate(camera_t_ns):
            if cursor[c] < len(times) and int(times[cursor[c]]) - opening <= tolerance_ns:
                frames[c], stamps[c] = cursor[c], int(times[cursor[c]])
                cursor[c] += 1
        rows.append((opening, frames, stamps))
    return FramesetIndex(
        t_ns=np.asarray([row[0] for row in rows], dtype=np.int64),
        frame=np.asarray([row[1] for row in rows], dtype=np.int64).reshape(-1, cameras),
        cam_t_ns=np.asarray([row[2] for row in rows], dtype=np.int64).reshape(-1, cameras),
    )
