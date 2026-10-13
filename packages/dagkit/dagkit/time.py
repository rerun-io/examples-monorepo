"""Stream time: i64 nanoseconds, monotonic per output port, one clock per graph."""

from typing import Final, TypeAlias

TimeNs: TypeAlias = int
"""i64 nanoseconds on the graph's clock (a timeline name such as `video_time`)."""
STATIC: Final[TimeNs] = -(2**63)
"""Time of a value valid for the whole stream (rig, calibration, pose table); consumers read it through AsOf."""
TIME_MAX: Final[TimeNs] = 2**63 - 1
"""Watermark after flush: every output is final."""


class GraphError(ValueError):
    """A graph, an edge or a stream broke the dagkit contract."""


def check_monotonic(previous: TimeNs | None, t_ns: TimeNs, what: str) -> None:
    """Raise `GraphError` if `t_ns` goes back in time after `previous` (equal times are allowed)."""
    if previous is not None and t_ns < previous:
        raise GraphError(f"{what}: time went backwards ({t_ns} < {previous})")
