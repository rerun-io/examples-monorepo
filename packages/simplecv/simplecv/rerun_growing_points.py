"""A cloud that grows along a walk, under one points entity; colsfm writes it, dataforge's Insta360 layout draws it.

- ``<entity>/blocks/NNNN``: one entity per ``GROWING_BLOCK_S`` block of first views, with one row at the block's last
  first view. A layout draws the blocks as they are (latest-at).
- ``<entity>/recent``: one row per first-seen time. A layout draws it over the last ``GROWING_BLOCK_S`` of the cursor.

Writer and layout must agree on the block span, so it lives here: both depend on simplecv, not on each other.
"""

from typing import Final

import numpy as np
import rerun as rr
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

GROWING_BLOCK_S: Final[float] = 30.0
"""Span of one ``blocks/NNNN`` entity and window of the ``recent`` rows, seconds. Clip 007 (1,068 s) makes 36 blocks,
and the recent rows hold a few percent of the cloud."""


def log_points_by_first_view(
    recording: rr.RecordingStream,
    entity: str,
    xyz: Float64[ndarray, "p 3"],
    rgb: UInt8[ndarray, "p 3"],
    seen_ns: Int64[ndarray, " p"],
    timeline: str,
    *,
    block_ns: int = round(GROWING_BLOCK_S * 1e9),
) -> None:
    """Log the cloud by when each point is first observed, in blocks of `block_ns` from the first view.

    Each block is one entity, `entity/blocks/NNNN`, with one row: the points first seen in it, at the block's last first
    view, so latest-at shows it once the cursor has seen all of it. `entity/recent` gets one row per first-seen time (a
    rig frame). A layout that draws the blocks as they are and `entity/recent` over the last `block_ns` shows every point
    seen by the cursor and none after it: the block the cursor is in comes from the recent rows, which also redraw at
    most one finished block in place. A cloud without points logs nothing.

    Why blocks: the viewer (Rerun 0.42) makes a point batch per row and processes a query's rows again whenever its result
    changes. A range from the start to the cursor changes on every frame of playback, so one row per rig frame since the
    start (3,096 rows, 2.32 M points on clip 007) was processed in full on every frame. A block's latest-at result stays
    the same, so only the recent rows are processed again.

    Args:
        recording: The layer's recording.
        entity: The points entity; the cloud goes under it.
        xyz: Positions, metres.
        rgb: Colours.
        seen_ns: Per point, when it is first observed.
        timeline: The segment's video timeline.
        block_ns: Span of one block, nanoseconds of `timeline`.
    """
    if len(seen_ns) == 0:
        return
    order: Int64[ndarray, " p"] = np.argsort(seen_ns, kind="stable")
    seen_sorted: Int64[ndarray, " p"] = seen_ns[order]
    xyz_sorted: Float64[ndarray, "p 3"] = xyz[order]
    rgb_sorted: UInt8[ndarray, "p 3"] = rgb[order]
    recent: str = f"{entity}/recent"
    times, counts = np.unique(seen_sorted, return_counts=True)
    recording.send_columns(
        recent,
        indexes=[rr.TimeColumn(timeline, duration=times.astype("timedelta64[ns]"))],
        columns=rr.Points3D.columns(positions=xyz_sorted, colors=rgb_sorted).partition(counts.tolist()),
    )
    recording.log(recent, rr.Points3D.from_fields(radii=rr.Radius.ui_points(1.5)), static=True)
    # Sorted by first view, so each block is one slice; like the recent rows, sent as columns (no stream time set).
    block_ids, starts = np.unique((seen_sorted - seen_sorted[0]) // block_ns, return_index=True)
    ends: Int64[ndarray, " b"] = np.r_[starts[1:], len(seen_sorted)]
    for block_id, start, end in zip(block_ids.tolist(), starts.tolist(), ends.tolist(), strict=True):
        recording.send_columns(
            f"{entity}/blocks/{block_id:04d}",
            indexes=[rr.TimeColumn(timeline, duration=seen_sorted[end - 1 : end].astype("timedelta64[ns]"))],
            columns=rr.Points3D.columns(positions=xyz_sorted[start:end], colors=rgb_sorted[start:end]).partition([end - start]),
        )
