"""Shipped action annotations as half-open intervals, logged as active-label tracks.

At every interval boundary a track holds the labels active there. A text track joins them with
newlines, and an empty document clears the labels whose intervals ended. A state track
(``StateChange``, shown by a ``StateTimelineView``) gives each label a lane for its whole interval,
and an empty state ends it.
"""

from dataclasses import dataclass

import numpy as np
import rerun as rr
from jaxtyping import Int64
from numpy import ndarray

from dataforge import schema
from dataforge.logging_toolkit import frame_index_column, time_column


@dataclass(frozen=True, slots=True, order=True)
class Segment:
    """Half-open action interval on the dataset's annotation frame clock."""

    start: int
    """First active annotation frame."""
    end: int
    """First inactive annotation frame."""
    text: str
    """Label as shipped."""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid action interval")


def interval_boundaries(segments: list[Segment], *, stop: int | None = None) -> list[int]:
    """Frame 0 and every segment start and end below ``stop``, ascending."""
    boundaries: list[int] = sorted({0, *(boundary for segment in segments for boundary in (segment.start, segment.end))})
    return boundaries if stop is None else [boundary for boundary in boundaries if boundary < stop]


def active_labels(segments: list[Segment], *, stop: int | None = None) -> tuple[Int64[ndarray, "k"], list[str]]:
    """Frame 0 and every segment boundary below ``stop``, each with the labels active there in segment order."""
    boundaries: list[int] = interval_boundaries(segments, stop=stop)
    ordered: list[Segment] = sorted(segments)
    return (
        np.array(boundaries, dtype=np.int64),
        ["\n".join(segment.text for segment in ordered if segment.start <= boundary < segment.end) for boundary in boundaries],
    )


def active_states(segments: list[Segment]) -> tuple[Int64[ndarray, "k"], list[list[str]]]:
    """The boundaries of ``active_labels``, each with one state per lane; ``""`` is an idle lane.

    In segment order, each segment takes the lowest lane that is free at its start and keeps it for its
    whole interval, so overlapping labels sit in separate lanes. Every row has the same number of lanes
    (at least one), as ``StateChange`` asks.
    """
    boundaries: list[int] = interval_boundaries(segments)
    lane_ends: list[int] = []
    placed: list[tuple[int, Segment]] = []
    for segment in sorted(segment for segment in segments if segment.end > segment.start):
        lane: int = next((index for index, end in enumerate(lane_ends) if end <= segment.start), len(lane_ends))
        if lane == len(lane_ends):
            lane_ends.append(segment.end)
        else:
            lane_ends[lane] = segment.end
        placed.append((lane, segment))
    states: list[list[str]] = []
    for boundary in boundaries:
        row: list[str] = [""] * max(len(lane_ends), 1)
        for lane, segment in placed:
            if segment.start <= boundary < segment.end:
                row[lane] = segment.text
        states.append(row)
    return np.array(boundaries, dtype=np.int64), states


def log_actions(
    recording: rr.RecordingStream, level: str, *, times_ns: Int64[ndarray, "k"], frame_indices: Int64[ndarray, "k"], texts: list[str]
) -> None:
    """One columnar text document per boundary at ``schema.actions_path(level)``."""
    rr.send_columns(
        schema.actions_path(level),
        indexes=[time_column(times_ns), frame_index_column(frame_indices)],
        columns=rr.TextDocument.columns(text=texts),
        recording=recording,
    )


def log_action_states(
    recording: rr.RecordingStream, level: str, *, times_ns: Int64[ndarray, "k"], frame_indices: Int64[ndarray, "k"], states: list[list[str]]
) -> None:
    """One columnar ``StateChange`` per boundary at ``schema.actions_path(level)``, one instance per lane."""
    rr.send_columns(
        schema.actions_path(level),
        indexes=[time_column(times_ns), frame_index_column(frame_indices)],
        columns=rr.StateChange.columns(state=[state for row in states for state in row]).partition([len(row) for row in states]),
        recording=recording,
    )
