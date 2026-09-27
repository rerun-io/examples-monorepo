"""Shipped action annotations as half-open intervals, logged as active-label text tracks.

At every interval boundary the track holds all labels active there, joined by
newlines; an empty document clears the labels whose intervals ended.
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


def active_labels(segments: list[Segment], *, stop: int | None = None) -> tuple[Int64[ndarray, "k"], list[str]]:
    """Frame 0 and every segment boundary below ``stop``, each with the labels active there in segment order."""
    boundaries: list[int] = sorted({0, *(boundary for segment in segments for boundary in (segment.start, segment.end))})
    if stop is not None:
        boundaries = [boundary for boundary in boundaries if boundary < stop]
    ordered: list[Segment] = sorted(segments)
    return (
        np.array(boundaries, dtype=np.int64),
        ["\n".join(segment.text for segment in ordered if segment.start <= boundary < segment.end) for boundary in boundaries],
    )


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
