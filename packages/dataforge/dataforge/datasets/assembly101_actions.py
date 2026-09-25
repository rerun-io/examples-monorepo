"""Official 30 Hz annotations, unioned across views with overlap preserved."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from jaxtyping import Int64
from numpy import ndarray
from serde import coerce, from_dict, serde

from dataforge.datasets.assembly101_source import ANNOTATION_RATE, FRAME_RATE


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class FineRow:
    """Fields used from the official per-view CSV."""

    video: str
    """Sequence/view.mp4."""
    start_frame: int
    """Start at 30 Hz."""
    end_frame: int
    """End at 30 Hz."""
    action_cls: str
    """Shipped action text."""


@dataclass(frozen=True, slots=True, order=True)
class Segment:
    """Half-open official action interval at 30 Hz."""

    start: int
    """First active annotation frame."""
    end: int
    """First inactive annotation frame."""
    text: str
    """Action label, including the coarse assembly/disassembly part."""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid action interval")


Actions: TypeAlias = dict[str, list[Segment]]


def read_actions(root: Path, sequences: set[str]) -> dict[str, Actions]:
    """Scan each official CSV once for the selected sequences; absent annotations are allowed."""
    result: dict[str, Actions] = {sequence: {"coarse": [], "fine": []} for sequence in sequences}
    unique: dict[str, set[Segment]] = {sequence: set() for sequence in sequences}
    for split in ("train", "validation", "test"):
        path: Path = root / "fine-grained-annotations" / f"{split}.csv"
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                sequence: str = raw["video"].split("/")[0]
                if sequence in sequences:
                    row: FineRow = from_dict(FineRow, raw)
                    unique[sequence].add(Segment(row.start_frame, row.end_frame, row.action_cls))
    for sequence in sorted(sequences):
        result[sequence]["fine"] = sorted(unique[sequence])
        for part in ("assembly", "disassembly"):
            path = root / "coarse-annotations/coarse_labels" / f"{part}_{sequence}.txt"
            if not path.is_file():
                continue
            with path.open(newline="") as handle:
                for fields in csv.reader(handle, delimiter="\t"):
                    if fields:
                        result[sequence]["coarse"].append(Segment(int(fields[0]), int(fields[1]), f"{part}: {fields[2]}"))
        result[sequence]["coarse"].sort()
    return result


def action_rows(segments: list[Segment], frame_limit: int | None = None) -> tuple[Int64[ndarray, "n"], list[str]]:
    """All active labels at each boundary; an empty document clears ended actions."""
    boundaries: list[int] = sorted({boundary for segment in segments for boundary in (segment.start, segment.end)})
    ratio: int = FRAME_RATE // ANNOTATION_RATE
    if frame_limit is not None:
        boundaries = [boundary for boundary in boundaries if boundary * ratio < frame_limit]
    ordered: list[Segment] = sorted(segments)
    return (
        np.array(boundaries, dtype=np.int64) * ratio,
        ["\n".join(segment.text for segment in ordered if segment.start <= boundary < segment.end) for boundary in boundaries],
    )
