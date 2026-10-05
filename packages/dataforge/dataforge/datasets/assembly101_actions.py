"""Official 30 Hz annotations, unioned across views with overlap preserved."""

import csv
from dataclasses import dataclass
from pathlib import Path

from serde import SerdeError, coerce, from_dict, serde

from dataforge.actions import Segment


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


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class CoarseRow:
    """One headerless row of an official coarse-label TSV; columns past the third are ignored."""

    start_frame: int
    """Start at 30 Hz."""
    end_frame: int
    """End at 30 Hz."""
    label: str
    """Shipped coarse action text."""


COARSE_COLUMNS: tuple[str, ...] = ("start_frame", "end_frame", "label")


@dataclass(frozen=True, slots=True)
class Actions:
    """Both official annotation granularities for one sequence."""

    coarse: list[Segment]
    """Assembly/disassembly segments."""
    fine: list[Segment]
    """View-unioned fine action segments."""


def read_actions(root: Path, sequences: set[str]) -> dict[str, Actions]:
    """Scan each official CSV once for the selected sequences; absent annotations are allowed."""
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
    result: dict[str, Actions] = {}
    for sequence in sorted(sequences):
        coarse: list[Segment] = []
        for part in ("assembly", "disassembly"):
            path = root / "coarse-annotations/coarse_labels" / f"{part}_{sequence}.txt"
            if not path.is_file():
                continue
            with path.open(newline="") as handle:
                try:
                    rows: list[CoarseRow] = [from_dict(CoarseRow, dict(zip(COARSE_COLUMNS, fields, strict=False))) for fields in csv.reader(handle, delimiter="\t") if fields]
                except (SerdeError, ValueError) as error:
                    raise ValueError(f"{path}: {error}") from error
            coarse.extend(Segment(row.start_frame, row.end_frame, f"{part}: {row.label}") for row in rows)
        result[sequence] = Actions(sorted(coarse), sorted(unique[sequence]))
    return result

