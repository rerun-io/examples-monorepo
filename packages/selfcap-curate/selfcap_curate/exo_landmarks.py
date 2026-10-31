"""Typed optical landmarks, stored once as a visible catalog layer."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from dataforge.logging_toolkit import time_column
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from serde import serde
from serde.json import from_json, to_json

from selfcap_curate.catalog_io import scalar


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Landmark:
    view: int
    """Observing exo rig index."""
    video_time: float
    """Aligned video time in seconds."""
    target_id: str
    """quest0/quest1 optical centres or phoneN exo centre."""
    u: float
    """Horizontal source pixel coordinate."""
    v: float
    """Vertical source pixel coordinate."""
    confidence: float
    """Label confidence in (0, 1]."""

    def __post_init__(self) -> None:
        if self.view < 0 or not all(math.isfinite(x) for x in (self.video_time, self.u, self.v, self.confidence)):
            raise ValueError("landmark coordinates and time must be finite; view must be nonnegative")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("landmark confidence must be in (0, 1]")
        if self.target_id not in ("quest0", "quest1") and not (self.target_id.startswith("phone") and self.target_id[5:].isdigit()):
            raise ValueError(f"unknown optical target: {self.target_id}")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Landmarks:
    labels: list[Landmark]
    """Optical aperture observations."""
    labeller: str
    """Agent model or human identifier; unknown historical models stay explicit."""
    method: str
    """Annotation procedure and uncertainty."""
    date: str
    """Source annotation date, supplied explicitly for reproducibility."""

    @property
    def digest(self) -> str:
        """Stable hash of the complete source record, including provenance."""
        return hashlib.sha256(to_json(self).encode()).hexdigest()


def write_landmarks(path: Path, segment: str, record: Landmarks) -> None:
    """Write a JSON sidecar record and its pixel annotations as one catalog layer."""
    if path.exists():
        raise FileExistsError(path)
    chunks: list[Chunk] = [Chunk.from_property("exo_landmarks", rr.AnyValues(record_json=to_json(record), sha256=record.digest))]
    for view in sorted({label.view for label in record.labels}):
        labels = [label for label in record.labels if label.view == view]
        stamps = sorted({label.video_time for label in labels})
        ordered = [label for stamp in stamps for label in labels if label.video_time == stamp]
        counts = [sum(label.video_time == stamp for label in labels) for stamp in stamps]
        chunks.append(Chunk.from_columns(
            f"/world/rig_{view:02d}/cam_00/pinhole/exo_landmarks",
            indexes=[time_column(np.rint(np.asarray(stamps) * 1e9).astype(np.int64))],
            columns=[
                *rr.Points2D.columns(positions=[[label.u, label.v] for label in ordered], labels=[label.target_id for label in ordered],
                                     radii=[2.5] * len(ordered), colors=[[255, 220, 60]] * len(ordered), show_labels=[True] * len(ordered)).partition(counts),
                *rr.AnyValues.columns(confidence=np.asarray([label.confidence for label in ordered], dtype=np.float64)).partition(counts),
            ],
        ))
    path.parent.mkdir(parents=True, exist_ok=True)
    LazyChunkStream.from_iter(chunks).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        str(path), application_id="selfcap-exo-landmarks", recording_id=segment)


def read_landmarks(dataset: DatasetEntry, segment: str) -> Landmarks | None:
    """Read the annotation record from the catalog; absent layers return None."""
    table = dataset.filter_segments(segment).filter_contents("/__properties/exo_landmarks").reader(index=None).to_arrow_table()
    column = next((name for name in table.column_names if name.endswith(":exo_landmarks:record_json")), None)
    if column is None or not len(table[column].drop_null()):
        return None
    return from_json(Landmarks, str(scalar(table, column)))
