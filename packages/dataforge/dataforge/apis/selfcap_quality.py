"""Rank catalog SelfCap segments in a replaceable summary property layer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import rerun as rr
from datafusion import col
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from serde import serde

from dataforge import records, writing


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class QualityInput:
    """One catalog segment's current quality signals."""

    segment_id: str
    """Catalog segment identity, shared by every layer."""
    refined: bool | None
    """Whether the deployed exo calibration is refined."""
    soma_status: str | None
    """SOMA quality decision, with pass preferred over all other states."""
    body_mm: float | None
    """Median body residual in millimetres."""
    shoulders_mm: float | None
    """Median shoulder residual in millimetres."""


def rank_segments(rows: list[QualityInput]) -> dict[str, int]:
    """Return ordinal ranks, best = 1, using refinement, pass, mean residual, then ID.

    Both residuals must be finite and nonnegative. Missing or invalid values sort
    last within the same refinement/pass tier. Ties use segment ID. Ranks describe
    this catalog snapshot; they are not absolute scores or cross-dataset measures.
    """
    def key(row: QualityInput) -> tuple[bool, bool, float, str]:
        residual: float = math.inf
        if (row.body_mm is not None and row.shoulders_mm is not None
                and all(math.isfinite(value) and value >= 0.0 for value in (row.body_mm, row.shoulders_mm))):
            residual = row.body_mm / 2.0 + row.shoulders_mm / 2.0
        return row.refined is not True, row.soma_status != "pass", residual, row.segment_id

    return {row.segment_id: rank for rank, row in enumerate(sorted(rows, key=key), start=1)}


def publish_summary(entry: DatasetEntry, output_root: Path) -> None:
    """Read one metadata snapshot and replace only its summary layer with fresh files.

    Registered files are never overwritten. Each run keeps its stamped directory,
    so two catalogs can share the same files without invalidating either server.
    """
    columns: dict[str, str] = {
        "refined": "property:calib:calib_exo_refined",
        "soma_status": "property:soma_quality:status",
        "body_mm": "property:soma_quality:body_median_mm",
        "shoulders_mm": "property:soma_quality:shoulders_median_mm",
    }
    rows: list[QualityInput] = [
        records.decode_values(QualityInput, row, source="SelfCap catalog quality snapshot")
        for row in entry.segment_table().select(
            col("rerun_segment_id").alias("segment_id"),
            *(col(column)[0].alias(name) for name, column in columns.items()),
        ).to_arrow_table().to_pylist()
    ]
    ranks: dict[str, int] = rank_segments(rows)
    stamp: str = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    targets: list[Path] = []
    for segment_id, rank in ranks.items():
        target: Path = (output_root / "summary" / stamp / f"{rr.escape_entity_path_part(segment_id)}.rrd").resolve()
        chunk: Chunk = Chunk.from_property("summary", rr.AnyValues(quality_rank=rank, formula_version="selfcap-quality-rank-v1", computed_at=stamp))
        with writing.atomic_write(target) as temporary:
            LazyChunkStream.from_iter([chunk]).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
                temporary, application_id=writing.APPLICATION_ID, recording_id=segment_id,
            )
        targets.append(target)
    if targets:
        entry.register([target.as_uri() for target in targets], layer_name="summary", on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    print(f"Published {len(targets)} summary ranks (1 = best), snapshot {stamp}")
