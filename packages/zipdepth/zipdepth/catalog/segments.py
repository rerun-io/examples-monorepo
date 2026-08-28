"""PromptDA catalog connection and segment metadata."""

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from rerun.catalog import CatalogClient, DatasetEntry

# Matches arkitscenes_download.ingest.catalog.DEFAULT_CATALOG_URL. Keeping the
# literal here lets pure policy tests import this module without catalog-only deps.
DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
PROMPTDA_LAYER: str = "promptda"
"""Catalog layer containing the chosen-frame pseudo-depth targets."""

_ROW_FIELDS: tuple[str, ...] = (
    "rerun_segment_id",
    "property:capture:orientation",
    "property:capture:orientation_quarter_turns_ccw",
)


@dataclass(slots=True, frozen=True)
class PromptDACatalog:
    """Connected PromptDA catalog metadata shared by train, eval, and smoke."""

    dataset_entry: DatasetEntry
    """Connected Rerun catalog dataset."""
    row_by_id: dict[str, dict[str, Any]]
    """Required PromptDA segment properties keyed by segment identifier."""
    segment_ids: list[str]
    """Sorted identifiers for every segment carrying the PromptDA layer."""

    def require_segments(self, segment_ids: list[str]) -> None:
        """Reject identifiers that are absent or lack the PromptDA layer."""
        missing_ids: list[str] = [segment_id for segment_id in segment_ids if segment_id not in self.row_by_id]
        if missing_ids:
            raise ValueError(f"segments are absent or lack the {PROMPTDA_LAYER!r} layer: {missing_ids}")


def load_promptda_catalog(catalog_url: str, dataset_name: str) -> PromptDACatalog:
    """Connect once and collect the PromptDA segment-selection metadata."""
    dataset_entry: DatasetEntry = CatalogClient(catalog_url).get_dataset(dataset_name)
    segment_table: pa.Table = pa.Table.from_batches(dataset_entry.segment_table().collect())
    row_by_id: dict[str, dict[str, Any]] = {}
    row: dict[str, Any]
    for row in segment_table.to_pylist():
        layer_names: list[str] = [str(layer_name) for layer_name in (row.get("rerun_layer_names") or [])]
        if PROMPTDA_LAYER in layer_names:
            segment_id: str = str(row["rerun_segment_id"])
            row_by_id[segment_id] = {field: row.get(field) for field in _ROW_FIELDS}
    segment_ids: list[str] = sorted(row_by_id)
    if not segment_ids:
        raise RuntimeError(f"dataset {dataset_name!r} has no segments with the {PROMPTDA_LAYER!r} layer")
    return PromptDACatalog(dataset_entry=dataset_entry, row_by_id=row_by_id, segment_ids=segment_ids)
