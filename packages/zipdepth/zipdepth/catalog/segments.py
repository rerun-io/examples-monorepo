"""PromptDA catalog connection and segment metadata."""

from dataclasses import dataclass
from random import Random
from typing import Literal, TypeAlias

import pyarrow as pa
from rerun.catalog import CatalogClient, DatasetEntry

# Matches arkitscenes_download.ingest.catalog.DEFAULT_CATALOG_URL. Keeping the
# literal here lets pure policy tests import this module without catalog-only deps.
DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
DEFAULT_DATASET_NAME: str = "arkitscenes-v2"
PROMPTDA_LAYER: str = "promptda"
"""Catalog layer containing the chosen-frame pseudo-depth targets."""

_ROW_FIELDS: tuple[str, ...] = (
    "rerun_segment_id",
    "property:capture:orientation",
    "property:capture:orientation_quarter_turns_ccw",
)
_SELECTION_FIELDS: tuple[str, ...] = (*_ROW_FIELDS, "rerun_layer_names")
CaptureOrientation: TypeAlias = Literal["portrait", "landscape"]
CatalogCell: TypeAlias = str | int | list[str] | list[int] | None


@dataclass(frozen=True, slots=True)
class SegmentRow:
    """Typed catalog metadata consumed by the PromptDA dataset."""

    id: str
    """Rerun segment identifier."""
    orientation: CaptureOrientation
    """Stored capture orientation."""
    orientation_quarter_turns_ccw: int
    """Counter-clockwise turns used when baking the stored orientation."""
    layer_names: tuple[str, ...]
    """Rerun layers registered for the segment."""


@dataclass(slots=True, frozen=True)
class PromptDACatalog:
    """Connected PromptDA catalog metadata shared by train, eval, and smoke."""

    dataset_entry: DatasetEntry
    """Connected Rerun catalog dataset."""
    row_by_id: dict[str, SegmentRow]
    """Required PromptDA segment metadata keyed by segment identifier."""
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
    segment_table: pa.Table = pa.Table.from_batches(dataset_entry.segment_table().select(*_SELECTION_FIELDS).collect())
    row_by_id: dict[str, SegmentRow] = {}
    rows: list[dict[str, CatalogCell]] = segment_table.to_pylist()
    row: dict[str, CatalogCell]
    for row in rows:
        raw_segment_id: CatalogCell = row.get("rerun_segment_id")
        if not isinstance(raw_segment_id, str):
            raise ValueError(f"catalog segment id must be a string, got {raw_segment_id!r}")

        raw_orientation: CatalogCell = row.get("property:capture:orientation")
        if isinstance(raw_orientation, list):
            if len(raw_orientation) != 1:
                raise ValueError(f"segment {raw_segment_id!r} orientation must contain one value")
            raw_orientation = raw_orientation[0]
        if raw_orientation not in ("portrait", "landscape"):
            raise ValueError(f"segment {raw_segment_id!r} has invalid orientation {raw_orientation!r}")
        orientation: CaptureOrientation = raw_orientation

        raw_quarter_turns: CatalogCell = row.get("property:capture:orientation_quarter_turns_ccw")
        if isinstance(raw_quarter_turns, list):
            if len(raw_quarter_turns) != 1:
                raise ValueError(f"segment {raw_segment_id!r} orientation quarter turns must contain one value")
            raw_quarter_turns = raw_quarter_turns[0]
        if not isinstance(raw_quarter_turns, int) or isinstance(raw_quarter_turns, bool):
            raise ValueError(f"segment {raw_segment_id!r} has invalid orientation quarter turns {raw_quarter_turns!r}")

        raw_layer_names: CatalogCell = row.get("rerun_layer_names")
        if raw_layer_names is None:
            layer_names: tuple[str, ...] = ()
        elif isinstance(raw_layer_names, list) and all(isinstance(layer_name, str) for layer_name in raw_layer_names):
            layer_names = tuple(layer_name for layer_name in raw_layer_names if isinstance(layer_name, str))
        else:
            raise ValueError(f"segment {raw_segment_id!r} has invalid layer names {raw_layer_names!r}")
        if PROMPTDA_LAYER in layer_names:
            row_by_id[raw_segment_id] = SegmentRow(
                id=raw_segment_id,
                orientation=orientation,
                orientation_quarter_turns_ccw=raw_quarter_turns,
                layer_names=layer_names,
            )
    segment_ids: list[str] = sorted(row_by_id)
    if not segment_ids:
        raise RuntimeError(f"dataset {dataset_name!r} has no segments with the {PROMPTDA_LAYER!r} layer")
    return PromptDACatalog(dataset_entry=dataset_entry, row_by_id=row_by_id, segment_ids=segment_ids)


def split_holdout_segments(segment_ids: list[str], holdout_count: int, seed: int) -> tuple[list[str], list[str]]:
    """Split sorted identifiers into deterministic disjoint train and holdout lists.

    Args:
        segment_ids: Unique segment identifiers in any input order.
        holdout_count: Number of identifiers to reserve.
        seed: Random seed used to select the holdout.

    Returns:
        The training identifiers followed by the holdout identifiers, both sorted.

    Raises:
        ValueError: If identifiers repeat or the holdout count is invalid.
    """
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("segment_ids must be unique")
    if not 0 <= holdout_count <= len(segment_ids):
        raise ValueError(f"holdout_count must be in [0, {len(segment_ids)}]")
    ordered_ids: list[str] = sorted(segment_ids)
    holdout_set: set[str] = set(Random(seed).sample(ordered_ids, holdout_count))
    train_ids: list[str] = [segment_id for segment_id in ordered_ids if segment_id not in holdout_set]
    holdout_ids: list[str] = [segment_id for segment_id in ordered_ids if segment_id in holdout_set]
    return train_ids, holdout_ids
