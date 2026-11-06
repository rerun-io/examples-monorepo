"""Register MetricTrail layer files on a dataforge VSLAM-LAB dataset (SPEC section 6; thin, the files are the data).

The dataset and its segments must exist (dataforge registers ``base`` and ``gt`` first); this never creates a
dataset. First registration uses ``ERROR`` so it cannot silently replace a layer; a newer run of the same layer is a
new file registered with ``REPLACE``. After every registration on the shared ``:51235`` catalog, take the Spark
snapshot (SPEC 2.7), or a restart loses it.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import pyarrow as pa
import rerun as rr
from rerun.catalog import OnDuplicateSegmentLayer


@runtime_checkable
class Waitable(Protocol):
    """What a registration call returns."""

    def wait(self) -> object: ...


@runtime_checkable
class SegmentTable(Protocol):
    """``DatasetEntry.segment_table()``'s result, as far as the checks read it."""

    def to_arrow_table(self) -> pa.Table: ...


@runtime_checkable
class LayerDataset(Protocol):
    """The part of ``rerun.catalog.DatasetEntry`` registration uses."""

    def register(
        self, recording_uri: list[str], *, layer_name: str | Sequence[str] = ..., on_duplicate: OnDuplicateSegmentLayer = ...
    ) -> Waitable: ...

    def segment_ids(self) -> list[str]: ...

    def segment_table(self) -> SegmentTable: ...


def dataset_name(dataset: str) -> str:
    """The catalog dataset of a VSLAM-LAB dataset: ``dataforge-vslam-lab-<dataset>``."""
    return f"dataforge-vslam-lab-{dataset}"


def layer_uris(paths: Sequence[Path]) -> list[str]:
    """``file://`` URIs of layer files (absolute; the server reads them from its own filesystem).

    Raises:
        FileNotFoundError: If a file is missing.
    """
    missing: list[Path] = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"layer files not found: {missing}")
    return [path.resolve().as_uri() for path in paths]


def register_layers(dataset: LayerDataset, paths: Sequence[Path], segments: Sequence[str], layer_name: str, *, replace: bool) -> None:
    """Register ``paths`` (one per segment, its recording id) under ``layer_name`` in one call and check the result.

    Raises:
        ValueError: If a segment is not in the dataset (its base must come first), or after registration a segment
            lacks ``base``/``gt``/``layer_name`` or does not point at the new file (SPEC check 6.2.1).
    """
    known: set[str] = set(dataset.segment_ids())
    absent: list[str] = [segment for segment in segments if segment not in known]
    if absent:
        raise ValueError(f"segments {absent} are not in the dataset; register dataforge's base and gt layers first")
    uris: list[str] = layer_uris(paths)
    on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE if replace else OnDuplicateSegmentLayer.ERROR
    dataset.register(uris, layer_name=layer_name, on_duplicate=on_duplicate).wait()
    rows: list[dict[str, object]] = dataset.segment_table().to_arrow_table().to_pylist()
    by_segment: dict[str, dict[str, object]] = {str(row["rerun_segment_id"]): row for row in rows}
    for segment, uri in zip(segments, uris, strict=True):
        row: dict[str, object] | None = by_segment.get(segment)
        layers: list[str] = list(row["rerun_layer_names"]) if row is not None else []  # pyrefly: ignore[bad-argument-type]
        urls: list[str] = list(row["rerun_storage_urls"]) if row is not None else []  # pyrefly: ignore[bad-argument-type]
        if not {"base", "gt", layer_name} <= set(layers) or dict(zip(layers, urls, strict=True)).get(layer_name) != uri:
            raise ValueError(f"{segment}: after registration the layers are {dict(zip(layers, urls, strict=True))}, expected {layer_name} -> {uri}")


def open_dataset(catalog_url: str, name: str) -> rr.catalog.DatasetEntry:
    """The dataset ``name`` on ``catalog_url``."""
    return rr.catalog.CatalogClient(catalog_url).get_dataset(name)
