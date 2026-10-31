"""Read-only catalog inventory and restoration from server-visible storage URIs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer, TableEntry
from serde import serde
from serde.json import to_json

from dataforge import records, writing


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LayerRef:
    """One registered segment/layer and its original storage URI."""

    rerun_segment_id: str
    """Original segment identity."""
    rerun_layer_name: str
    """Layer name to retain on restoration."""
    rerun_storage_url: str
    """Existing server-visible file; never copied or rewritten."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BlueprintRef:
    """A blueprint file and its default roles."""

    uri: str
    """Server-visible .rbl file."""
    default: bool
    """Whether this is the default recording blueprint."""
    segment_table: bool
    """Whether this is the default segment-table blueprint."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EntrySnapshot:
    """A dataset or table and all its external references."""

    name: str
    """Catalog entry name to recreate."""
    original_id: str
    """Original entry ID; new catalogs assign new entry IDs."""
    layers: list[LayerRef]
    """All registered segment/layer files for a dataset."""
    blueprints: list[BlueprintRef]
    """Registered blueprints with default roles."""
    assets: list[str]
    """Registered asset URIs for a dataset."""
    table_url: str | None
    """Table storage URL, or None for a dataset."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """Restorable references, not copies of the recording data."""

    source_url: str
    """Read-only source catalog."""
    entries: list[EntrySnapshot]
    """Every visible dataset and table, including owned blueprints/assets."""


def blueprint_refs(entry: DatasetEntry | TableEntry) -> list[BlueprintRef]:
    """Read blueprint URIs and distinguish the two default roles."""
    dataset: DatasetEntry | None = entry.blueprint_dataset()
    if dataset is None:
        return []
    default: str | None = entry.default_blueprint()
    table_default: str | None = entry.default_segment_table_blueprint() if isinstance(entry, DatasetEntry) else None
    return [
        BlueprintRef(row["rerun_storage_url"], row["rerun_segment_id"] == default, row["rerun_segment_id"] == table_default)
        for row in dataset.manifest().select("rerun_segment_id", "rerun_storage_url").to_arrow_table().to_pylist()
    ]


def layer_refs(dataset: DatasetEntry, source: str) -> list[LayerRef]:
    """Decode a dataset manifest into its registered segment/layer files."""
    return [
        records.decode_values(LayerRef, row, source=source)
        for row in dataset.manifest().select("rerun_segment_id", "rerun_layer_name", "rerun_storage_url").to_arrow_table().to_pylist()
    ]


def snapshot_catalog(catalog_url: str, target: Path) -> None:
    """Save every visible entry and its layers, assets, blueprints, or table URL."""
    client: CatalogClient = CatalogClient(catalog_url)
    entries: list[EntrySnapshot] = []
    for dataset in client.datasets():
        layers: list[LayerRef] = layer_refs(dataset, f"{dataset.name} manifest")
        asset_dataset: DatasetEntry | None = dataset.asset_dataset()
        assets: list[str] = [] if asset_dataset is None else asset_dataset.manifest().select("rerun_storage_url").to_arrow_table().column(0).to_pylist()
        entries.append(EntrySnapshot(dataset.name, str(dataset.id), layers, blueprint_refs(dataset), assets, None))
        print(f"Snapshot {dataset.name}: {len(layers)} segment/layers", flush=True)
    for table in client.tables():
        entries.append(EntrySnapshot(table.name, str(table.id), [], blueprint_refs(table), [], table.storage_url))
    with writing.atomic_write(target) as temporary:
        temporary.write_text(to_json(CatalogSnapshot(catalog_url, entries)))
    print(f"Saved {len(entries)} entries to {target}")


def restore_catalog(snapshot_path: Path, catalog_url: str, names: tuple[str, ...] = (), *, allow_source_url: bool = False) -> None:
    """Restore selected or all entries on another server, retaining existing files.

    Re-running replaces changed segment/layers on the destination only, skipping
    identical URI registrations. Dataset
    entry IDs change, but names and segment IDs are preserved. Source must differ
    unless allow_source_url explicitly permits recovery of a restarted server.
    """
    snapshot: CatalogSnapshot = records.decode(CatalogSnapshot, snapshot_path.read_text(), source=str(snapshot_path))
    if not allow_source_url and catalog_url.rstrip("/") == snapshot.source_url.rstrip("/"):
        raise ValueError("Restore target must differ from the snapshot source")
    missing: set[str] = set(names) - {entry.name for entry in snapshot.entries}
    if missing:
        raise ValueError(f"Unknown snapshot entries: {sorted(missing)}")
    client: CatalogClient = CatalogClient(catalog_url)
    for entry in snapshot.entries:
        if names and entry.name not in names:
            continue
        started: float = perf_counter()
        target: DatasetEntry | TableEntry
        if entry.table_url is not None:
            target = client.register_table(entry.name, entry.table_url)
        else:
            target = client.create_dataset(entry.name, exist_ok=True)
            registered: set[LayerRef] = set(layer_refs(target, f"{entry.name} destination manifest"))
            by_layer: defaultdict[str, list[str]] = defaultdict(list)
            for layer in entry.layers:
                if layer not in registered:
                    by_layer[layer.rerun_layer_name].append(layer.rerun_storage_url)
            for layer_name, uris in by_layer.items():
                target.register(uris, layer_name=layer_name, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
            for uri in entry.assets:
                target.register_asset(uri)
        existing_blueprints: list[BlueprintRef] = blueprint_refs(target)
        for blueprint in entry.blueprints:
            if blueprint in existing_blueprints:
                continue
            if isinstance(target, DatasetEntry):
                target.register_blueprint(blueprint.uri, set_default=blueprint.default or blueprint.segment_table, segment_table=blueprint.segment_table)
                if blueprint.default and blueprint.segment_table:
                    target.set_default_blueprint(target.default_segment_table_blueprint())
            else:
                target.register_blueprint(blueprint.uri, set_default=blueprint.default)
        print(f"Restored {entry.name}: {len(entry.layers)} segment/layers in {perf_counter() - started:.3f}s", flush=True)
