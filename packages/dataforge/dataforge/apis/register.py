"""``dataforge-register``: register a dataset's rrd layers into a local Rerun catalog.

One catalog dataset holds every layer under ``output_root()``, keyed by the
directory it sits in: ``base`` is the sensor recording each converter writes and
is required. Each dataset declares its derived layers, which share the base
recording identity. A declared layer with no files is not registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow as pa
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from dataforge import paths, schema, writing
from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig


@dataclass
class Config:
    """Register a dataset's converted recordings into a catalog."""

    dataset: AnnotatedDatasetUnion = field(default_factory=RobocapConfig)
    """Dataset whose rrds get registered; its registry key is the catalog dataset name."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """gRPC URL of a locally running ``rerun server`` catalog."""
    replace: bool = False
    """Re-register a layer the catalog already holds for a segment, instead of skipping it.

    Registration is normally ``SKIP``, so re-running it over a corpus is cheap
    and idempotent. But a regenerated layer — ``rm gt/*.rrd`` and a convert, per
    README#the-layer-rule — is a *new file at a registered path*, and ``SKIP``
    leaves the server serving the old registration: the rebuilt rrd stays
    unregistered and nothing says so. ``--replace`` is what to use after a
    rebuild."""
    refresh_blueprints: bool = False
    """Replace the registered default and table blueprints with ones written from the current code.

    Writes new dated ``.rbl`` files beside the old ones, makes them the defaults, then unregisters
    every other blueprint entry of the dataset and deletes those entries' files when they sit in
    this dataset's blueprint directory. The dataset id and its segments are unchanged, so links
    keep working. Also run it after registering a new layer: the table blueprint hides undeclared
    columns by name, so columns the new layer brings stay visible until the next refresh."""


def main(config: Config) -> None:
    """Create the dataset if needed and register every layer's rrds (idempotent)."""
    dataset_config: DataforgeDatasetConfig = config.dataset
    dataset: DataforgeDataset = dataset_config.setup()
    name: str = dataset_config.name
    output_root: Path = paths.output_root()
    paths_by_layer: dict[str, list[Path]] = {layer: sorted((output_root / layer).glob(f"{name}__*.rrd")) for layer in dataset.layers}
    if not paths_by_layer[paths.BASE_LAYER]:
        raise FileNotFoundError(f"no {paths.BASE_LAYER}-layer rrds for {name} under {output_root / paths.BASE_LAYER}")

    client: CatalogClient = CatalogClient(config.catalog_url)
    entry: DatasetEntry = client.create_dataset(name, exist_ok=True)
    on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE if config.replace else OnDuplicateSegmentLayer.SKIP
    for layer, rrd_paths in paths_by_layer.items():
        if not rrd_paths:
            continue  # a derived layer nobody has produced yet
        entry.register([path.resolve().as_uri() for path in rrd_paths], layer_name=layer, on_duplicate=on_duplicate).wait()

    # Blueprints register once: every register_blueprint call adds a NEW entry to the
    # catalog dataset's blueprint list (cluttering the viewer's selector), so an
    # incremental re-register skips a blueprint the catalog already has a default for,
    # and --refresh-blueprints retires the entries its new files replace.
    refresh: bool = config.refresh_blueprints
    stamp: str | None = datetime.now().strftime("%Y%m%d-%H%M%S") if refresh else None
    written: set[Path] = set()
    if refresh or entry.default_blueprint() is None:
        blueprint_path: Path = paths.blueprint_path(output_root, name, stamp=stamp).resolve()
        with writing.atomic_write(blueprint_path) as temp_path:
            dataset.default_blueprint().save(name, str(temp_path))
        entry.register_blueprint(blueprint_path.as_uri(), set_default=True)
        written.add(blueprint_path)
    if refresh or entry.default_segment_table_blueprint() is None:
        table_path: Path = paths.blueprint_path(output_root, name, segment_table=True, stamp=stamp).resolve()
        columns: list[str] = entry.segment_table().schema().names
        writing.save_table_blueprint(dataset.table_blueprint(), table_path, timeline=schema.TIMELINE, fields=dataset.table_fields(), columns=columns)
        entry.register_blueprint(table_path.as_uri(), segment_table=True)
        written.add(table_path)
    # Every blueprint entry lives in the dataset's hidden blueprint dataset, one segment per .rbl.
    blueprint_entries: DatasetEntry | None = entry.blueprint_dataset() if refresh else None
    if blueprint_entries is not None:
        listed: pa.Table = blueprint_entries.segment_table().select("rerun_segment_id", "rerun_storage_urls").to_arrow_table()
        files: dict[str, Path] = {
            segment: Path(unquote(urlparse(urls[0]).path))
            for segment, urls in zip(listed["rerun_segment_id"].to_pylist(), listed["rerun_storage_urls"].to_pylist(), strict=True)
        }
        stale: dict[str, Path] = {segment: path for segment, path in files.items() if path not in written}
        if stale:
            blueprint_entries.unregister(segments_to_drop=list(stale), layers_to_drop=[]).wait()
            blueprint_dir: Path = paths.blueprint_path(output_root, name).parent.resolve()  # registered URLs are resolved paths
            ours: list[Path] = [path for path in stale.values() if path.parent == blueprint_dir]
            for path in ours:
                path.unlink(missing_ok=True)
            print(f"retired {len(stale)} older blueprint entries of '{name}' and deleted {len(ours)} of their files")
    counted: str = ", ".join(f"{len(found)} {layer}" for layer, found in paths_by_layer.items() if found)
    how: str = "replacing duplicates" if config.replace else "skipping duplicates"
    print(f"registered {counted} rrds into '{name}' at {config.catalog_url} ({how})")
