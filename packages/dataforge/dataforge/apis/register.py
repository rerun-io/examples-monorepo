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
from socket import gethostname
from typing import Any
from urllib.parse import unquote, urlparse

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from dataforge import paths, schema, writing
from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.timing import RegisterRecord, SequenceTimer, append_record


@dataclass
class Config:
    """Register a dataset's converted recordings into a catalog."""

    dataset: AnnotatedDatasetUnion = field(default_factory=RobocapConfig)
    """Dataset whose rrds get registered; its registry key is the catalog dataset name."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """gRPC URL of a locally running ``rerun server`` catalog."""
    catalog_name: str | None = None
    """Override the catalog name without changing file identities."""
    sequences: tuple[str, ...] | None = None
    """Register only these recording IDs."""
    replace: bool = False
    """Re-register a layer the catalog already holds for a segment, instead of skipping it.

    Registration is normally ``SKIP``, so re-running it over a corpus is cheap
    and idempotent. But a regenerated layer — ``rm gt/*.rrd`` and a convert, per
    README#the-layer-rule — is a *new file at a registered path*, and ``SKIP``
    leaves the server serving the old registration: the rebuilt rrd stays
    unregistered and nothing says so. ``--replace`` is what to use after a
    rebuild."""


def main(config: Config) -> None:
    """Create the dataset if needed, register every layer's rrds (idempotent), and republish its blueprints."""
    timer: SequenceTimer = SequenceTimer()
    dataset_config: DataforgeDatasetConfig = config.dataset
    dataset: DataforgeDataset = dataset_config.setup()
    name: str = dataset_config.name
    output_root: Path = paths.output_root()
    paths_by_layer: dict[str, list[Path]] = {layer: sorted((output_root / layer).glob(f"{name}__*.rrd")) for layer in dataset.layers}
    if config.sequences is not None:
        missing: set[str] = set(config.sequences) - {path.stem for path in paths_by_layer[paths.BASE_LAYER]}
        if missing:
            raise ValueError(f"Unknown recording IDs: {sorted(missing)}")
        paths_by_layer = {layer: [path for path in files if path.stem in config.sequences] for layer, files in paths_by_layer.items()}
    if not paths_by_layer[paths.BASE_LAYER]:
        raise FileNotFoundError(f"no {paths.BASE_LAYER}-layer rrds for {name} under {output_root / paths.BASE_LAYER}")

    name = config.catalog_name or name
    client: CatalogClient = CatalogClient(config.catalog_url)
    entry: DatasetEntry = client.create_dataset(name, exist_ok=True)
    on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE if config.replace else OnDuplicateSegmentLayer.SKIP
    for layer, rrd_paths in paths_by_layer.items():
        if not rrd_paths:
            continue  # a derived layer nobody has produced yet
        with timer.stage(layer):
            entry.register([path.resolve().as_uri() for path in rrd_paths], layer_name=layer, on_duplicate=on_duplicate).wait()

    # Every run republishes the blueprints from the current code, so a new layer's columns and a changed
    # layout both land without a separate step. register_blueprint only ever ADDS an entry, and a live
    # server holds registered files open (never overwrite them), so the run writes new stamped files,
    # registers them as the defaults, and retires every entry the blueprint dataset listed before it.
    with timer.stage("blueprint"):
        blueprint_entries: DatasetEntry | None = entry.blueprint_dataset()  # None while the dataset has no blueprints
        retiring: dict[str, Path] = {}
        if blueprint_entries is not None:
            listed: list[dict[str, Any]] = (
                blueprint_entries.segment_table().select("rerun_segment_id", "rerun_storage_urls").to_arrow_table().to_pylist()
            )
            retiring = {row["rerun_segment_id"]: Path(unquote(urlparse(row["rerun_storage_urls"][0]).path)) for row in listed}
        stamp: str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  # microseconds: back-to-back runs never share a file name
        blueprint_path: Path = paths.blueprint_path(output_root, name, stamp=stamp).resolve()
        with writing.atomic_write(blueprint_path) as temp_path:
            dataset.default_blueprint().save(name, str(temp_path))
        entry.register_blueprint(blueprint_path.as_uri(), set_default=True)
        table_path: Path = paths.blueprint_path(output_root, name, segment_table=True, stamp=stamp).resolve()
        columns: list[str] = entry.segment_table().schema().names
        writing.save_table_blueprint(dataset.table_blueprint(), table_path, timeline=schema.TIMELINE, fields=dataset.table_fields(), columns=columns)
        entry.register_blueprint(table_path.as_uri(), set_default=True, segment_table=True)
        if blueprint_entries is not None and retiring:
            blueprint_entries.unregister(segments_to_drop=list(retiring), layers_to_drop=[]).wait()
            ours: list[Path] = [path for path in retiring.values() if path.parent == blueprint_path.parent]  # both resolved
            for path in ours:
                path.unlink(missing_ok=True)
            print(f"retired {len(retiring)} older blueprint entries of '{name}' and deleted {len(ours)} of their files")
    counted: str = ", ".join(f"{len(found)} {layer}" for layer, found in paths_by_layer.items() if found)
    how: str = "replacing duplicates" if config.replace else "skipping duplicates"
    print(f"registered {counted} rrds into '{name}' at {config.catalog_url} ({how})")

    append_record(
        output_root / "timing/register.jsonl",
        RegisterRecord(
            dataset=name,
            started_at=timer.started_at,
            layer_s={key: value for key, value in timer.stage_s.items() if key != "blueprint"},
            blueprint_s=timer.stage_s["blueprint"],
            segment_count=len(paths_by_layer[paths.BASE_LAYER]),
            total_s=timer.total_s,
            host=gethostname(),
        ),
    )
