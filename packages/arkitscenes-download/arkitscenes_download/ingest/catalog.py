"""Register ingested ARKitScenes RRDs into a local Rerun catalog.

The local catalog is the in-memory ``rerun server`` (start it with
``pixi run serve``). Each sequence is one segment assembled from seven layer
RRDs sharing its ``recording_id`` (the ARKitScenes video id); generic portrait
and landscape layouts are registered as dataset blueprints.

Hard-won registration rules baked in here (see docs/full-run-runbook.md §8):
``SKIP`` duplicates instead of ``REPLACE`` (REPLACE invalidates the server's
schema cache, forcing an O(all-sources) recompute per file); a dropped client
gRPC call does NOT stop the server, so on connection errors we poll the
segment table for server-side completion instead of resubmitting (duplicate
checks run *after* the expensive work, so a resubmit pays nearly full price);
and completion is verified per-segment layer count, never by segment count
alone.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rich.console import Console

from arkitscenes_download.ingest.blueprint import make_blueprint
from arkitscenes_download.ingest.layers import LAYER_NAMES

DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
"""gRPC URL of a locally-running ``rerun server`` catalog."""
CONSOLE: Console = Console(markup=False)
POLL_INTERVAL_S: float = 30.0
"""Segment-table poll cadence while the server grinds through a dropped call."""
STALL_LIMIT: int = 6
"""Consecutive no-progress polls before a dropped registration counts as failed."""


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for registering ingested sequences into a catalog."""

    rrd_dir: Path = Path("data/rrd")
    """RRD root: layer-major ``<layer>/<video_id>.rrd`` (canonical) or legacy ``<video_id>/<layer>.rrd``."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the local Rerun catalog (``rerun server``)."""
    dataset_name: str = "arkitscenes"
    """Catalog dataset name to (re)create."""
    recreate: bool = False
    """Delete and recreate the dataset before registering."""
    video_ids: list[str] | None = None
    """Optional sequence subset to register from the RRD root."""


def layer_files(config: Config) -> dict[str, list[Path]]:
    """Map each layer to its RRD paths, auto-detecting layer-major vs legacy layout."""
    layer_major: bool = (config.rrd_dir / LAYER_NAMES[0]).is_dir()
    if config.video_ids is None:
        if layer_major:
            return {layer: sorted((config.rrd_dir / layer).glob("*.rrd")) for layer in LAYER_NAMES}
        return {layer: sorted(config.rrd_dir.glob(f"*/{layer}.rrd")) for layer in LAYER_NAMES}
    return {
        layer: [
            path
            for video_id in config.video_ids
            if (path := (config.rrd_dir / layer / f"{video_id}.rrd") if layer_major else (config.rrd_dir / video_id / f"{layer}.rrd")).is_file()
        ]
        for layer in LAYER_NAMES
    }


def registered_layer_count(dataset: DatasetEntry, layer_name: str) -> int:
    """Count segments that already carry ``layer_name``."""
    table = dataset.segment_table().df.to_pandas()
    layer_column: str = next(column for column in table.columns if "layer" in column.lower())
    return sum(1 for layers in table[layer_column] if layer_name in layers)


def register_layer(client: CatalogClient, config: Config, layer_name: str, paths: list[Path]) -> None:
    """Register one layer's files, riding out client-side connection drops.

    The server keeps processing (and eventually commits) after the client's
    gRPC call drops, so on error we poll the segment table until the layer
    reaches ``len(paths)`` registered segments or stops making progress.
    """
    dataset: DatasetEntry = client.get_dataset(config.dataset_name)
    expected: int = len(paths)
    if registered_layer_count(dataset, layer_name) >= expected:
        CONSOLE.print(f"{layer_name}: already complete")
        return
    started: float = time.perf_counter()
    try:
        dataset.register(
            [path.resolve().as_uri() for path in paths],
            layer_name=layer_name,
            on_duplicate=OnDuplicateSegmentLayer.SKIP,
        ).wait()
    except Exception as error:  # noqa: BLE001 — dropped call; poll server-side progress instead of resubmitting
        CONSOLE.print(f"{layer_name}: client call dropped ({str(error)[:70]}) — polling server-side progress")
        last_count, stalls = -1, 0
        while True:
            time.sleep(POLL_INTERVAL_S)
            try:
                dataset = CatalogClient(config.catalog_url).get_dataset(config.dataset_name)
                count: int = registered_layer_count(dataset, layer_name)
            except Exception:  # noqa: BLE001 — server refuses connections while grinding
                continue
            if count >= expected:
                break
            stalls = stalls + 1 if count == last_count else 0
            last_count = count
            if stalls >= STALL_LIMIT:
                raise RuntimeError(f"{layer_name} stalled at {count}/{expected} registered segments") from error
    CONSOLE.print(f"{layer_name}: {expected} files in {time.perf_counter() - started:.1f}s")


def register_sequences(config: Config) -> DatasetEntry:
    """Create (or resume) the dataset and register every RRD in ``rrd_dir``."""
    layer_paths: dict[str, list[Path]] = layer_files(config)
    if not any(layer_paths.values()):
        raise FileNotFoundError(f"no .rrd files found in {config.rrd_dir}")

    client: CatalogClient = CatalogClient(config.catalog_url)
    if config.recreate and config.dataset_name in client.dataset_names():
        client.get_dataset(config.dataset_name).delete()
    dataset: DatasetEntry = client.create_dataset(config.dataset_name, exist_ok=True)

    for layer_name, paths in layer_paths.items():
        if paths:
            register_layer(client, config, layer_name, paths)

    # Most ARKitScenes captures are handheld portrait scans, so the portrait
    # layout is the dataset default; landscape stays selectable in the viewer.
    blueprint_dir: Path = config.rrd_dir / "blueprints" if (config.rrd_dir / "blueprints").is_dir() else config.rrd_dir
    for orientation, is_portrait in (("landscape", False), ("portrait", True)):
        blueprint_path: Path = blueprint_dir / f"{config.dataset_name}-{orientation}.rbl"
        make_blueprint(is_portrait).save(f"arkitscenes-{orientation}", blueprint_path)
        dataset.register_blueprint(blueprint_path.resolve().as_uri(), set_default=is_portrait)
    return dataset


def main(config: Config) -> None:
    """Register all ingested sequences and verify per-segment layer completeness."""
    dataset: DatasetEntry = register_sequences(config)
    table = dataset.segment_table().df.to_pandas()
    layer_column: str = next(column for column in table.columns if "layer" in column.lower())
    complete: int = sum(1 for layers in table[layer_column] if len(layers) == len(LAYER_NAMES))
    CONSOLE.print(f"dataset '{config.dataset_name}' at {config.catalog_url}: {len(table)} segments, {complete} with all {len(LAYER_NAMES)} layers")
    if complete != len(table):
        raise RuntimeError(f"incomplete registration: only {complete}/{len(table)} segments carry all {len(LAYER_NAMES)} layers")
