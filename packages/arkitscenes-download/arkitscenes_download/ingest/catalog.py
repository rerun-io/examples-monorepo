"""Register ingested ARKitScenes RRDs into a local Rerun catalog.

The local catalog is the in-memory ``rerun server`` (start it with
``pixi run arkitscenes-download-serve``). Each sequence is one segment assembled from eight required
layer RRDs sharing its ``recording_id`` (the ARKitScenes video id), plus two optional
CA-1M laser-GT layers on covered captures; generic portrait and landscape layouts
are registered as dataset blueprints.

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

from beartype.roar import BeartypeException
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from arkitscenes_download.ingest.blueprint import make_blueprint, make_table_blueprint
from arkitscenes_download.schema import ALL_LAYER_NAMES, OPTIONAL_LAYER_NAMES, REQUIRED_LAYER_NAMES

DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
"""gRPC URL of a locally-running ``rerun server`` catalog."""
CONSOLE: Console = Console(markup=False)
LAYER_COLUMN: str = "rerun_layer_names"
"""Segment-table column listing each segment's registered layers."""
POLL_INTERVAL_S: float = 30.0
"""Segment-table poll cadence while the server grinds through a dropped call."""
STALL_LIMIT: int = 6
"""Consecutive no-progress polls before a dropped registration counts as failed."""
POLL_DEADLINE_S: float = 2 * 3600.0
"""Hard ceiling on polling a dropped registration — turns a dead server into an error."""


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
    layer_major: bool = (config.rrd_dir / REQUIRED_LAYER_NAMES[0]).is_dir()

    def rrd_path(layer: str, video_id: str) -> Path:
        return config.rrd_dir / layer / f"{video_id}.rrd" if layer_major else config.rrd_dir / video_id / f"{layer}.rrd"

    if config.video_ids is None:
        pattern: str = "{layer}/*.rrd" if layer_major else "*/{layer}.rrd"
        return {layer: sorted(config.rrd_dir.glob(pattern.format(layer=layer))) for layer in ALL_LAYER_NAMES}
    return {layer: [path for video_id in config.video_ids if (path := rrd_path(layer, video_id)).is_file()] for layer in ALL_LAYER_NAMES}


def registered_layer_count(dataset: DatasetEntry, layer_name: str, video_ids: set[str]) -> int:
    """Count segments among ``video_ids`` that already carry ``layer_name``."""
    table = dataset.segment_table().df.to_pandas()
    table = table[table["rerun_segment_id"].isin(video_ids)]
    return sum(1 for layers in table[LAYER_COLUMN] if layer_name in layers)


def register_layer(config: Config, layer_name: str, paths: list[Path], progress: Progress, task_id: TaskID) -> None:
    """Register one layer's files, riding out client-side connection drops.

    The server keeps processing (and eventually commits) after the client's
    gRPC call drops, so on error we poll the segment table until the layer
    reaches ``len(paths)`` registered segments or stops making progress.
    """
    dataset: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset_name)
    expected: int = len(paths)
    # Segment id per file: layer-major names files <video_id>.rrd; legacy nests <video_id>/<layer>.rrd.
    video_ids: set[str] = {path.parent.name if path.stem == layer_name else path.stem for path in paths}
    if registered_layer_count(dataset, layer_name, video_ids) >= expected:
        CONSOLE.print(f"{layer_name}: already complete")
        return
    started: float = time.perf_counter()
    try:
        dataset.register(
            [path.resolve().as_uri() for path in paths],
            layer_name=layer_name,
            on_duplicate=OnDuplicateSegmentLayer.SKIP,
        ).wait()
    except BeartypeException:
        raise
    except Exception as error:  # noqa: BLE001 — dropped call; poll server-side progress instead of resubmitting
        if not isinstance(error, ConnectionError) and "(Connection)" not in str(error):
            raise
        progress.update(task_id, description=f"{layer_name} · {expected} files · connection dropped; polling server")
        deadline: float = time.perf_counter() + POLL_DEADLINE_S
        last_count, stalls = -1, 0
        while True:
            time.sleep(POLL_INTERVAL_S)
            if time.perf_counter() > deadline:
                raise RuntimeError(f"{layer_name} still incomplete after {POLL_DEADLINE_S / 3600:.0f}h of polling — server dead?") from error
            try:
                dataset = CatalogClient(config.catalog_url).get_dataset(config.dataset_name)
                count: int = registered_layer_count(dataset, layer_name, video_ids)
            except BeartypeException:
                raise
            except Exception:  # noqa: BLE001 — server refuses connections while grinding
                progress.update(task_id, description=f"{layer_name} · {expected} files · server busy")
                continue
            progress.update(task_id, description=f"{layer_name} · {count}/{expected} registered")
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

    active_layers: list[tuple[str, list[Path]]] = [(layer_name, paths) for layer_name, paths in layer_paths.items() if paths]
    overall_progress: Progress = Progress(TextColumn("layers"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=CONSOLE)
    current_progress: Progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=CONSOLE)
    overall_task: TaskID = overall_progress.add_task("layers", total=len(active_layers))
    with Live(Group(overall_progress, current_progress), console=CONSOLE, refresh_per_second=10.0):
        for layer_name, paths in active_layers:
            current_task: TaskID = current_progress.add_task(f"{layer_name} · {len(paths)} files · submitting", total=None)
            register_layer(config, layer_name, paths, current_progress, current_task)
            current_progress.remove_task(current_task)
            overall_progress.advance(overall_task)

    # Most ARKitScenes captures are handheld portrait scans, so the portrait
    # layout is the dataset default; landscape stays selectable in the viewer.
    blueprint_dir: Path = config.rrd_dir / "blueprints"
    blueprint_dir.mkdir(parents=True, exist_ok=True)
    for orientation, is_portrait in (("landscape", False), ("portrait", True)):
        blueprint_path: Path = blueprint_dir / f"{config.dataset_name}-{orientation}.rbl"
        make_blueprint(is_portrait).save("arkitscenes", blueprint_path)
        blueprint_path.chmod(0o644)  # uid-squashing NFS mounts can land fresh writes as mode 0000
        dataset.register_blueprint(blueprint_path.resolve().as_uri(), set_default=is_portrait)
    # Segment-table blueprint: 3D preview cards in the dataset review (experimental,
    # needs "Table cards and blueprints" enabled in the viewer's settings). A failure
    # here must not abort the run: the layers are already registered, and aborting
    # would also skip main()'s completeness verification.
    table_blueprint_path: Path = blueprint_dir / f"{config.dataset_name}-table.rbl"
    make_table_blueprint().save("arkitscenes", table_blueprint_path)
    table_blueprint_path.chmod(0o644)  # uid-squashing NFS mounts can land fresh writes as mode 0000
    try:
        dataset.register_blueprint(table_blueprint_path.resolve().as_uri(), segment_table=True)
    except BeartypeException:
        raise
    except Exception as error:  # noqa: BLE001
        CONSOLE.print(f"segment-table blueprint (experimental) failed: {str(error)[:70]} — skipping")
    return dataset


def main(config: Config) -> None:
    """Register all ingested sequences and verify per-segment layer completeness."""
    dataset: DatasetEntry = register_sequences(config)
    table = dataset.segment_table().df.to_pandas()
    required_layers: set[str] = set(REQUIRED_LAYER_NAMES)
    optional_layers: set[str] = set(OPTIONAL_LAYER_NAMES)
    segment_layers: list[set[str]] = [set(layers) for layers in table[LAYER_COLUMN]]
    complete: int = sum(1 for layers in segment_layers if required_layers <= layers)
    gt_covered: int = sum(1 for layers in segment_layers if optional_layers <= layers)
    CONSOLE.print(f"dataset '{config.dataset_name}' at {config.catalog_url}: {len(table)} segments, {complete} complete, {gt_covered} gt-covered")
    if complete != len(table):
        raise RuntimeError(f"incomplete registration: only {complete}/{len(table)} segments carry all {len(REQUIRED_LAYER_NAMES)} required layers")
