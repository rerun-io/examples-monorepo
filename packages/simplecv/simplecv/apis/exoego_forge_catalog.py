"""Utilities for serving ExoEgo Forge RRD files through a Rerun catalog.

The public ``main`` entrypoint mounts converted ExoEgo Forge recordings as
catalog datasets and creates one on-demand table per source dataset.
"""

from __future__ import annotations

import atexit
import base64
import os
import subprocess
import tempfile
import threading
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from rerun import bindings
from rerun.recording_stream import RecordingStream
from tqdm import tqdm

from simplecv.apis.view_exoego import create_container
from simplecv.rig import entity_id

APPLICATION_ID: str = "exoego-forge"
"""Rerun application id used by converted ExoEgo Forge recordings."""
TABLE_BLUEPRINT_METADATA_KEY: bytes = b"rerun:table_blueprint"
"""Arrow schema metadata key used by Rerun for experimental table blueprints."""
MARKER_FLAG_COLUMN: str = "marker_flag"
"""Boolean table flag column used by the Rerun table UI."""
TABLE_CARD_PREVIEW_START_SECONDS: float = 0.0
"""Start of the absolute ``video_time`` loop used by table-card previews."""
TABLE_CARD_PREVIEW_END_SECONDS: float = 10.0
"""End of the absolute ``video_time`` loop used by table-card previews."""
CATALOG_SHUTDOWN_TIMEOUT_SECONDS: float = 5.0
"""Maximum graceful shutdown wait after Ctrl-C before forcing process exit."""
DEFAULT_CATALOG_RRD_CACHE_DIR: Path = Path("~/.cache/simplecv/exoego-forge-catalog-optimized")
"""Default persistent cache root for catalog-compatible optimized RRD copies."""

DEFAULT_CATALOG_DATASETS: tuple[str, ...] = (
    "aria-gen2",
    "assembly101",
    "epfl-smart-kitchen",
    "hocap",
    "hot3d-aria",
    "hot3d-quest3",
    "umetrack",
    "ego-dex",
)

DEFAULT_CATALOG_OPTIMIZE_DATASETS: tuple[str, ...] = (
    "aria-gen2",
    "assembly101",
    "epfl-smart-kitchen",
    "hocap",
    "hot3d-aria",
    "hot3d-quest3",
    "umetrack",
    "ego-dex",
)
"""Datasets whose RRD chunk layout must be migrated before catalog registration.

This is not a size-only optimization. Registering these source RRDs directly
through the Rerun catalog importer can produce lossy catalog segments: visual
component columns such as pinholes, camera transforms, 2D keypoints, and 3D
points may be missing even though opening the same raw RRD directly in the
viewer works. The optimized copy preserves those columns for catalog tables and
clicked segment URLs.
"""

CATALOG_CAMERA_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "aria-gen2": {
        "ego": ("camera-rgb", "slam-front-left", "slam-front-right", "slam-side-left", "slam-side-right"),
        "exo": (),
    },
    "assembly101": {
        "ego": ("e1", "e2", "e3", "e4"),
        "exo": ("C10095", "C10115", "C10118", "C10119", "C10379", "C10390", "C10395", "C10404"),
    },
    "epfl-smart-kitchen": {
        "ego": ("hololens",),
        "exo": (
            "output0",
            "Aoutput0",
            "Aoutput1",
            "Aoutput2",
            "Aoutput3",
            "Boutput0",
            "Boutput1",
            "Boutput2",
            "Boutput3",
        ),
    },
    "hocap": {
        "ego": ("hololens_kv5h72",),
        "exo": (
            "037522251142",
            "043422252387",
            "046122250168",
            "105322251225",
            "105322251564",
            "108222250342",
            "115422250549",
            "117222250549",
        ),
    },
    "hot3d-aria": {
        "ego": ("camera-rgb", "camera-slam-left", "camera-slam-right"),
        "exo": (),
    },
    "hot3d-quest3": {
        "ego": ("camera-slam-left", "camera-slam-right"),
        "exo": (),
    },
    "umetrack": {
        "ego": ("BL", "BR", "TL", "TR"),
        "exo": (),
    },
    "ego-dex": {
        "ego": ("avp_camera",),
        "exo": (),
    },
}

CATALOG_TABLE_PREVIEW_CAMERAS: dict[str, tuple[str, str]] = {
    "assembly101": ("ego", "e3"),
}
"""Dataset-specific table-card video preview overrides."""


@runtime_checkable
class CatalogServer(Protocol):
    """Minimal server interface used by catalog registration."""

    def url(self) -> str:
        """Return the catalog URL."""
        ...

    def client(self) -> Any:
        """Return a Rerun catalog client."""
        ...

    def shutdown(self) -> None:
        """Stop the catalog server."""
        ...


def _catalog_cam_node(camera_names: dict[str, tuple[str, ...]], kind: str, camera_name: str) -> Path:
    """Map a catalog camera (by ``kind`` + name) to its ``exoego:v2`` cam entity node.

    Mirrors ``BaseExoEgoSequence.build_rig_layout``'s deterministic indexing for the
    rigid case (every current dataset): each exo camera is its own world-anchored rig
    ``rig_<i>/cam_00``; the worn ego device is one moving rig ``rig_<num_exo>`` with
    one camera per ego stream ``cam_<j>``.

    This holds only while ``camera_names`` lists the exo/ego streams in the same order
    the builder enumerates them and every ego stream is calibrated. A future
    non-rigidly-factorable ego device would fall back to per-camera ego rigs
    (``rig_<num_exo + j>/cam_00``), which this static mapping cannot represent; no
    catalog dataset hits that today.
    """
    exo_names: tuple[str, ...] = camera_names.get("exo", ())
    ego_names: tuple[str, ...] = camera_names.get("ego", ())
    if kind == "exo":
        return Path("world") / entity_id("rig", exo_names.index(camera_name)) / entity_id("cam", 0)
    return Path("world") / entity_id("rig", len(exo_names)) / entity_id("cam", ego_names.index(camera_name))


def build_exoego_catalog_blueprint(dataset_name: str) -> rrb.Blueprint:
    """Build the default catalog blueprint for one ExoEgo Forge dataset.

    Args:
        dataset_name: Dataset key used to select known ego and exo camera names.

    Returns:
        Rerun blueprint used as the default view when opening a dataset segment.
    """
    camera_names: dict[str, tuple[str, ...]] = CATALOG_CAMERA_NAMES.get(dataset_name, {"ego": (), "exo": ()})
    # Recordings now follow the exoego:v2 rig layout (/world/rig_NN/cam_MM); map each
    # known camera to its cam node and feed the human names through so 2D panels carry
    # the skip list + readable rig/cam titles (matching direct view_exoego viewing).
    ego_video_log_paths: list[Path] = []
    exo_video_log_paths: list[Path] = []
    video_path_to_name: dict[Path, str] = {}
    for kind, video_log_paths in (("ego", ego_video_log_paths), ("exo", exo_video_log_paths)):
        for name in camera_names[kind]:
            video_path: Path = _catalog_cam_node(camera_names, kind, name) / "pinhole" / "video"
            video_log_paths.append(video_path)
            video_path_to_name[video_path] = name
    container: rrb.ContainerLike = create_container(
        ego_video_log_paths=ego_video_log_paths,
        exo_video_log_paths=exo_video_log_paths,
        skip_camera_names=frozenset(),
        video_path_to_name=video_path_to_name,
    )
    return rrb.Blueprint(container, collapse_panels=True)


def _register_default_dataset_blueprint(
    server: Any,
    dataset_entry: Any,
    *,
    dataset_name: str,
    application_id: str = APPLICATION_ID,
) -> Path:
    """Save and register the full per-segment default blueprint for one dataset.

    Args:
        server: Rerun server that owns the dataset and temporary blueprint lifetime.
        dataset_entry: Catalog dataset entry returned by the Rerun server client.
        dataset_name: Dataset key used to build the dataset-specific blueprint.
        application_id: Rerun application id used when serializing the blueprint.

    Returns:
        Path to the temporary ``.rbl`` blueprint file registered with the dataset.
    """
    blueprint: rrb.Blueprint = build_exoego_catalog_blueprint(dataset_name)
    tmp_dir = tempfile.TemporaryDirectory(prefix=f"{dataset_name}-")
    # Keep the temporary blueprint file alive for as long as the server object is alive.
    weakref.finalize(server, tmp_dir.cleanup)
    atexit.register(tmp_dir.cleanup)
    blueprint_path: Path = Path(tmp_dir.name) / f"{dataset_name}.rbl"
    blueprint.save(application_id, path=str(blueprint_path))
    dataset_entry.register_blueprint(blueprint_path.resolve().as_uri(), set_default=True)
    return blueprint_path


def discover_rrd_paths(
    rrd_root: Path,
    *,
    datasets: tuple[str, ...] = DEFAULT_CATALOG_DATASETS,
) -> dict[str, list[Path]]:
    """Discover local RRD paths grouped by first-level catalog dataset directory.

    Args:
        rrd_root: Directory containing one subdirectory per dataset.
        datasets: Dataset directory names to include. An empty tuple scans all
            first-level directories under ``rrd_root``.

    Returns:
        Mapping from dataset name to absolute paths for every discovered
        ``.rrd`` file. Datasets without RRD files are omitted.

    Raises:
        FileNotFoundError: If ``rrd_root`` does not exist or no requested
            datasets contain RRD files.
    """
    root: Path = rrd_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"RRD root directory does not exist: {root}")

    dataset_dirs: list[Path] = (
        [root / dataset for dataset in datasets] if datasets else sorted(d for d in root.iterdir() if d.is_dir())
    )

    paths_by_dataset: dict[str, list[Path]] = {}
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            continue
        rrd_paths: list[Path] = sorted(path.resolve() for path in dataset_dir.rglob("*.rrd"))
        if rrd_paths:
            paths_by_dataset[dataset_dir.name] = rrd_paths

    if not paths_by_dataset:
        dataset_desc: str = ", ".join(datasets) if datasets else "all first-level directories"
        raise FileNotFoundError(f"No RRD files found under {root} for datasets: {dataset_desc}")

    return paths_by_dataset


def discover_rrd_uris(
    rrd_root: Path,
    *,
    datasets: tuple[str, ...] = DEFAULT_CATALOG_DATASETS,
) -> dict[str, list[str]]:
    """Discover local RRD URIs grouped by first-level catalog dataset directory.

    Args:
        rrd_root: Directory containing one subdirectory per dataset.
        datasets: Dataset directory names to include. An empty tuple scans all
            first-level directories under ``rrd_root``.

    Returns:
        Mapping from dataset name to absolute ``file://`` URI strings for every
        discovered ``.rrd`` file.

    Raises:
        FileNotFoundError: If ``rrd_root`` does not exist or no requested
            datasets contain RRD files.
    """
    paths_by_dataset: dict[str, list[Path]] = discover_rrd_paths(rrd_root, datasets=datasets)
    uris_by_dataset: dict[str, list[str]] = {
        dataset_name: [path.as_uri() for path in rrd_paths] for dataset_name, rrd_paths in paths_by_dataset.items()
    }
    return uris_by_dataset


def _optimize_rrd_for_catalog(source_path: Path, *, rrd_root: Path, cache_root: Path) -> Path:
    """Build or reuse a catalog-compatible optimized RRD copy.

    ``rerun rrd optimize`` rewrites the recording into a chunk layout that the
    catalog importer indexes correctly. Without this step, some ExoEgo Forge
    datasets register as incomplete catalog segments: table previews and opened
    segment URLs can lose camera frustums, pinholes, hand keypoints, and other
    visual columns despite the raw RRD containing them.

    Args:
        source_path: Original source RRD path.
        rrd_root: Root used to preserve relative dataset paths in the cache.
        cache_root: Directory where optimized RRDs are stored.

    Returns:
        Path to a catalog-compatible RRD copy.

    Raises:
        ValueError: If ``source_path`` is not under ``rrd_root``.
        RuntimeError: If Rerun cannot optimize the source RRD.
    """
    source_resolved: Path = source_path.expanduser().resolve()
    rrd_root_resolved: Path = rrd_root.expanduser().resolve()
    cache_root_resolved: Path = cache_root.expanduser().resolve()
    try:
        relative_path: Path = source_resolved.relative_to(rrd_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Cannot optimize {source_resolved} for the catalog cache because it is not under "
            f"RRD root {rrd_root_resolved}."
        ) from exc
    optimized_path: Path = cache_root_resolved / relative_path
    if optimized_path.exists() and optimized_path.stat().st_mtime_ns >= source_resolved.stat().st_mtime_ns:
        return optimized_path

    optimized_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path = optimized_path.with_suffix(f"{optimized_path.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    completed_process: subprocess.CompletedProcess[str] = subprocess.run(
        ["rerun", "rrd", "optimize", str(source_resolved), "-o", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode != 0:
        raise RuntimeError(
            f"Failed to optimize {source_resolved} for Rerun catalog registration.\n{completed_process.stderr.strip()}"
        )

    tmp_path.replace(optimized_path)
    return optimized_path


def mount_catalog(
    rrd_root: Path,
    *,
    datasets: tuple[str, ...] = DEFAULT_CATALOG_DATASETS,
    port: int | None = None,
    application_id: str = APPLICATION_ID,
    show_progress: bool = True,
    optimize_for_catalog: bool = True,
    catalog_rrd_cache_dir: Path = DEFAULT_CATALOG_RRD_CACHE_DIR,
    optimize_datasets: tuple[str, ...] = DEFAULT_CATALOG_OPTIMIZE_DATASETS,
) -> CatalogServer:
    """Mount local ExoEgo Forge RRDs as one Rerun catalog dataset per source.

    The default path registers optimized cache copies rather than raw source
    RRDs. Keep this enabled unless you are explicitly debugging Rerun catalog
    import behavior: raw registration has been observed to drop important
    visual columns from nontrivial recordings even though direct ``rerun
    file.rrd`` playback still looks correct.

    Args:
        rrd_root: Directory containing local RRD files grouped by dataset.
        datasets: Dataset directory names to mount. An empty tuple scans all
            first-level directories under ``rrd_root``.
        port: gRPC port for the Rerun server. If ``None``, Rerun chooses a port.
        application_id: Rerun application id used for registered blueprints.
        show_progress: Whether to show a ``tqdm`` progress bar while registering.
        optimize_for_catalog: Whether to register optimized cache copies. Turning
            this off can recreate the catalog schema-loss bug described above.
        catalog_rrd_cache_dir: Persistent cache root for optimized RRD copies.
        optimize_datasets: Dataset names to optimize before registration.

    Returns:
        Running Rerun server with discovered RRD files registered as datasets.

    Raises:
        FileNotFoundError: If no matching RRD files are found.
    """
    paths_by_dataset: dict[str, list[Path]] = discover_rrd_paths(rrd_root, datasets=datasets)
    dataset_names: list[str] = sorted(paths_by_dataset)
    total_files: int = sum(len(paths_by_dataset[name]) for name in dataset_names)

    print(
        f"Mounting catalog from {rrd_root.expanduser().resolve()} "
        f"({total_files} RRDs across {len(dataset_names)} datasets: {', '.join(dataset_names)})",
        flush=True,
    )

    registration_paths_by_dataset: dict[str, list[Path]] = {}
    iterator = tqdm(dataset_names, desc="prepare", unit="dataset", disable=not show_progress)
    for dataset_name in iterator:
        source_paths: list[Path] = paths_by_dataset[dataset_name]
        if optimize_for_catalog and dataset_name in optimize_datasets:
            iterator.set_postfix_str(f"{dataset_name} optimize ({len(source_paths)} files)")
            registration_paths: list[Path] = [
                _optimize_rrd_for_catalog(path, rrd_root=rrd_root, cache_root=catalog_rrd_cache_dir)
                for path in source_paths
            ]
        else:
            registration_paths = source_paths
        registration_paths_by_dataset[dataset_name] = registration_paths

    server_datasets: dict[str, os.PathLike[str] | Sequence[os.PathLike[str] | str] | str] = {}
    for dataset_name, registration_paths in registration_paths_by_dataset.items():
        registration_pathlikes: list[os.PathLike[str] | str] = list(registration_paths)
        server_datasets[dataset_name] = registration_pathlikes

    # Loading datasets at server startup avoids cumulative catalog RPC pressure
    # for large datasets such as Assembly101 while preserving segment URLs.
    server: rr.server.Server = rr.server.Server(datasets=server_datasets, port=port)
    client = server.client()

    for dataset_name in tqdm(dataset_names, desc="blueprint", unit="dataset", disable=not show_progress):
        dataset = client.get_dataset(dataset_name)
        _register_default_dataset_blueprint(
            server,
            dataset,
            dataset_name=dataset_name,
            application_id=application_id,
        )

    return server


@dataclass
class CatalogConfig:
    """Config for the general ExoEgo Forge catalog index server."""

    rrd_root: Path = Path("data/exoego-forge-catalog")
    """Directory containing ``<dataset>/**/*.rrd`` files."""
    datasets: tuple[str, ...] = DEFAULT_CATALOG_DATASETS
    """Dataset directories to mount. Empty tuple scans all first-level directories."""
    port: int = 9988
    """gRPC port for the catalog server."""
    application_id: str = APPLICATION_ID
    """Application id used to save default dataset blueprints. Must match converted RRDs."""
    optimize_for_catalog: bool = True
    """Register optimized cache copies to avoid lossy Rerun catalog segment imports."""
    catalog_rrd_cache_dir: Path = DEFAULT_CATALOG_RRD_CACHE_DIR
    """Persistent cache directory for catalog-compatible optimized RRD copies."""
    optimize_datasets: tuple[str, ...] = DEFAULT_CATALOG_OPTIMIZE_DATASETS
    """Dataset names whose RRDs should be optimized before catalog registration.

    Removing a dataset here can make its catalog table/segment schema incomplete
    even when the underlying raw RRD opens correctly in the viewer.
    """
    open_browser: bool = False
    """Also host a web viewer and open it."""
    web_port: int = 9091
    """Web viewer port. Only used when ``open_browser`` is true."""


@dataclass(frozen=True, slots=True)
class RRDIndexRow:
    """One lightweight index row for a registered RRD recording segment."""

    id: int
    """Stable row id after sorting by sequence key."""
    dataset: str
    """Source ExoEgo Forge dataset name."""
    sequence_key: str
    """Human-readable sequence key under the source dataset."""
    recording_uri: str
    """Catalog segment URL used by the Rerun table preview column."""
    path: str
    """Absolute filesystem path for the RRD recording."""
    size_bytes: int
    """RRD file size in bytes."""
    marker_flag: bool = False
    """User-editable marker flag column for table review workflows."""


def table_name_for_dataset(dataset_name: str) -> str:
    """Build the catalog table name for one source dataset.

    Args:
        dataset_name: Source dataset name.

    Returns:
        Catalog table name using underscores instead of dashes.
    """
    dataset_slug: str = dataset_name.replace("-", "_")
    return f"{dataset_slug}_table"


def build_rrd_index_rows_from_paths(rrd_root: Path, *, dataset_name: str) -> list[RRDIndexRow]:
    """Build filesystem-only RRD index rows.

    This helper bypasses the Rerun catalog and is mainly useful for tests and
    diagnostics.

    Args:
        rrd_root: Catalog root containing one subdirectory per dataset.
        dataset_name: Dataset directory name to scan under ``rrd_root``.

    Returns:
        Rows sorted by filesystem path with sequence keys relative to the
        resolved dataset root.

    Raises:
        FileNotFoundError: If the catalog root is missing or the dataset
            contains no RRD files.
    """
    root: Path = rrd_root.expanduser().resolve()
    dataset_dir: Path = root / dataset_name
    paths_by_dataset: dict[str, list[Path]] = discover_rrd_paths(root, datasets=(dataset_name,))
    rrd_paths: list[Path] = paths_by_dataset[dataset_name]

    rows: list[RRDIndexRow] = []
    for idx, rrd_path in enumerate(rrd_paths):
        resolved_path: Path = rrd_path.resolve()
        sequence_key: str = resolved_path.relative_to(dataset_dir).with_suffix("").as_posix()
        row: RRDIndexRow = RRDIndexRow(
            id=idx,
            dataset=dataset_name,
            sequence_key=sequence_key,
            recording_uri=str(resolved_path),
            path=str(resolved_path),
            size_bytes=resolved_path.stat().st_size,
        )
        rows.append(row)
    return rows


def _optional_segment_column_values(table: pa.Table, column_name: str) -> list[Any | None]:
    """Read optional segment metadata values from a catalog segment table.

    Args:
        table: Segment metadata table returned by the Rerun catalog client.
        column_name: Metadata column to read.

    Returns:
        One value per row. Missing columns are represented as ``None`` values.
    """
    if column_name not in table.schema.names:
        return [None] * table.num_rows
    values: list[Any | None] = []
    for value in table.column(column_name).to_pylist():
        # Rerun catalog metadata columns can arrive as one-item Arrow lists.
        # Empty lists represent missing metadata for that row.
        if isinstance(value, list):
            values.append(value[0] if value else None)
        else:
            values.append(value)
    return values


def _sequence_key_from_recording_id(dataset_name: str, recording_id: str) -> str:
    """Recover a slash-separated sequence key from a catalog recording id.

    Args:
        dataset_name: Dataset prefix expected at the start of ``recording_id``.
        recording_id: Segment id using ``__`` separators.

    Returns:
        Sequence key with path separators restored.
    """
    prefix: str = f"{dataset_name}__"
    if recording_id.startswith(prefix):
        return recording_id[len(prefix) :].replace("__", "/")
    return recording_id.replace("__", "/")


def build_rrd_index_rows_from_dataset(
    dataset_entry: Any,
    *,
    dataset_dir: Path,
    dataset_name: str,
) -> list[RRDIndexRow]:
    """Build table rows from registered catalog segment URLs.

    Args:
        dataset_entry: Rerun catalog dataset entry containing registered
            recording segments.
        dataset_dir: Local dataset root used to resolve file paths and sizes.
        dataset_name: Dataset prefix used when deriving sequence keys from
            recording ids.

    Returns:
        RRD index rows sorted by sequence key and re-numbered with stable ids.

    Raises:
        FileNotFoundError: If the registered dataset has no segment rows.
    """
    segment_batches: list[pa.RecordBatch] = dataset_entry.segment_table().collect()
    if not segment_batches:
        raise FileNotFoundError(f"Registered {dataset_name} dataset has no segments.")
    segment_table: pa.Table = pa.Table.from_batches(segment_batches)
    if segment_table.num_rows == 0:
        raise FileNotFoundError(f"Registered {dataset_name} dataset has no segments.")
    recording_ids: list[str] = [
        str(recording_id) for recording_id in segment_table.column("rerun_segment_id").to_pylist()
    ]
    sequence_key_values: list[Any | None] = _optional_segment_column_values(segment_table, "property:info:sequence_key")

    rows: list[RRDIndexRow] = []
    for idx, (recording_id, sequence_key_value) in enumerate(zip(recording_ids, sequence_key_values, strict=True)):
        sequence_key: str = (
            str(sequence_key_value)
            if sequence_key_value is not None
            else _sequence_key_from_recording_id(dataset_name, recording_id)
        )
        rrd_path: Path = (dataset_dir / f"{sequence_key}.rrd").resolve()
        recording_uri: str = str(dataset_entry.segment_url(recording_id))
        size_bytes: int = rrd_path.stat().st_size if rrd_path.exists() else 0
        row: RRDIndexRow = RRDIndexRow(
            id=idx,
            dataset=dataset_name,
            sequence_key=sequence_key,
            recording_uri=recording_uri,
            path=str(rrd_path),
            size_bytes=size_bytes,
        )
        rows.append(row)

    rows.sort(key=lambda row: row.sequence_key)
    return [
        RRDIndexRow(
            id=idx,
            dataset=row.dataset,
            sequence_key=row.sequence_key,
            recording_uri=row.recording_uri,
            path=row.path,
            size_bytes=row.size_bytes,
            marker_flag=row.marker_flag,
        )
        for idx, row in enumerate(rows)
    ]


def _table_preview_camera(dataset_name: str, camera_names: dict[str, tuple[str, ...]]) -> tuple[str, str] | None:
    """Choose the 2D video preview camera for a dataset table card.

    Args:
        dataset_name: Dataset key used to select an explicit override when one
            is needed to preserve known-good visual behavior.
        camera_names: Camera stream names available for the dataset.

    Returns:
        Camera group and name, or ``None`` if no camera stream is available.
    """
    configured_camera: tuple[str, str] | None = CATALOG_TABLE_PREVIEW_CAMERAS.get(dataset_name)
    if configured_camera is not None:
        configured_kind: str = configured_camera[0]
        configured_name: str = configured_camera[1]
        if configured_name in camera_names.get(configured_kind, ()):
            return configured_camera

    for kind in ("ego", "exo"):
        camera_names_for_kind: tuple[str, ...] = camera_names[kind]
        if camera_names_for_kind:
            return kind, camera_names_for_kind[0]

    return None


def build_table_card_blueprint(dataset_name: str, *, timeline: str = "video_time") -> rrb.Blueprint:
    """Build the lightweight table-card blueprint for dataset previews.

    This blueprint is embedded into the Arrow schema for the table itself. It is
    intentionally separate from the full per-recording blueprint registered on
    each catalog dataset, so changing table cards does not alter the layout used
    after opening a segment URL.

    Args:
        dataset_name: Dataset key used to select known ego and exo camera names.
        timeline: Timeline name used by the preview views.

    Returns:
        Rerun blueprint embedded into the dataset table schema.
    """
    camera_names: dict[str, tuple[str, ...]] = CATALOG_CAMERA_NAMES.get(dataset_name, {"ego": (), "exo": ()})

    # The 3D card should show poses, points, and camera frustums. Video entities
    # are explicitly excluded because they render in the sibling 2D card instead.
    video_exclusion_queries: list[str] = []
    for kind in ("ego", "exo"):
        for camera_name in camera_names[kind]:
            video_entity_path: str = f"/{_catalog_cam_node(camera_names, kind, camera_name)}/pinhole/video"
            video_exclusion_queries.append(f"- {video_entity_path}")
            video_exclusion_queries.append(f"- {video_entity_path}/**")

    scene_preview_view: rrb.Spatial3DView = rrb.Spatial3DView(
        origin="/",
        name="3D Preview",
        contents=["+ /**", *video_exclusion_queries],
        spatial_information=rrb.SpatialInformation.from_fields(show_axes=True),
    )

    preview_views: list[Any] = [scene_preview_view]
    preview_camera: tuple[str, str] | None = _table_preview_camera(dataset_name, camera_names)
    if preview_camera is not None:
        # The table card also includes one concrete video stream for quick visual
        # recognition of the sequence.
        preview_video_kind: str = preview_camera[0]
        preview_video_camera: str = preview_camera[1]
        video_origin: str = f"/{_catalog_cam_node(camera_names, preview_video_kind, preview_video_camera)}/pinhole"
        video_preview_view: rrb.Spatial2DView = rrb.Spatial2DView(
            origin=video_origin,
            name=f"{preview_video_kind} {preview_video_camera}",
            contents=f"{video_origin}/**",
        )
        preview_views.append(video_preview_view)

    # Keep table previews deterministic across datasets by selecting the first
    # ten seconds and asking Rerun to play that selected window in a loop. This
    # belongs on the TimePanel rather than the 3D view: a 3D VisibleTimeRange
    # switches the spatial view into range-query mode and stacks temporal hand
    # keypoints together.
    preview_start_time: rr.datatypes.TimeInt = rr.datatypes.TimeInt(seconds=TABLE_CARD_PREVIEW_START_SECONDS)
    preview_end_time: rr.datatypes.TimeInt = rr.datatypes.TimeInt(seconds=TABLE_CARD_PREVIEW_END_SECONDS)
    preview_time_selection: rrb.components.AbsoluteTimeRange = rrb.components.AbsoluteTimeRange(
        min=preview_start_time,
        max=preview_end_time,
    )
    table_preview_time_panel: rrb.TimePanel = rrb.TimePanel(
        timeline=timeline,
        play_state="playing",
        loop_mode="selection",
        time_selection=preview_time_selection,
    )

    return rrb.Blueprint(
        *preview_views,
        table_preview_time_panel,
        collapse_panels=True,
    )


def build_rrd_index_table_blueprint(dataset_name: str, *, timeline: str = "video_time") -> str:
    """Build a Rerun table blueprint with on-demand recording previews.

    Args:
        dataset_name: Dataset key used to select the table-card blueprint.
        timeline: Timeline name used by the preview views.

    Returns:
        Base64-encoded Rerun blueprint string suitable for Arrow schema metadata.

    Raises:
        RuntimeError: If the installed Rerun SDK does not support table blueprints.
    """
    experimental_api: Any | None = getattr(rrb, "experimental", None)
    table_blueprint_archetype: Any | None = getattr(experimental_api, "TableBlueprint", None)
    if table_blueprint_archetype is None:
        raise RuntimeError(
            "Experimental table blueprints require Rerun SDK 0.32 or newer. Run from the default Pixi environment."
        )

    blueprint: rrb.Blueprint = build_table_card_blueprint(dataset_name, timeline=timeline)
    blueprint_stream = RecordingStream._from_native(
        bindings.new_blueprint(
            application_id="embedded",
            make_default=False,
            make_thread_default=False,
            default_enabled=True,
        )
    )
    blueprint_stream.set_time("blueprint", sequence=0)
    blueprint._log_to_stream(blueprint_stream)
    blueprint_stream.log(
        "/table",
        table_blueprint_archetype(
            segment_preview_column="recording_uri",
            flag_column=MARKER_FLAG_COLUMN,
            grid_view_card_title="sequence_key",
            url_column="recording_uri",
        ),
    )

    # Rerun reads this base64 payload from Arrow schema metadata to configure table cards.
    rbl_bytes: bytes = blueprint_stream.memory_recording().drain_as_bytes()
    encoded_blueprint: str = base64.b64encode(rbl_bytes).decode("ascii")
    return f"base64:{encoded_blueprint}"


def build_rrd_index_table_schema(encoded_blueprint: str) -> pa.Schema:
    """Build the Arrow schema for the lightweight RRD index table.

    Args:
        encoded_blueprint: Base64 blueprint payload returned by
            ``build_rrd_index_table_blueprint``.

    Returns:
        Arrow schema with Rerun table index, flag column, and blueprint metadata.
    """
    return pa.schema(
        [
            pa.field("id", pa.int64(), metadata={rr.SORBET_IS_TABLE_INDEX: "true"}),
            pa.field("dataset", pa.utf8()),
            pa.field("sequence_key", pa.utf8()),
            pa.field("recording_uri", pa.utf8()),
            pa.field("path", pa.utf8()),
            pa.field("size_bytes", pa.int64()),
            pa.field(MARKER_FLAG_COLUMN, pa.bool_(), metadata={"rerun:is_flag_column": "true"}),
        ],
        metadata={TABLE_BLUEPRINT_METADATA_KEY: encoded_blueprint.encode("ascii")},
    )


def create_rrd_index_table(
    client: Any,
    *,
    dataset_name: str,
    table_name: str,
    rows: list[RRDIndexRow],
) -> Any:
    """Create or replace a lightweight RRD URL index table.

    Args:
        client: Rerun catalog client connected to the hosting server.
        dataset_name: Dataset key used to build the embedded table-card
            blueprint.
        table_name: Name of the catalog table to create.
        rows: Row records to append to the table.

    Returns:
        Created Rerun catalog table entry.

    Raises:
        RuntimeError: If the installed Rerun SDK does not support table blueprints.
    """
    existing_table_names: set[str] = set(client.table_names())
    if table_name in existing_table_names:
        client.get_table(table_name).delete()

    # The embedded TableBlueprint enables preview cards for each catalog row.
    # Creating these previews can dominate startup for large datasets, but they
    # are intentionally kept because the catalog is much less useful without
    # visual row previews.
    encoded_blueprint: str = build_rrd_index_table_blueprint(dataset_name)
    schema: pa.Schema = build_rrd_index_table_schema(encoded_blueprint)
    table = client.create_table(table_name, schema)
    table.append(
        id=[row.id for row in rows],
        dataset=[row.dataset for row in rows],
        sequence_key=[row.sequence_key for row in rows],
        recording_uri=[row.recording_uri for row in rows],
        path=[row.path for row in rows],
        size_bytes=[row.size_bytes for row in rows],
        marker_flag=[row.marker_flag for row in rows],
    )
    return table


def _shutdown_catalog_server(server: CatalogServer, *, timeout_seconds: float = CATALOG_SHUTDOWN_TIMEOUT_SECONDS) -> bool:
    """Shutdown a Rerun server without letting native teardown block forever.

    Args:
        server: Running Rerun catalog server.
        timeout_seconds: Maximum time to wait for graceful shutdown.

    Returns:
        True when shutdown completed before the timeout, false otherwise.

    Raises:
        RuntimeError: If the shutdown thread returns an exception.
    """
    shutdown_errors: list[BaseException] = []

    def shutdown() -> None:
        try:
            server.shutdown()
        except BaseException as exc:  # noqa: BLE001 - relay shutdown failures from the worker thread.
            shutdown_errors.append(exc)

    shutdown_thread: threading.Thread = threading.Thread(
        target=shutdown,
        name="rerun-catalog-shutdown",
        daemon=True,
    )
    shutdown_thread.start()
    try:
        shutdown_thread.join(timeout=timeout_seconds)
    except KeyboardInterrupt:
        return False
    if shutdown_thread.is_alive():
        return False
    if shutdown_errors:
        raise RuntimeError("Rerun catalog server shutdown failed.") from shutdown_errors[0]
    return True


def main(config: CatalogConfig) -> None:
    """Host a Rerun catalog for converted ExoEgo Forge RRD files.

    Args:
        config: Runtime configuration for the catalog server.
    """
    rrd_root: Path = config.rrd_root.expanduser().resolve()
    paths_by_dataset: dict[str, list[Path]] = discover_rrd_paths(rrd_root, datasets=config.datasets)
    dataset_names: list[str] = sorted(paths_by_dataset)

    server: CatalogServer | None = None
    client: Any | None = None
    try:
        server = mount_catalog(
            rrd_root,
            datasets=config.datasets,
            port=config.port,
            application_id=config.application_id,
            optimize_for_catalog=config.optimize_for_catalog,
            catalog_rrd_cache_dir=config.catalog_rrd_cache_dir,
            optimize_datasets=config.optimize_datasets,
        )
        client = server.client()
        table_urls_by_name: dict[str, str] = {}
        catalog_url: str = server.url()
        for dataset_name in dataset_names:
            dataset_dir: Path = rrd_root / dataset_name
            dataset_entry = client.get_dataset(dataset_name)
            rows: list[RRDIndexRow] = build_rrd_index_rows_from_dataset(
                dataset_entry,
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
            )
            table_name: str = table_name_for_dataset(dataset_name)
            total_size_bytes: int = sum(row.size_bytes for row in rows)
            print(f"Creating {table_name} ({len(rows)} RRDs, {total_size_bytes:,} bytes).", flush=True)
            table = create_rrd_index_table(
                client,
                dataset_name=dataset_name,
                table_name=table_name,
                rows=rows,
            )
            table_urls_by_name[table_name] = f"{catalog_url}/entry/{table.id}"

        print()
        print("-" * 72)
        print(f"  Catalog URL:  {catalog_url}")
        print()
        print("  Tables:")
        for table_name, table_url in table_urls_by_name.items():
            print(f"    {table_name}: {table_url}")
        print()
        print("  In the Rerun viewer: + -> Open Data Source -> paste the URL")
        print("  Open a table with:")
        print("    pixi run rerun <table-url>")
        print()
        print("  Enable: Settings > Experimental > Table cards and blueprints")
        print("-" * 72, flush=True)

        if config.open_browser:
            rr.serve_web_viewer(web_port=config.web_port, open_browser=True, connect_to=catalog_url)
            print(f"\nWeb viewer hosted at http://127.0.0.1:{config.web_port}")

        print("\nServer is up. Ctrl-C to stop.", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("shutting down", flush=True)
    finally:
        client = None
        if server is not None:
            shutdown_completed: bool = _shutdown_catalog_server(server)
            if not shutdown_completed:
                print(
                    f"Rerun catalog server did not shut down within {CATALOG_SHUTDOWN_TIMEOUT_SECONDS:.1f}s; "
                    "forcing process exit.",
                    flush=True,
                )
                os._exit(130)
