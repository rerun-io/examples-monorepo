"""Utilities for registering ExoEgo Forge RRD files into a Rerun catalog.

The flow is **two-tier**: a separate ``rerun server`` (tier 1) plus the re-runnable
client step ``RegisterConfig`` / ``register_main`` (tier 2) that attaches converted
recordings as catalog datasets with a default per-segment blueprint.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import rerun.blueprint as rrb

from simplecv.apis.view_exoego import create_container
from simplecv.rig import entity_id

APPLICATION_ID: str = "exoego-forge"
"""Rerun application id used by converted ExoEgo Forge recordings."""

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


def _blueprint_cam_node(
    camera_names: dict[str, tuple[str, ...]], kind: str, camera_name: str, *, layout: Literal["flat", "rig"]
) -> Path:
    """Return a catalog camera's video-parent entity node for the recording layout.

    ``rig`` (``exoego:v2``) maps to ``/world/rig_NN/cam_MM`` via ``_catalog_cam_node``;
    ``flat`` (legacy v1) is simply ``/world/<kind>/<camera_name>``.
    """
    if layout == "rig":
        return _catalog_cam_node(camera_names, kind, camera_name)
    return Path("world") / kind / camera_name


def build_exoego_catalog_blueprint(dataset_name: str, *, layout: Literal["flat", "rig"] = "rig") -> rrb.Blueprint:
    """Build the default catalog blueprint for one ExoEgo Forge dataset.

    Args:
        dataset_name: Dataset key used to select known ego and exo camera names.
        layout: Entity layout of the recordings the blueprint targets. ``"rig"``
            (``exoego:v2``, cameras at ``/world/rig_NN/cam_MM``) for migrated
            recordings; ``"flat"`` (legacy v1 catalog, ``/world/{exo,ego}/<name>``).
            The video entity paths must match the recording or 2D panels render blank.

    Returns:
        Rerun blueprint used as the default view when opening a dataset segment.
    """
    camera_names: dict[str, tuple[str, ...]] = CATALOG_CAMERA_NAMES.get(dataset_name, {"ego": (), "exo": ()})
    # Map each known camera to its video entity node for the recording's layout, and
    # feed the human names through so 2D panels carry the skip list + readable titles
    # (matching direct view_exoego viewing).
    ego_video_log_paths: list[Path] = []
    exo_video_log_paths: list[Path] = []
    video_path_to_name: dict[Path, str] = {}
    for kind, video_log_paths in (("ego", ego_video_log_paths), ("exo", exo_video_log_paths)):
        for name in camera_names[kind]:
            cam_node: Path = _blueprint_cam_node(camera_names, kind, name, layout=layout)
            video_path: Path = cam_node / "pinhole" / "video"
            video_log_paths.append(video_path)
            video_path_to_name[video_path] = name
    container: rrb.ContainerLike = create_container(
        ego_video_log_paths=ego_video_log_paths,
        exo_video_log_paths=exo_video_log_paths,
        skip_camera_names=frozenset(),
        video_path_to_name=video_path_to_name,
    )
    return rrb.Blueprint(container, collapse_panels=True)


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


@dataclass
class RegisterConfig:
    """Config for tier 2: register ExoEgo Forge RRD roots into a running catalog."""

    rrd_root: Path = Path("data/exoego-forge-catalog")
    """Directory containing ``<dataset>/**/*.rrd`` files to register."""
    catalog_url: str = "rerun+http://127.0.0.1:9988"
    """URL of the running catalog server (the ``simplecv-catalog-serve`` task / ``rerun server``)."""
    datasets: tuple[str, ...] = DEFAULT_CATALOG_DATASETS
    """Dataset directories to register. Empty tuple scans all first-level directories."""
    layout: Literal["flat", "rig"] = "rig"
    """Entity layout of the recordings under ``rrd_root``, used to build each dataset's
    default blueprint. ``rig`` for exoego:v2 roots; ``flat`` for the legacy v1 root."""
    suffix: str = ""
    """Entry-name suffix. The rig root must register as ``-rig`` so its recording_ids do
    not collide with the v1 root registered under the same dataset names."""
    application_id: str = APPLICATION_ID
    """Application id used to save default dataset blueprints. Must match converted RRDs."""
    recreate: bool = True
    """Delete and recreate each dataset entry before registering. Pass ``--no-recreate`` to
    re-register onto existing entries (REPLACE per segment layer)."""


def register_main(config: RegisterConfig) -> None:
    """Register local ExoEgo Forge RRD roots into a running catalog (tier 2).

    Mirrors the cloud ingestion flow: one catalog dataset per source, every RRD attached
    under ``layer_name="base"``, plus a default blueprint. Idempotent (REPLACE), so the
    v1 and rig roots register into one live catalog without a restart. Registers the raw
    RRDs directly: the upstream footer-first store-id enumeration (rerun-io/reality#2496)
    makes register fast without a ``rerun rrd optimize`` rewrite.

    Args:
        config: Registration configuration.
    """
    from rerun.catalog import CatalogClient, OnDuplicateSegmentLayer

    rrd_root: Path = config.rrd_root.expanduser().resolve()
    uris_by_dataset: dict[str, list[str]] = discover_rrd_uris(rrd_root, datasets=config.datasets)

    client: Any = CatalogClient(config.catalog_url)
    existing_names: set[str] = set(client.dataset_names())
    total_files: int = sum(len(uris) for uris in uris_by_dataset.values())
    print(
        f"Registering {total_files} RRDs from {rrd_root} into {config.catalog_url} "
        f"(layout={config.layout}, suffix={config.suffix or '<none>'}).",
        flush=True,
    )

    for dataset_name in sorted(uris_by_dataset):
        uris: list[str] = uris_by_dataset[dataset_name]
        entry_name: str = f"{dataset_name}{config.suffix}"
        if config.recreate and entry_name in existing_names:
            client.get_dataset(entry_name).delete()
        dataset_entry: Any = client.create_dataset(entry_name, exist_ok=True)
        print(f"  {entry_name}: registering {len(uris)} RRD(s)...", flush=True)
        registration_handle: Any = dataset_entry.register(
            uris, layer_name="base", on_duplicate=OnDuplicateSegmentLayer.REPLACE
        )
        registration_handle.wait()

        # Install the per-dataset default blueprint. register_blueprint uploads the .rbl
        # synchronously, so a with-scoped temp dir (removed at the end of each iteration) is
        # all this one-shot client step needs.
        blueprint: rrb.Blueprint = build_exoego_catalog_blueprint(dataset_name, layout=config.layout)
        with tempfile.TemporaryDirectory(prefix=f"{entry_name}-") as tmp_name:
            blueprint_path: Path = Path(tmp_name) / f"{entry_name}.rbl"
            blueprint.save(config.application_id, path=str(blueprint_path))
            dataset_entry.register_blueprint(blueprint_path.resolve().as_uri(), set_default=True)

    print("\nRegistration complete.", flush=True)
