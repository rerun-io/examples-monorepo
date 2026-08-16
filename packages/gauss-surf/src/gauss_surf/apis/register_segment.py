"""Seed a development catalog with one ARKitScenes segment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from arkitscenes_download.ingest.blueprint import MeshFraming, make_blueprint
from arkitscenes_download.ingest.paths import PROMPTDA_MESH
from arkitscenes_download.schema import ALL_LAYER_NAMES
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from rerun.experimental import RrdReader
from simplecv.rerun_log_utils import mesh_bounding_geometry

from gauss_surf.catalog import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, connect_catalog, register_layer
from gauss_surf.contracts import LAYERS, PROMPTDA_LAYER

DEFAULT_DATASET_ROOT: Path = Path("/mnt/nas/datasets/arkitscenes/arkitscenes.2026.07.22")
"""NAS root containing layer-major ARKitScenes RRDs."""
V2_BLUEPRINT_FILENAMES: tuple[str, str] = ("arkitscenes-v2-landscape.rbl", "arkitscenes-v2-portrait.rbl")
"""The two v2 dataset blueprints used by the development catalog."""
EXTENDED_BLUEPRINT_FILENAMES: tuple[str, str] = (
    "arkitscenes-v2-gauss-surf-landscape.rbl",
    "arkitscenes-v2-gauss-surf-portrait.rbl",
)
"""Generated PromptDA + gauss-surf blueprint pair."""


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for reseeding one development-catalog segment."""

    video_id: str
    """ARKitScenes segment identifier to register."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """Target Rerun catalog server URL."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Target dataset, created when absent."""
    dataset_root: Path = DEFAULT_DATASET_ROOT
    """Layer-major root containing ``<layer>/<video_id>.rrd`` files."""
    blueprint_dir: Path = DEFAULT_DATASET_ROOT / "blueprints"
    """Directory containing the two NAS ``arkitscenes-v2`` blueprint files used when generation is disabled."""
    build_extended_blueprints: bool = True
    """Build and register the PromptDA + gauss-surf blueprint pair instead of the NAS pair."""
    extended_blueprint_dir: Path = Path("data/blueprints")
    """Local output directory for generated extended blueprint files."""
    extra_rrds: list[Path] = field(default_factory=list)
    """Optional derived-layer RRDs; each layer name is inferred from its parent directory."""
    recover_derived_layers: bool = False
    """Register the complete local derived-layer recovery list in addition to standard layers."""
    default_orientation: Literal["landscape", "portrait"] = "landscape"
    """Which blueprint of the pair opens by default; match the target segment's orientation."""


def promptda_mesh_framing(promptda_rrd_path: Path) -> MeshFraming:
    """Compute the scene's AABB center and bounding radius from the registered PromptDA mesh.

    Args:
        promptda_rrd_path: PromptDA layer RRD containing the TSDF-fused mesh.

    Returns:
        World-frame AABB center and bounding-sphere radius in metres.
    """
    reader: RrdReader = RrdReader(str(promptda_rrd_path))
    for chunk in reader.stream():
        if str(chunk.entity_path).lstrip("/") != PROMPTDA_MESH:
            continue
        batch = chunk.to_record_batch()
        for column_name in batch.schema.names:
            if "vertex_positions" not in column_name:
                continue
            cell = batch.column(column_name)[0]
            flat = cell.values.flatten() if hasattr(cell.values, "flatten") else cell.values
            vertices_n3: Float64[ndarray, "n 3"] = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
            center_3: Float64[ndarray, "3"]
            radius_m: float
            center_3, radius_m = mesh_bounding_geometry(vertices_n3)
            return center_3, radius_m
    raise ValueError(f"{promptda_rrd_path} has no {PROMPTDA_MESH} vertex positions to frame")


def write_extended_blueprints(output_dir: Path, dataset_name: str, framing: MeshFraming | None = None) -> list[Path]:
    """Write the landscape and portrait PromptDA + gauss-surf blueprint artifacts.

    Args:
        output_dir: Directory for the generated ``.rbl`` files.
        dataset_name: Filename prefix and dataset identity.
        framing: Optional scene AABB center and bounding radius for the 3D eye controls.

    Returns:
        Landscape path followed by portrait path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, portrait in zip(EXTENDED_BLUEPRINT_FILENAMES, (False, True), strict=True):
        path: Path = output_dir / filename.replace("arkitscenes-v2", dataset_name, 1)
        make_blueprint(portrait=portrait, framing=framing, include_promptda=True, include_gauss_surf=True).save("arkitscenes", path)
        path.chmod(0o644)
        paths.append(path)
    return paths


def recovery_rrds(video_id: str) -> list[Path]:
    """Resolve the complete local derived-layer recovery list for one segment."""
    return [Path(pattern.format(video_id=video_id)) for pattern in LAYERS.values()]


def existing_recovery_rrds(video_id: str) -> list[Path]:
    """Resolve the derived-layer recovery paths that have already been generated."""
    return [path for path in recovery_rrds(video_id) if path.is_file()]


def main(config: Config) -> None:
    """Register ten standard layers, extras, and one replaceable blueprint pair."""
    standard_paths: dict[str, Path] = {
        layer_name: config.dataset_root / layer_name / f"{config.video_id}.rrd" for layer_name in ALL_LAYER_NAMES
    }
    missing_paths: list[Path] = [path for path in standard_paths.values() if not path.is_file()]
    extra_rrds: list[Path] = list(
        dict.fromkeys([*(existing_recovery_rrds(config.video_id) if config.recover_derived_layers else []), *config.extra_rrds])
    )
    missing_extras: list[Path] = [path for path in extra_rrds if not path.is_file()]
    if missing_paths or missing_extras:
        missing_lines: str = "\n".join(f"  {path}" for path in [*missing_paths, *missing_extras])
        raise SystemExit(f"required RRD files are missing:\n{missing_lines}")
    if config.build_extended_blueprints:
        promptda_candidates: list[Path] = [
            *[path for path in extra_rrds if path.parent.name == PROMPTDA_LAYER],
            Path(LAYERS[PROMPTDA_LAYER].format(video_id=config.video_id)),
        ]
        framing: MeshFraming | None = next(
            (promptda_mesh_framing(path) for path in promptda_candidates if path.is_file()),
            None,
        )
        blueprint_paths: list[Path] = write_extended_blueprints(config.extended_blueprint_dir, config.dataset_name, framing)
    else:
        blueprint_paths = [config.blueprint_dir / filename for filename in V2_BLUEPRINT_FILENAMES]
    missing_blueprints: list[Path] = [path for path in blueprint_paths if not path.is_file()]
    if missing_blueprints:
        missing_lines = "\n".join(f"  {path}" for path in missing_blueprints)
        raise SystemExit(f"required blueprint files are missing:\n{missing_lines}")

    extra_layers: dict[str, Path] = {}
    extra_path: Path
    for extra_path in extra_rrds:
        # Layer-major trees name the parent after the layer; per-segment trees
        # (data/ultrawide_signals/<video_id>/<layer>.rrd) name the file instead.
        layer_name: str = extra_path.stem if extra_path.parent.name == config.video_id else extra_path.parent.name
        if layer_name in standard_paths or layer_name in extra_layers:
            raise SystemExit(f"duplicate or reserved extra layer name {layer_name!r} inferred from {extra_path}")
        extra_layers[layer_name] = extra_path

    client: CatalogClient = connect_catalog(config.catalog_url, config.dataset_name, create_missing=True)
    dataset: DatasetEntry = client.get_dataset(config.dataset_name)
    layer_name: str
    rrd_path: Path
    for layer_name, rrd_path in (*standard_paths.items(), *extra_layers.items()):
        register_layer(dataset, rrd_path, layer_name)
        print(f"registered {layer_name}: {rrd_path}")

    existing_blueprints: list[str] = dataset.blueprints()
    if existing_blueprints and len(existing_blueprints) != len(blueprint_paths):
        raise RuntimeError(
            f"dataset already has {len(existing_blueprints)} blueprints; expected zero or {len(blueprint_paths)}, refusing to add duplicates"
        )
    if existing_blueprints:
        blueprint_dataset: DatasetEntry | None = dataset.blueprint_dataset()
        if blueprint_dataset is None:
            raise RuntimeError("dataset reports blueprints but has no blueprint dataset")
        print(f"replacing blueprint ids: {', '.join(existing_blueprints)}")
        blueprint_dataset.unregister(segments_to_drop=existing_blueprints, layers_to_drop=[]).wait()
    landscape_is_default: bool = config.default_orientation == "landscape"
    dataset.register_blueprint(blueprint_paths[0].resolve().as_uri(), set_default=landscape_is_default)
    dataset.register_blueprint(blueprint_paths[1].resolve().as_uri(), set_default=not landscape_is_default)
    print(f"registered blueprints: {blueprint_paths[0].name}, {blueprint_paths[1].name} (default: {config.default_orientation})")
    print(
        f"segment {config.video_id}: cleanly replaced {len(standard_paths) + len(extra_layers)} layers "
        f"in dataset {config.dataset_name!r}"
    )
