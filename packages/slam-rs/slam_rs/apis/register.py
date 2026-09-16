"""Register downloaded MSD recording layers and blueprints into a catalog."""

from dataclasses import dataclass
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from slam_rs.config import load_slam_config


@dataclass(slots=True)
class Config:
    """Register downloaded recordings into a Rerun catalog."""

    root: Path = Path("data/msd-rrd")
    """Directory containing recording layers and blueprints."""
    catalog: str | None = None
    """Catalog URL; defaults to slam.toml."""
    replace: bool = False
    """Replace registered segment layers instead of skipping duplicates."""


def register(root: Path, client: CatalogClient, replace: bool = False) -> dict[str, dict[str, int]]:
    """Register files by dataset prefix and layer, returning registered file counts."""
    paths_by_dataset: dict[str, dict[str, list[Path]]] = {}
    for layer in ("base", "gt", "sensor_metadata"):
        for path in sorted((root / layer).glob("*__*.rrd")):
            if path.is_file():
                name: str = path.name.split("__", 1)[0]
                paths_by_dataset.setdefault(name, {}).setdefault(layer, []).append(path)
    if not any("base" in layers for layers in paths_by_dataset.values()):
        raise FileNotFoundError(f"no base-layer rrds under {root / 'base'}")

    counts: dict[str, dict[str, int]] = {}
    on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE if replace else OnDuplicateSegmentLayer.SKIP
    for name, layers in paths_by_dataset.items():
        entry: DatasetEntry = client.create_dataset(name, exist_ok=True)
        counts[name] = {}
        for layer, paths in layers.items():
            entry.register([path.resolve().as_uri() for path in paths], layer_name=layer, on_duplicate=on_duplicate).wait()
            counts[name][layer] = len(paths)
        blueprint: Path = root / "blueprints" / f"{name}.rbl"
        if blueprint.is_file() and entry.default_blueprint() is None:
            entry.register_blueprint(blueprint.resolve().as_uri(), set_default=True)
        table: Path = root / "blueprints" / f"{name}-table.rbl"
        if table.is_file() and entry.default_segment_table_blueprint() is None:
            entry.register_blueprint(table.resolve().as_uri(), segment_table=True)
    return counts


def main(config: Config) -> None:
    """Connect to the catalog and register the downloaded recordings."""
    catalog: str = config.catalog if config.catalog is not None else load_slam_config().catalog_url
    client: CatalogClient = CatalogClient(catalog)
    counts: dict[str, dict[str, int]] = register(config.root, client, config.replace)
    summary: str = "; ".join(f"{name}: " + ", ".join(f"{count} {layer}" for layer, count in layers.items()) for name, layers in counts.items())
    print(f"registered {summary} at {catalog}")
