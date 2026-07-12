"""Register ingested ARKitScenes RRDs into a local Rerun catalog.

The local catalog is the in-memory ``rerun server`` (start it with
``pixi run serve``). Each sequence RRD is one segment whose id is its
``recording_id`` (the ARKitScenes video id); the shared layout is registered
as the dataset's default blueprint.
"""

from dataclasses import dataclass
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from arkitscenes_download.ingest.blueprint import make_blueprint
from arkitscenes_download.ingest.layers import LAYER_NAMES

DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
"""gRPC URL of a locally-running ``rerun server`` catalog."""


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for registering ingested sequences into a catalog."""

    rrd_dir: Path = Path("data/rrd")
    """Root directory holding ``<video_id>/<layer>.rrd`` files."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the local Rerun catalog (``rerun server``)."""
    dataset_name: str = "arkitscenes"
    """Catalog dataset name to (re)create."""
    recreate: bool = False
    """Delete and recreate the dataset before registering."""
    video_ids: list[str] | None = None
    """Optional sequence subset to register from the RRD root."""


def register_sequences(config: Config) -> DatasetEntry:
    """Create (or replace) the dataset and register every RRD in ``rrd_dir``."""
    layer_paths: dict[str, list[Path]]
    if config.video_ids is None:
        layer_paths = {layer_name: sorted(config.rrd_dir.glob(f"*/{layer_name}.rrd")) for layer_name in LAYER_NAMES}
    else:
        layer_paths = {
            layer_name: [path for video_id in config.video_ids if (path := config.rrd_dir / video_id / f"{layer_name}.rrd").is_file()]
            for layer_name in LAYER_NAMES
        }
    if not any(layer_paths.values()):
        raise FileNotFoundError(f"no .rrd files found in {config.rrd_dir}")

    client: CatalogClient = CatalogClient(config.catalog_url)
    if config.recreate and config.dataset_name in client.dataset_names():
        client.get_dataset(config.dataset_name).delete()
    dataset: DatasetEntry = client.create_dataset(config.dataset_name, exist_ok=True)

    for layer_name, paths in layer_paths.items():
        if paths:
            dataset.register(
                [path.resolve().as_uri() for path in paths],
                layer_name=layer_name,
                on_duplicate=OnDuplicateSegmentLayer.REPLACE,
            ).wait()

    # Most ARKitScenes captures are handheld portrait scans, so the portrait
    # layout is the dataset default; landscape stays selectable in the viewer.
    for orientation, is_portrait in (("landscape", False), ("portrait", True)):
        blueprint_path: Path = config.rrd_dir / f"{config.dataset_name}-{orientation}.rbl"
        make_blueprint(is_portrait).save(f"arkitscenes-{orientation}", blueprint_path)
        dataset.register_blueprint(blueprint_path.resolve().as_uri(), set_default=is_portrait)
    return dataset


def main() -> None:
    """CLI entry point: register all ingested sequences."""
    config: Config = tyro.cli(Config)
    dataset: DatasetEntry = register_sequences(config)
    segment_ids: list[str] = [str(segment) for segment in dataset.segment_ids()]
    print(f"dataset '{config.dataset_name}' at {config.catalog_url}: {len(segment_ids)} segments")
    for segment_id in segment_ids:
        print(f"  {segment_id}")


if __name__ == "__main__":
    main()
