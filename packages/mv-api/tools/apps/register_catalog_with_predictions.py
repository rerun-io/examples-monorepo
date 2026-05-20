"""Mount the full ExoEgo catalog and register existing prediction RRD layers.

This script intentionally avoids importing MVAPI inference code. It is only a
small catalog restore/serve entrypoint around SimpleCV's catalog mounting API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import tyro

DEFAULT_RRD_ROOT: Path = Path("/mnt/8tb/data/exoego-forge-catalog")
"""Default root containing the source ExoEgo Forge catalog RRDs."""
DEFAULT_PREDICTION_ROOT: Path = Path("/mnt/8tb/data/exoego-forge-catalog-predictions")
"""Default root containing catalog prediction-layer RRDs."""
DEFAULT_CATALOG_PORT: int = 9991
"""Default local Rerun catalog server port."""
DEFAULT_APPLICATION_ID: str = "exoego-forge"
"""Application id used by SimpleCV's registered catalog blueprints."""
DEFAULT_CATALOG_RRD_CACHE_DIR: Path = Path("~/.cache/simplecv/exoego-forge-catalog-optimized")
"""Default SimpleCV cache root for catalog-compatible optimized RRD copies."""


@dataclass(frozen=True, slots=True)
class CatalogRegistrationConfig:
    """Configuration for mounting a full catalog with prediction layers."""

    rrd_root: Path = DEFAULT_RRD_ROOT
    """Root containing source ``<dataset>/**/*.rrd`` files."""
    prediction_root: Path = DEFAULT_PREDICTION_ROOT
    """Root containing prediction ``<dataset>/**/*.rrd`` files."""
    prediction_rrds: Annotated[tuple[Path, ...], tyro.conf.arg(name="prediction-rrd")] = ()
    """Explicit prediction RRDs to register. Empty means discover all under ``prediction_root``."""
    datasets: Annotated[tuple[str, ...], tyro.conf.arg(name="dataset")] = ()
    """Dataset directories to mount. Empty means scan every first-level source catalog directory."""
    prediction_dataset_name: Annotated[str | None, tyro.conf.arg(name="prediction-dataset")] = None
    """Optional dataset name override for explicit prediction RRDs outside ``prediction_root``."""
    layer_name: str | None = None
    """Optional catalog layer name override. Empty means use each prediction RRD stem."""
    port: int = DEFAULT_CATALOG_PORT
    """Local Rerun catalog server port."""
    application_id: str = DEFAULT_APPLICATION_ID
    """Application id passed to SimpleCV when registering default blueprints."""
    optimize_for_catalog: bool = False
    """Whether SimpleCV should register catalog-optimized source RRD copies."""
    catalog_rrd_cache_dir: Path | None = DEFAULT_CATALOG_RRD_CACHE_DIR
    """Optional SimpleCV optimized RRD cache root."""
    optimize_datasets: Annotated[tuple[str, ...] | None, tyro.conf.arg(name="optimize-dataset")] = None
    """Optional dataset names to optimize. ``None`` preserves SimpleCV's default."""
    show_progress: Annotated[bool, tyro.conf.arg(name="progress")] = True
    """Whether SimpleCV should show registration progress bars."""
    exit_after_register: bool = False
    """Exit after registering layers instead of keeping the catalog server alive."""


@dataclass(frozen=True, slots=True)
class RegisteredPrediction:
    """One prediction RRD registered into a catalog dataset."""

    dataset_name: str
    """Catalog dataset that received the prediction layer."""
    layer_name: str
    """Catalog layer name used for registration."""
    rrd_path: Path
    """Prediction RRD path registered with the catalog."""


@dataclass(frozen=True, slots=True)
class RegisteredReviewTable:
    """One dataset review table registered into the catalog."""

    dataset_name: str
    """Source catalog dataset represented by the table."""
    table_name: str
    """Catalog table name."""
    table_url: str
    """Rerun catalog URL for opening the table."""
    row_count: int
    """Number of source RRD rows in the table."""


@dataclass(frozen=True, slots=True)
class CatalogRegistrationResult:
    """Running catalog server plus prediction registration summary."""

    catalog_url: str
    """Rerun catalog URL exposed by the running server."""
    registered_review_tables: list[RegisteredReviewTable]
    """Dataset review/index tables registered into the running catalog."""
    registered_predictions: list[RegisteredPrediction]
    """Prediction layers registered into the running catalog."""
    server: Any
    """Running SimpleCV/Rerun catalog server object."""


def discover_prediction_rrd_paths(config: CatalogRegistrationConfig) -> list[Path]:
    """Discover prediction RRDs that should be registered."""
    if config.prediction_rrds:
        explicit_paths: list[Path] = []
        for prediction_rrd in config.prediction_rrds:
            resolved_path: Path = prediction_rrd.expanduser().resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Prediction RRD does not exist: {resolved_path}")
            if resolved_path.suffix != ".rrd":
                raise ValueError(f"Prediction path is not an .rrd file: {resolved_path}")
            explicit_paths.append(resolved_path)
        return explicit_paths

    prediction_root: Path = config.prediction_root.expanduser().resolve()
    if not prediction_root.is_dir():
        raise FileNotFoundError(f"Prediction root directory does not exist: {prediction_root}")

    discovered_paths: list[Path] = sorted(path.resolve() for path in prediction_root.rglob("*.rrd") if path.is_file())
    if not discovered_paths:
        raise FileNotFoundError(f"No prediction .rrd files found under {prediction_root}")
    return discovered_paths


def infer_prediction_dataset_name(
    *,
    prediction_rrd: Path,
    prediction_root: Path,
    override: str | None,
) -> str:
    """Infer the catalog dataset name for one prediction RRD."""
    if override is not None:
        return override

    prediction_root_resolved: Path = prediction_root.expanduser().resolve()
    prediction_rrd_resolved: Path = prediction_rrd.expanduser().resolve()
    try:
        relative_path: Path = prediction_rrd_resolved.relative_to(prediction_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Cannot infer dataset for {prediction_rrd_resolved}; pass --prediction-dataset or place it under {prediction_root_resolved}."
        ) from exc

    if len(relative_path.parts) < 2:
        raise ValueError(f"Prediction RRD must live under <prediction-root>/<dataset>/..., got {prediction_rrd_resolved}")
    dataset_name: str = relative_path.parts[0]
    return dataset_name


def register_prediction_rrds(*, client: Any, config: CatalogRegistrationConfig, prediction_rrds: list[Path]) -> list[RegisteredPrediction]:
    """Register prediction RRDs into the matching catalog datasets."""
    registered_predictions: list[RegisteredPrediction] = []
    for prediction_rrd in prediction_rrds:
        dataset_name: str = infer_prediction_dataset_name(
            prediction_rrd=prediction_rrd,
            prediction_root=config.prediction_root,
            override=config.prediction_dataset_name,
        )
        layer_name: str = config.layer_name if config.layer_name is not None else prediction_rrd.stem
        dataset_entry: Any = client.get_dataset(dataset_name)
        registration_handle: Any = dataset_entry.register([prediction_rrd.resolve().as_uri()], layer_name=layer_name)
        wait = getattr(registration_handle, "wait", None)
        if callable(wait):
            wait()
        registered_predictions.append(
            RegisteredPrediction(
                dataset_name=dataset_name,
                layer_name=layer_name,
                rrd_path=prediction_rrd,
            )
        )
    return registered_predictions


def register_dataset_review_tables(
    *,
    client: Any,
    config: CatalogRegistrationConfig,
    rrd_root: Path,
    catalog_url: str,
) -> list[RegisteredReviewTable]:
    """Create SimpleCV's dataset review tables with embedded table-card blueprints."""
    from simplecv.apis.exoego_forge_catalog import (
        build_rrd_index_rows_from_dataset,
        create_rrd_index_table,
        discover_rrd_paths,
        table_name_for_dataset,
    )

    paths_by_dataset: dict[str, list[Path]] = discover_rrd_paths(rrd_root, datasets=config.datasets)
    registered_tables: list[RegisteredReviewTable] = []
    for dataset_name in sorted(paths_by_dataset):
        dataset_dir: Path = rrd_root / dataset_name
        dataset_entry: Any = client.get_dataset(dataset_name)
        rows: list[Any] = build_rrd_index_rows_from_dataset(
            dataset_entry,
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
        )
        table_name: str = table_name_for_dataset(dataset_name)
        total_size_bytes: int = sum(row.size_bytes for row in rows)
        print(f"Creating {table_name} ({len(rows)} RRDs, {total_size_bytes:,} bytes).", flush=True)
        table: Any = create_rrd_index_table(
            client,
            dataset_name=dataset_name,
            table_name=table_name,
            rows=rows,
        )
        registered_tables.append(
            RegisteredReviewTable(
                dataset_name=dataset_name,
                table_name=table_name,
                table_url=f"{catalog_url}/entry/{table.id}",
                row_count=len(rows),
            )
        )
    return registered_tables


def mount_catalog_with_predictions(config: CatalogRegistrationConfig) -> CatalogRegistrationResult:
    """Mount the full SimpleCV catalog and register prediction RRD layers."""
    from simplecv.apis.exoego_forge_catalog import mount_catalog

    rrd_root: Path = config.rrd_root.expanduser().resolve()
    mount_kwargs: dict[str, Any] = {
        "datasets": config.datasets,
        "port": config.port,
        "application_id": config.application_id,
        "show_progress": config.show_progress,
        "optimize_for_catalog": config.optimize_for_catalog,
    }
    if config.catalog_rrd_cache_dir is not None:
        mount_kwargs["catalog_rrd_cache_dir"] = config.catalog_rrd_cache_dir.expanduser().resolve()
    if config.optimize_datasets is not None:
        mount_kwargs["optimize_datasets"] = config.optimize_datasets

    server: Any = mount_catalog(rrd_root, **mount_kwargs)
    try:
        client: Any = server.client()
        catalog_url: str = str(server.url())
        registered_review_tables: list[RegisteredReviewTable] = register_dataset_review_tables(
            client=client,
            config=config,
            rrd_root=rrd_root,
            catalog_url=catalog_url,
        )
        prediction_rrds: list[Path] = discover_prediction_rrd_paths(config)
        registered_predictions: list[RegisteredPrediction] = register_prediction_rrds(
            client=client,
            config=config,
            prediction_rrds=prediction_rrds,
        )
        return CatalogRegistrationResult(
            catalog_url=catalog_url,
            registered_review_tables=registered_review_tables,
            registered_predictions=registered_predictions,
            server=server,
        )
    except BaseException:
        shutdown = getattr(server, "shutdown", None)
        if callable(shutdown):
            shutdown()
        raise


def print_summary(result: CatalogRegistrationResult, *, config: CatalogRegistrationConfig) -> None:
    """Print the catalog URL and registered prediction layers."""
    dataset_scope: str = "all first-level dataset directories" if config.datasets == () else ", ".join(config.datasets)
    print()
    print("-" * 72)
    print(f"Catalog URL: {result.catalog_url}")
    print(f"Catalog scope: {dataset_scope}")
    print()
    print("Dataset review tables:")
    for registered_table in result.registered_review_tables:
        print(f"  {registered_table.table_name}: {registered_table.table_url} ({registered_table.row_count} rows)")
    print()
    print("Registered prediction layers:")
    for registered_prediction in result.registered_predictions:
        print(
            f"  {registered_prediction.dataset_name}: {registered_prediction.layer_name} "
            f"from {registered_prediction.rrd_path}"
        )
    print()
    print("SimpleCV registered default segment blueprints and dataset review table blueprints.")
    print("Enable: Settings > Experimental > Table cards and blueprints")
    if config.exit_after_register:
        print("Exiting now; in-memory catalog registration will not remain served.")
    else:
        print("Server is up. Ctrl-C to stop.")
    print("-" * 72, flush=True)


def main(config: CatalogRegistrationConfig) -> None:
    """CLI entrypoint."""
    result: CatalogRegistrationResult = mount_catalog_with_predictions(config)
    try:
        print_summary(result, config=config)
        if config.exit_after_register:
            return
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("shutting down", flush=True)
    finally:
        shutdown = getattr(result.server, "shutdown", None)
        if callable(shutdown):
            shutdown()


if __name__ == "__main__":
    main(
        tyro.cli(
            CatalogRegistrationConfig,
            description="Mount the full SimpleCV ExoEgo catalog and register existing prediction RRD layers.",
        )
    )
