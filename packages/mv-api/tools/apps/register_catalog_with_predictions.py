"""Register existing MVAPI prediction RRD layers into a running Rerun catalog.

This script intentionally avoids importing MVAPI inference code. It connects to a
catalog server that is already serving the base ExoEgo Forge catalog (start it with
the ``simplecv-catalog-serve`` task and register the base recordings with
``simplecv-catalog-register``) and attaches each prediction RRD as a segment layer,
replacing any existing layer of the same name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import tyro

DEFAULT_PREDICTION_ROOT: Path = Path("/mnt/8tb/data/exoego-forge-catalog-predictions")
"""Default root containing catalog prediction-layer RRDs."""


@dataclass(frozen=True, slots=True)
class CatalogRegistrationConfig:
    """Configuration for registering prediction-layer RRDs into a running catalog."""

    catalog_url: str = "rerun+http://127.0.0.1:9988"
    """URL of the running Rerun catalog server. Start it with the ``simplecv-catalog-serve``
    task and register the base recordings with ``simplecv-catalog-register``."""
    prediction_root: Path = DEFAULT_PREDICTION_ROOT
    """Root containing prediction ``<dataset>/**/*.rrd`` files."""
    prediction_rrds: Annotated[tuple[Path, ...], tyro.conf.arg(name="prediction-rrd")] = ()
    """Explicit prediction RRDs to register. Empty means discover all under ``prediction_root``."""
    prediction_dataset_name: Annotated[str | None, tyro.conf.arg(name="prediction-dataset")] = None
    """Optional dataset name override for explicit prediction RRDs outside ``prediction_root``."""
    layer_name: str | None = None
    """Optional catalog layer name override. Empty means use each prediction RRD stem."""


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
class CatalogRegistrationResult:
    """Prediction registration summary."""

    catalog_url: str
    """Rerun catalog URL the prediction layers were registered into."""
    registered_predictions: list[RegisteredPrediction]
    """Prediction layers registered into the running catalog."""


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
    """Register prediction RRDs into the matching catalog datasets, replacing duplicate layers."""
    from rerun.catalog import OnDuplicateSegmentLayer

    registered_predictions: list[RegisteredPrediction] = []
    for prediction_rrd in prediction_rrds:
        dataset_name: str = infer_prediction_dataset_name(
            prediction_rrd=prediction_rrd,
            prediction_root=config.prediction_root,
            override=config.prediction_dataset_name,
        )
        layer_name: str = config.layer_name if config.layer_name is not None else prediction_rrd.stem
        dataset_entry: Any = client.get_dataset(dataset_name)
        registration_handle: Any = dataset_entry.register(
            [prediction_rrd.resolve().as_uri()], layer_name=layer_name, on_duplicate=OnDuplicateSegmentLayer.REPLACE
        )
        registration_handle.wait()
        registered_predictions.append(
            RegisteredPrediction(
                dataset_name=dataset_name,
                layer_name=layer_name,
                rrd_path=prediction_rrd,
            )
        )
    return registered_predictions


def register_predictions_into_catalog(config: CatalogRegistrationConfig) -> CatalogRegistrationResult:
    """Connect to the running catalog server and register prediction RRD layers."""
    from rerun.catalog import CatalogClient

    client: Any = CatalogClient(config.catalog_url)
    prediction_rrds: list[Path] = discover_prediction_rrd_paths(config)
    registered_predictions: list[RegisteredPrediction] = register_prediction_rrds(
        client=client,
        config=config,
        prediction_rrds=prediction_rrds,
    )
    return CatalogRegistrationResult(
        catalog_url=config.catalog_url,
        registered_predictions=registered_predictions,
    )


def print_summary(result: CatalogRegistrationResult) -> None:
    """Print the catalog URL and registered prediction layers."""
    print()
    print("-" * 72)
    print(f"Catalog URL: {result.catalog_url}")
    print()
    print("Registered prediction layers:")
    for registered_prediction in result.registered_predictions:
        print(
            f"  {registered_prediction.dataset_name}: {registered_prediction.layer_name} "
            f"from {registered_prediction.rrd_path}"
        )
    print("-" * 72, flush=True)


def main(config: CatalogRegistrationConfig) -> None:
    """CLI entrypoint."""
    result: CatalogRegistrationResult = register_predictions_into_catalog(config)
    print_summary(result)


if __name__ == "__main__":
    main(
        tyro.cli(
            CatalogRegistrationConfig,
            description="Register existing MVAPI prediction RRD layers into a running Rerun catalog.",
        )
    )
