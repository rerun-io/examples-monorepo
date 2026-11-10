"""Tyro entry point: build the frozen KeyNet evaluation set (``data.keynet_holdout``) once, as one Parquet table."""
from dataclasses import dataclass
from pathlib import Path

import torch
from serde.json import from_json

from handtrack.apis.train import DetectorHoldout
from handtrack.data.batches import CropKind
from handtrack.data.catalog import CATALOG_URL
from handtrack.data.keynet_holdout import HOLDOUT_DATASETS, build_holdout, write_holdout


@dataclass(frozen=True, slots=True)
class Config:
    out: Path
    """The Parquet table to write (a ``.datasets.txt`` beside it names the datasets)."""
    datasets: tuple[str, ...] = HOLDOUT_DATASETS
    crops_per_dataset: int = 2048
    """Crops per dataset in the deployed variant (the rotated variant has as many)."""
    holdout: Path = Path("configs/detector_holdout.json")
    """Held-out episodes of GSI and Gen-HumanEgo (relative to the package directory)."""
    catalog_url: str = CATALOG_URL
    device: str = "cuda"
    seed: int = 0


def main(config: Config) -> None:
    held = from_json(DetectorHoldout, config.holdout.read_text())
    holdout = build_holdout(config.datasets, held.segments, config.catalog_url, torch.device(config.device), config.crops_per_dataset, seed=config.seed)
    write_holdout(config.out, holdout)
    positive = (holdout.samples.kind == int(CropKind.POSITIVE)) & (holdout.variant == 0)
    counts = {name: int((positive & (holdout.samples.dataset == i)).sum()) for i, name in enumerate(holdout.dataset_names)}
    print(f"wrote {config.out}: {holdout.samples.crops.shape[0]} crops; positives (deployed variant) {counts}", flush=True)
