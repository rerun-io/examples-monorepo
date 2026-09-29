"""Tyro entry point: decode a split once into a DetNet or KeyNet cache (``handtrack.data.cache``)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from handtrack.apis.train import DATASETS, SPLITS
from handtrack.data.cache import CacheManifest, write_cache
from handtrack.data.stream import CatalogStream, StreamConfig


@dataclass(frozen=True, slots=True)
class Config:
    """Decode one epoch of a split's DetNet or KeyNet samples into ``output``."""

    output: Path
    """Cache directory on a local disk; ``manifest.json`` is written last."""
    net: Literal["detnet", "keynet"] = "detnet"
    """'detnet' (pooled frames) or 'keynet' (augmented crops: build several passes with other seeds and row phases)."""
    row_phase: float = 0.0
    """Start of the kept rows within each segment's pool stride, as a fraction of it (``StreamConfig.row_phase``)."""
    datasets: tuple[str, ...] = ("dataforge-umetrack", "dataforge-show3d")
    split: str = "training"
    producers: int = 4
    fetchers: int = 4
    seed: int = 0
    device: str = "cuda"


def main(config: Config) -> None:
    stream: CatalogStream = CatalogStream(StreamConfig(
        datasets=tuple(DATASETS[name] for name in config.datasets),
        split=SPLITS[config.split],
        nets="keynet" if config.net == "keynet" else "detnet",
        producers=config.producers,
        fetchers=config.fetchers,
        seed=config.seed,
        device=config.device,
        row_phase=config.row_phase,
    ))
    with stream:
        manifest: CacheManifest = write_cache(stream, config.output, config.split, net=config.net)
    print(f"[cache] {manifest.samples} samples from {manifest.segments} segments ({manifest.skipped_segments} skipped) "
          f"in {manifest.build_seconds:.0f} s -> {config.output}", flush=True)
