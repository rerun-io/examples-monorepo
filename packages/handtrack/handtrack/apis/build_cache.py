"""Tyro entry point: decode a split once into a DetNet cache (``handtrack.data.cache``)."""
from dataclasses import dataclass
from pathlib import Path

from handtrack.apis.train import DATASETS, SPLITS
from handtrack.data.cache import CacheManifest, write_cache
from handtrack.data.stream import CatalogStream, StreamConfig


@dataclass(frozen=True, slots=True)
class Config:
    """Decode one epoch of a split's DetNet samples into ``output``."""

    output: Path
    """Cache directory on a local disk; ``manifest.json`` is written last."""
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
        nets="detnet",
        producers=config.producers,
        fetchers=config.fetchers,
        seed=config.seed,
        device=config.device,
    ))
    with stream:
        manifest: CacheManifest = write_cache(stream, config.output, config.split)
    print(f"[cache] {manifest.samples} samples from {manifest.segments} segments ({manifest.skipped_segments} skipped) "
          f"in {manifest.build_seconds:.0f} s -> {config.output}", flush=True)
