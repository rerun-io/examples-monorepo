"""Tyro entry point: decode a split once into a DetNet or KeyNet cache (``handtrack.data.cache``)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from handtrack.apis.train import DATASETS, SPLITS
from handtrack.data.cache import CacheManifest, write_cache
from handtrack.data.stream import CatalogStream, KeyNetAugment, StreamConfig


@dataclass(frozen=True, slots=True)
class Config:
    """Decode one epoch of a split's DetNet or KeyNet samples into ``output``."""

    output: Path
    """Cache directory on a local disk; ``manifest.json`` is written last."""
    net: Literal["detnet", "keynet"] = "detnet"
    """'detnet' (pooled frames) or 'keynet' (augmented crops: build several passes with other seeds and row phases)."""
    row_phase: float = 0.0
    """Start of the kept rows within each segment's pool stride, as a fraction of it (``StreamConfig.row_phase``)."""
    row_density: int = 1
    """Rows kept per pool stride (``StreamConfig.row_density``): more samples for the same decode."""
    keynet_crop: Literal["affine", "perspective"] = "affine"
    """KeyNet crop type (``StreamConfig.keynet_crop``)."""
    negative_margin: float = 0.0
    """KeyNet negatives keep their own hand this fraction of the crop side outside the crop (``KeyNetAugment.negative_margin``)."""
    keynet_scale_range: tuple[float, float] = (0.9, 1.25)
    """KeyNet crop side multiplier range (KeyNetAugment.scale_range); (1.0, 3.0) mimics DetNet's oversized acquisition circles."""
    keynet_max_shift: float = 0.1
    """KeyNet box centre shift, fraction of the side per axis (KeyNetAugment.max_shift)."""
    keynet_zero_input_probability: float = 0.2
    """Share of KeyNet samples with an all-zero keypoint input, the acquisition case (KeyNetAugment.zero_input_probability)."""
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
        row_density=config.row_density,
        keynet_crop=config.keynet_crop,
        keynet=KeyNetAugment(negative_margin=config.negative_margin, scale_range=config.keynet_scale_range, max_shift=config.keynet_max_shift,
                             zero_input_probability=config.keynet_zero_input_probability),
    ))
    with stream:
        manifest: CacheManifest = write_cache(stream, config.output, config.split, net=config.net)
    print(f"[cache] {manifest.samples} samples from {manifest.segments} segments ({manifest.skipped_segments} skipped) "
          f"in {manifest.build_seconds:.0f} s -> {config.output}", flush=True)
