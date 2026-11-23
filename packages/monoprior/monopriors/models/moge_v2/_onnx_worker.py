"""Uninstrumented subprocess entry point for MoGe ONNX export (dynamic batch, optionally dynamic size)."""

from dataclasses import dataclass
from pathlib import Path

import tyro

from monopriors.models.moge_v2.export import Encoder, HeadSet, export_moge_v2_onnx


@dataclass(frozen=True, slots=True)
class Config:
    """Export request forwarded by the parent predictor process."""

    heads: HeadSet
    """Head set to export in the clean interpreter."""
    encoder: Encoder
    """DINOv2 encoder size."""
    resolution_level: int
    """Detail level from 0 through 9."""
    max_batch_size: int
    """Largest batch encoded in the dynamic ONNX constraint."""
    cache_dir: Path
    """Cache root; the graph lands in ``cache_dir / "onnx"``."""
    height: int | None = None
    """Static network height; None with ``width`` None exports the dynamic graph."""
    width: int | None = None
    """Static network width."""


def main(config: Config) -> None:
    """Build the requested ONNX artifact in this clean interpreter."""
    export_moge_v2_onnx(
        heads=config.heads,
        encoder=config.encoder,
        image_hw=None if config.height is None or config.width is None else (config.height, config.width),
        resolution_level=config.resolution_level,
        max_batch_size=config.max_batch_size,
        cache_dir=config.cache_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
