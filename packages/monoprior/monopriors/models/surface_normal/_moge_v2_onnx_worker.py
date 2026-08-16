"""Uninstrumented subprocess entry point for dynamic-batch MoGe ONNX export."""

from dataclasses import dataclass
from pathlib import Path

import tyro

from monopriors.models.surface_normal.moge_v2_trt import Encoder, export_moge_v2_normal_onnx


@dataclass(frozen=True, slots=True)
class Config:
    """Export request forwarded by the parent predictor process."""

    encoder: Encoder
    """DINOv2 encoder size."""
    height: int
    """Static network height."""
    width: int
    """Static network width."""
    resolution_level: int
    """Detail level from 0 through 9."""
    max_batch_size: int
    """Largest batch encoded in the dynamic ONNX constraint."""
    cache_dir: Path
    """Cache root; the graph lands in ``cache_dir / "onnx"``."""


def main(config: Config) -> None:
    """Build the requested ONNX artifact in this clean interpreter."""
    export_moge_v2_normal_onnx(
        encoder=config.encoder,
        image_hw=(config.height, config.width),
        resolution_level=config.resolution_level,
        max_batch_size=config.max_batch_size,
        cache_dir=config.cache_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
