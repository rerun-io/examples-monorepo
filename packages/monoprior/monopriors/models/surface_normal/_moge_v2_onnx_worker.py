"""Uninstrumented subprocess entry point for dynamic-batch MoGe ONNX export."""

import sys
from pathlib import Path
from typing import cast

from monopriors.models.surface_normal.moge_v2_trt import Encoder, export_moge_v2_normal_onnx


def main() -> None:
    """Parse the parent export request and build its ONNX artifact."""
    if len(sys.argv) != 7:
        raise ValueError(f"Expected six MoGe v2 ONNX worker arguments, got {len(sys.argv) - 1}.")
    encoder: Encoder = cast(Encoder, sys.argv[1])
    height: int = int(sys.argv[2])
    width: int = int(sys.argv[3])
    resolution_level: int = int(sys.argv[4])
    max_batch_size: int = int(sys.argv[5])
    cache_dir: Path = Path(sys.argv[6])
    export_moge_v2_normal_onnx(
        encoder=encoder,
        image_hw=(height, width),
        resolution_level=resolution_level,
        max_batch_size=max_batch_size,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()
