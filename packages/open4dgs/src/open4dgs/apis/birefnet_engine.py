"""Build this machine's BiRefNet TensorRT engine once (ONNX export ~50 s on the CPU, then the engine build), so mattes
take the engine from the next run on (``open4dgs.birefnet_trt.load_matte_model``); a run never builds it itself."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from open4dgs.birefnet_trt import DEFAULT_CACHE_DIR, build_engine


@dataclass
class Config:
    model_dir: Path = field(default_factory=lambda: Path(os.environ.get("FDANYONE_MODEL_DIR", "data/models")) / "birefnet")
    """The BiRefNet snapshot (fdanyone's ``$FDANYONE_MODEL_DIR/birefnet``)."""
    cache_dir: Path = DEFAULT_CACHE_DIR
    """ONNX and engine cache."""


def main(config: Config) -> None:
    started: float = perf_counter()
    engine: Path = build_engine(config.model_dir, config.cache_dir)
    print(f"BiRefNet TensorRT engine ready in {perf_counter() - started:.0f} s: {engine}")
