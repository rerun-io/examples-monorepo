"""``fdanyone-birefnet-engine``: build this machine's BiRefNet TensorRT engine once (ONNX export ~50 s on the CPU, then
the engine build), so the person mattes take it from the next run on (``open4dgs.birefnet_trt.load_matte_model``); a
run never builds it itself."""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from open4dgs.birefnet_trt import DEFAULT_CACHE_DIR, build_engine

from fdanyone_rerun.assets import model_dir


@dataclass
class Config:
    model_dir: Path = field(default_factory=lambda: model_dir() / "birefnet")
    """The BiRefNet snapshot (``$FDANYONE_MODEL_DIR/birefnet``)."""
    cache_dir: Path = DEFAULT_CACHE_DIR
    """ONNX and engine cache."""


def main(config: Config) -> None:
    started: float = perf_counter()
    engine: Path = build_engine(config.model_dir, config.cache_dir)
    print(f"BiRefNet TensorRT engine ready in {perf_counter() - started:.0f} s: {engine}")
