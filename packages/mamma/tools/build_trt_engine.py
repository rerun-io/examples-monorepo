"""Build the machine-local MammaNet TensorRT engine (ONNX export + FP16 build).

Engines are sm-specific cache artifacts under ``.trt_cache/`` (gitignored);
demos/benchmark load them via ``LandmarkEstimator(backend="tensorrt")``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from mamma.landmarks.mammanet import load_mammanet
from mamma.landmarks.tensorrt_backend import ENGINE_BATCH, MammaNetTrtRunner, build_engine, export_mammanet_onnx


@dataclass
class BuildConfig:
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet state dict."""
    cache_dir: Path = Path(".trt_cache")
    """Machine-local engine cache."""
    force: bool = False
    """Rebuild even if the engine exists."""


def engine_path_for(cache_dir: Path) -> Path:
    import tensorrt as trt

    sm = "sm{}{}".format(*torch.cuda.get_device_capability())
    return cache_dir / f"mammanet_b{ENGINE_BATCH}_fp16_trt{trt.__version__.replace('.', '')}_{sm}.plan"


def main(config: BuildConfig) -> None:
    engine_path: Path = engine_path_for(config.cache_dir)
    if engine_path.exists() and not config.force:
        print(f"engine exists: {engine_path}")
        return
    model = load_mammanet(config.mammanet_weights, device="cuda")
    onnx_path: Path = config.cache_dir / "mammanet.onnx"
    print("exporting ONNX (legacy exporter, opset 17)...")
    export_mammanet_onnx(model, onnx_path)
    print(f"building FP16 engine ({engine_path.name})...")
    t0: float = time.perf_counter()
    build_engine(onnx_path, engine_path)
    print(f"built in {time.perf_counter() - t0:.0f}s -> {engine_path}")

    # Numerics check vs eager fp16.
    runner = MammaNetTrtRunner(engine_path)
    torch.manual_seed(0)
    x = torch.randn(ENGINE_BATCH, 3, 512, 384, device="cuda")
    masks = (torch.rand(ENGINE_BATCH, 1, 512, 384, device="cuda") > 0.5).float()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        ref = model(x, masks)
    out = runner(x, masks)
    ref_joints = ref["joints2d"]
    assert ref_joints is not None
    diff = (out["joints2d"] - ref_joints.float()).abs()
    print(f"joints2d abs diff vs eager fp16: max={diff.max().item():.5f} p99={diff.quantile(0.99).item():.5f} (normalized units)")

    # Latency.
    for _ in range(10):
        runner(x, masks)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        runner(x, masks)
    torch.cuda.synchronize()
    print(f"TRT latency: {(time.perf_counter() - t0) / 50 * 1000:.2f} ms per {ENGINE_BATCH}-crop call (eager fp16 was ~15.8)")


if __name__ == "__main__":
    main(tyro.cli(BuildConfig))
