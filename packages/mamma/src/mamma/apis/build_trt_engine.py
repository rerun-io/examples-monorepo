"""Build the machine-local MammaNet TensorRT engine (ONNX export + strongly-typed build) and check it against eager fp16.

Engines are GPU- and TensorRT-specific cache artifacts under ``.trt_cache/`` (gitignored); the capture App's landmarks
model loads the plan when it exists.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from trtkit.tensorrt_runtime import TensorRtRuntime

from mamma.models.landmarks.mammanet import load_mammanet
from mamma.models.landmarks.tensorrt_backend import ENGINE_BATCH, build_engine, export_mammanet_onnx, static_engine_path


@dataclass
class BuildConfig:
    mammanet_weights: Path | None = None
    """Converted MammaNet state dict; ``None`` downloads the published weights."""
    cache_dir: Path = Path(__file__).resolve().parents[3] / ".trt_cache"
    """Machine-local engine cache (``packages/mamma/.trt_cache``)."""
    force: bool = False
    """Rebuild even if the engine exists."""


def main(config: BuildConfig) -> None:
    engine_path: Path = static_engine_path(config.cache_dir)
    if engine_path.exists() and not config.force:
        print(f"engine exists: {engine_path}")
        return
    model = load_mammanet(config.mammanet_weights, device="cuda")
    onnx_path: Path = config.cache_dir / "mammanet.onnx"
    print("exporting ONNX (dynamo exporter, fp16-typed graph)...")
    export_mammanet_onnx(model, onnx_path)

    # Compute the eager-fp16 parity reference BEFORE the engine build and free
    # the model so its VRAM doesn't compete with the builder's workspace.
    torch.manual_seed(0)
    x = torch.randn(ENGINE_BATCH, 3, 512, 384, device="cuda")
    masks = (torch.rand(ENGINE_BATCH, 1, 512, 384, device="cuda") > 0.5).float()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        ref = model(x, masks)
    del model
    torch.cuda.empty_cache()

    print(f"building strongly-typed engine ({engine_path.name})...")
    t0: float = time.perf_counter()
    build_engine(onnx_path, engine_path)
    print(f"built in {time.perf_counter() - t0:.0f}s -> {engine_path}")

    runner = TensorRtRuntime(engine_path, use_cuda_graph=True)
    out = runner({"crops": x, "masks": masks})
    ref_joints = ref["joints2d"]
    assert ref_joints is not None
    diff = (out["joints2d"] - ref_joints.float()).abs()
    print(f"joints2d abs diff vs eager fp16: max={diff.max().item():.5f} p99={diff.quantile(0.99).item():.5f} (normalized units)")

    # Latency.
    for _ in range(10):
        runner({"crops": x, "masks": masks})
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        runner({"crops": x, "masks": masks})
    torch.cuda.synchronize()
    print(f"TRT latency: {(time.perf_counter() - t0) / 50 * 1000:.2f} ms per {ENGINE_BATCH}-crop call (eager fp16 was ~15.8)")
