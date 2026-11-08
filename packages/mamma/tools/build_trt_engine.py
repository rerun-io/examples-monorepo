"""Build the machine-local MammaNet TensorRT engine (ONNX export + strongly-typed build).

Engines are sm-specific cache artifacts under ``.trt_cache/`` (gitignored);
demos/benchmark load them via ``LandmarkEstimator(backend="tensorrt")``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from trtkit.tensorrt_runtime import TensorRtRuntime

from mamma.landmarks.mammanet import load_mammanet
from mamma.landmarks.tensorrt_backend import (
    ENGINE_BATCH,
    FP8_LINEARS,
    build_engine,
    calibrate_backbone_amax,
    export_mammanet_onnx,
    quantize_mammanet_fp8,
)


@dataclass
class BuildConfig:
    mammanet_weights: Path | None = None
    """Converted MammaNet state dict; ``None`` downloads the published weights."""
    cache_dir: Path = Path(".trt_cache")
    """Machine-local engine cache."""
    force: bool = False
    """Rebuild even if the engine exists."""
    precision: Literal["fp16", "fp8"] = "fp16"
    """``fp8`` adds FP8 Q/DQ to the ViT block linears (static input scales from ``calib_crops``)."""
    calib_crops: Path | None = None
    """``torch.save``'d ``{"crops": [n, b, 3, h, w], "masks": [n, b, 1, h, w]}`` real engine inputs
    (``tools/record_mammanet_crops.py``). Required for ``fp8``; also used for real-crop parity."""
    fp8_linears: tuple[str, ...] = FP8_LINEARS
    """ViT block linear names (``qkv``/``proj``/``fc1``/``fc2``) the fp8 build quantizes."""


def engine_path_for(cache_dir: Path, precision: str = "fp16") -> Path:
    import tensorrt as trt

    sm = "sm{}{}".format(*torch.cuda.get_device_capability())
    return cache_dir / f"mammanet_b{ENGINE_BATCH}_{precision}_trt{trt.__version__.replace('.', '')}_{sm}.plan"


def main(config: BuildConfig) -> None:
    engine_path: Path = engine_path_for(config.cache_dir, config.precision)
    if engine_path.exists() and not config.force:
        print(f"engine exists: {engine_path}")
        return
    if config.precision == "fp8" and config.calib_crops is None:
        raise ValueError("--precision fp8 needs --calib-crops (record them with tools/record_mammanet_crops.py).")
    model = load_mammanet(config.mammanet_weights, device="cuda")
    onnx_path: Path = config.cache_dir / "mammanet.onnx"
    print("exporting ONNX (dynamo exporter, fp16-typed graph)...")
    export_mammanet_onnx(model, onnx_path)
    calib: dict[str, torch.Tensor] | None = torch.load(config.calib_crops) if config.calib_crops is not None else None
    real_refs: list[torch.Tensor] = []
    if calib is not None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for batch_crops, batch_masks in zip(calib["crops"], calib["masks"], strict=True):
                joints = model(batch_crops.cuda().float(), batch_masks.cuda().float())["joints2d"]
                assert joints is not None
                real_refs.append(joints.float())
    if config.precision == "fp8":
        assert calib is not None
        amax: dict[str, float] = calibrate_backbone_amax(model, calib["crops"], calib["masks"])
        fp8_onnx: Path = config.cache_dir / "mammanet_fp8.onnx"
        print(f"inserting FP8 Q/DQ on ViT {', '.join(config.fp8_linears)} ({len(calib['crops'])} calibration batches)...")
        quantize_mammanet_fp8(onnx_path, fp8_onnx, amax, config.fp8_linears)
        onnx_path = fp8_onnx

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
    if calib is not None:
        real: torch.Tensor = torch.cat([
            (runner({"crops": batch_crops.cuda().float(), "masks": batch_masks.cuda().float()})["joints2d"][..., :2] - batch_ref[..., :2]).abs().flatten()
            for batch_crops, batch_masks, batch_ref in zip(calib["crops"], calib["masks"], real_refs, strict=True)
        ])
        print(f"joints2d xy diff vs eager fp16 on {len(real_refs)} real crop batches: p50={real.median().item():.5f} p99={real.quantile(0.99).item():.5f} max={real.max().item():.5f}")

    # Latency.
    for _ in range(10):
        runner({"crops": x, "masks": masks})
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(50):
        runner({"crops": x, "masks": masks})
    end.record()
    torch.cuda.synchronize()
    print(f"TRT latency: {start.elapsed_time(end) / 50:.2f} ms per {ENGINE_BATCH}-crop call (GB10: fp16 17.7, fp8 15.9)")


if __name__ == "__main__":
    main(tyro.cli(BuildConfig))
