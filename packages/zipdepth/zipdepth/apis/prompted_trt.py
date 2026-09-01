"""Static-batch fp16 ONNX/TensorRT export for ZipDepth-PromptDA."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monopriors.models.depth_completion.zipdepth_prompt import load_zipdepth_prompt
from monopriors.models.depth_completion.zipdepth_prompt_export import (
    DEFAULT_CACHE_DIR,
    DEPTH_OUTPUT_NAME,
    IMAGE_INPUT_NAME,
    PROMPT_INPUT_NAME,
    export_zipdepth_prompt_onnx,
    prepare_zipdepth_prompt_for_export,
)
from monopriors.models.relative_depth.zipdepth import download_zipdepth_checkpoint
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class TrtExportConfig:
    """Export, parity-check, and benchmark configuration."""

    checkpoint: Path | None = None
    """Prompted trainer checkpoint; None uses released ZipDepth-base with a zero branch."""
    cache_dir: Path = DEFAULT_CACHE_DIR
    """Portable ONNX, machine-local engine, and result cache root."""
    height: int = 768
    """Static image height."""
    width: int = 1024
    """Static image width."""
    batch_size: int = 8
    """Batch baked into both ONNX and TensorRT artifacts."""
    warmup: int = 10
    """Unmeasured TensorRT calls before timing."""
    iterations: int = 50
    """Measured TensorRT calls."""
    result_path: Path = Path("/tmp/zd-pda-research/trt-parity-latency.json")
    """JSON report containing artifact paths, parity errors, and CUDA-event latency."""


def _parity_inputs(batch_size: int, image_hw: tuple[int, int], device: torch.device) -> dict[str, Tensor]:
    """Create deterministic image plus alternating holey/narrow prompts."""
    generator = torch.Generator(device=device).manual_seed(7)
    image = torch.rand((batch_size, 3, *image_hw), generator=generator, dtype=torch.float32, device=device)
    wide = torch.linspace(0.2, 3.8, 192 * 256, dtype=torch.float32, device=device).reshape(1, 1, 192, 256)
    wide[:, :, 24:160:3, 32:224:4] = 0.0
    narrow = torch.linspace(1.500, 1.503, 192 * 256, dtype=torch.float32, device=device).reshape(1, 1, 192, 256)
    prompt_rows: list[Tensor] = [wide if index % 2 == 0 else narrow for index in range(batch_size)]
    return {IMAGE_INPUT_NAME: image, PROMPT_INPUT_NAME: torch.cat(prompt_rows, dim=0).contiguous()}


def _error_summary(actual: Tensor, expected: Tensor, rows: slice) -> dict[str, float]:
    """Summarize absolute metric-depth parity error for selected rows."""
    error: Tensor = (actual[rows] - expected[rows]).abs()
    return {"median_abs_m": float(error.median()), "max_abs_m": float(error.max())}


def verify_three_backend_parity(
    checkpoint: Path,
    onnx_path: Path,
    engine_path: Path,
    image_hw: tuple[int, int],
    batch_size: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Measure Torch-vs-ONNX-vs-TRT error for holey and narrow prompts."""
    from trtkit import OnnxCudaRuntime, TensorRtRuntime

    device = torch.device("cuda")
    inputs: dict[str, Tensor] = _parity_inputs(batch_size, image_hw, device)
    model: nn.Module = prepare_zipdepth_prompt_for_export(load_zipdepth_prompt(checkpoint), image_hw, device)
    with torch.inference_mode():
        expected: Tensor = model(inputs[IMAGE_INPUT_NAME], inputs[PROMPT_INPUT_NAME]).clone()
    onnx_depth: Tensor = OnnxCudaRuntime(onnx_path)(inputs)[DEPTH_OUTPUT_NAME].clone()
    trt_depth: Tensor = TensorRtRuntime(engine_path)(inputs)[DEPTH_OUTPUT_NAME].clone()
    result: dict[str, dict[str, dict[str, float]]] = {}
    for case_name, rows in (("holey", slice(0, None, 2)), ("narrow", slice(1, None, 2))):
        result[case_name] = {
            "onnx_vs_torch": _error_summary(onnx_depth, expected, rows),
            "trt_vs_torch": _error_summary(trt_depth, expected, rows),
        }
    del model
    torch.cuda.empty_cache()
    return result


def benchmark_tensorrt(
    engine_path: Path,
    image_hw: tuple[int, int],
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    """Measure engine-call latency with CUDA events, including input copies."""
    from trtkit import TensorRtRuntime

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    runtime = TensorRtRuntime(engine_path)
    inputs: dict[str, Tensor] = _parity_inputs(batch_size, image_hw, torch.device("cuda"))
    for _ in range(warmup):
        runtime(inputs)
    torch.cuda.synchronize()
    elapsed_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runtime(inputs)
        end.record()
        end.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))
    median_ms: float = float(np.median(elapsed_ms))
    return {
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "median_ms_per_batch": median_ms,
        "p95_ms_per_batch": float(np.percentile(elapsed_ms, 95)),
        "median_ms_per_frame": median_ms / batch_size,
        "frames_per_second": 1000.0 * batch_size / median_ms,
    }


def main(config: TrtExportConfig) -> Path:
    """Export, build, verify parity, benchmark, and write a JSON report."""
    if not torch.cuda.is_available():
        raise RuntimeError("ZipDepth TensorRT export requires CUDA")
    checkpoint: Path = config.checkpoint or download_zipdepth_checkpoint()
    image_hw: tuple[int, int] = (config.height, config.width)
    onnx_path: Path = export_zipdepth_prompt_onnx(
        checkpoint=checkpoint,
        image_hw=image_hw,
        batch_size=config.batch_size,
        cache_dir=config.cache_dir,
    )
    from trtkit import TrtBuildConfig, ensure_engine, onnx_static_batch_size

    static_batch: int | None = onnx_static_batch_size(onnx_path)
    if static_batch != config.batch_size:
        raise RuntimeError(f"ONNX batch must be static {config.batch_size}, got {static_batch}")
    engine_path: Path = ensure_engine(
        onnx_path,
        TrtBuildConfig(max_batch_size=config.batch_size, opt_batch_size=config.batch_size),
        cache_dir=config.cache_dir / "engines",
    )
    parity: dict[str, dict[str, dict[str, float]]] = verify_three_backend_parity(
        checkpoint,
        onnx_path,
        engine_path,
        image_hw,
        config.batch_size,
    )
    for case_name, backends in parity.items():
        for backend_name, errors in backends.items():
            if errors["median_abs_m"] >= 0.003 or errors["max_abs_m"] >= 0.02:
                raise RuntimeError(f"parity failed for {backend_name}:{case_name}: {errors}")
    latency: dict[str, float | int] = benchmark_tensorrt(
        engine_path,
        image_hw,
        config.batch_size,
        config.warmup,
        config.iterations,
    )
    report: dict[str, Any] = {
        "config": {**asdict(config), "checkpoint": str(checkpoint), "cache_dir": str(config.cache_dir), "result_path": str(config.result_path)},
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "static_batch": static_batch,
        "parity": parity,
        "latency": latency,
        "gpu": torch.cuda.get_device_name(),
    }
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return config.result_path
