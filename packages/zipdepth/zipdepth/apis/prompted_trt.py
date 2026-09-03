"""Static-batch fp16 ONNX/TensorRT export for ZipDepth-PromptDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from jaxtyping import Float32
from monopriors.models.depth_completion.base_completion_depth import PROMPT_DEPTH_HW
from monopriors.models.depth_completion.zipdepth_prompt import load_zipdepth_prompt
from monopriors.models.depth_completion.zipdepth_prompt_export import (
    DEFAULT_CACHE_DIR,
    DEPTH_OUTPUT_NAME,
    IMAGE_INPUT_NAME,
    PROMPT_INPUT_NAME,
    PromptedDepthModel,
    export_zipdepth_prompt_onnx,
    prepare_zipdepth_prompt_for_export,
)
from monopriors.models.relative_depth.zipdepth import download_zipdepth_checkpoint
from serde import serde
from serde.json import to_json
from torch import Tensor
from trtkit import TensorRuntime

ParityInputs: TypeAlias = dict[
    str,
    Float32[Tensor, "b 3 h w"] | Float32[Tensor, "b 1 192 256"],
]


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


@serde
@dataclass(frozen=True, slots=True)
class TrtReportConfig:
    """Resolved TensorRT report configuration."""

    checkpoint: Path
    """Checkpoint used for the export and parity run."""
    cache_dir: Path
    """Portable ONNX and engine cache root."""
    height: int
    """Static image height."""
    width: int
    """Static image width."""
    batch_size: int
    """Static export and engine batch size."""
    warmup: int
    """Unmeasured TensorRT calls before timing."""
    iterations: int
    """Measured TensorRT calls."""
    result_path: Path
    """Written report path."""


@serde
@dataclass(frozen=True, slots=True)
class ParityError:
    """Absolute metric-depth error for one backend comparison."""

    median_abs_m: float
    """Median absolute error in metres."""
    max_abs_m: float
    """Maximum absolute error in metres."""


@serde
@dataclass(frozen=True, slots=True)
class BackendParity:
    """ONNX and TensorRT errors relative to Torch for one prompt case."""

    onnx_vs_torch: ParityError
    """ONNX Runtime error relative to Torch."""
    trt_vs_torch: ParityError
    """TensorRT error relative to Torch."""


@serde
@dataclass(frozen=True, slots=True)
class TrtParityReport:
    """Backend parity for the wide/holey and narrow prompt cases."""

    holey: BackendParity
    """Parity for prompts with invalid holes and a wide metric range."""
    narrow: BackendParity
    """Parity for prompts with a narrow metric range."""


@serde
@dataclass(frozen=True, slots=True)
class TrtLatency:
    """CUDA-event TensorRT latency measurements."""

    batch_size: int
    """Frames processed per engine call."""
    warmup: int
    """Unmeasured engine calls."""
    iterations: int
    """Measured engine calls."""
    median_ms_per_batch: float
    """Median milliseconds per engine call."""
    p95_ms_per_batch: float
    """95th-percentile milliseconds per engine call."""
    median_ms_per_frame: float
    """Median amortized milliseconds per frame."""
    frames_per_second: float
    """Median aggregate throughput."""


@serde
@dataclass(frozen=True, slots=True)
class TrtReport:
    """TensorRT artifacts, parity results, and latency report."""

    config: TrtReportConfig
    """Resolved run configuration."""
    onnx_path: Path
    """Exported ONNX graph path."""
    engine_path: Path
    """Built TensorRT engine path."""
    static_batch: int
    """Batch dimension found in the ONNX graph."""
    parity: TrtParityReport
    """Torch, ONNX Runtime, and TensorRT parity results."""
    latency: TrtLatency
    """TensorRT CUDA-event latency results."""
    gpu: str
    """CUDA device name used for the run."""


def _parity_inputs(batch_size: int, image_hw: tuple[int, int], device: torch.device) -> ParityInputs:
    """Create deterministic image plus alternating holey/narrow prompts."""
    generator: torch.Generator = torch.Generator(device=device).manual_seed(7)
    image: Float32[Tensor, "b 3 h w"] = torch.rand((batch_size, 3, *image_hw), generator=generator, dtype=torch.float32, device=device)
    prompt_pixels: int = PROMPT_DEPTH_HW[0] * PROMPT_DEPTH_HW[1]
    wide: Float32[Tensor, "1 1 192 256"] = torch.linspace(
        0.2, 3.8, prompt_pixels, dtype=torch.float32, device=device
    ).reshape(1, 1, *PROMPT_DEPTH_HW)
    wide[:, :, 24:160:3, 32:224:4] = 0.0
    narrow: Float32[Tensor, "1 1 192 256"] = torch.linspace(
        1.500, 1.503, prompt_pixels, dtype=torch.float32, device=device
    ).reshape(1, 1, *PROMPT_DEPTH_HW)
    prompt_rows: list[Float32[Tensor, "1 1 192 256"]] = [wide if index % 2 == 0 else narrow for index in range(batch_size)]
    return {IMAGE_INPUT_NAME: image, PROMPT_INPUT_NAME: torch.cat(prompt_rows, dim=0).contiguous()}


def _error_summary(
    actual: Float32[Tensor, "b 1 h w"],
    expected: Float32[Tensor, "b 1 h w"],
    rows: slice,
) -> ParityError:
    """Summarize absolute metric-depth parity error for selected rows."""
    error: Float32[Tensor, "selected 1 h w"] = (actual[rows] - expected[rows]).abs()
    return ParityError(median_abs_m=float(error.median()), max_abs_m=float(error.max()))


def verify_three_backend_parity(
    checkpoint: Path,
    onnx_path: Path,
    image_hw: tuple[int, int],
    inputs: ParityInputs,
    trt_runtime: TensorRuntime,
) -> TrtParityReport:
    """Measure Torch-vs-ONNX-vs-TRT error for holey and narrow prompts."""
    from trtkit import OnnxCudaRuntime

    device = torch.device("cuda")
    model: PromptedDepthModel = prepare_zipdepth_prompt_for_export(load_zipdepth_prompt(checkpoint), image_hw, device)
    with torch.inference_mode():
        expected: Float32[Tensor, "b 1 h w"] = model(inputs[IMAGE_INPUT_NAME], inputs[PROMPT_INPUT_NAME]).clone()
    onnx_depth: Float32[Tensor, "b 1 h w"] = OnnxCudaRuntime(onnx_path)(inputs)[DEPTH_OUTPUT_NAME].clone()
    trt_depth: Float32[Tensor, "b 1 h w"] = trt_runtime(inputs)[DEPTH_OUTPUT_NAME].clone()
    result: TrtParityReport = TrtParityReport(
        holey=BackendParity(
            onnx_vs_torch=_error_summary(onnx_depth, expected, slice(0, None, 2)),
            trt_vs_torch=_error_summary(trt_depth, expected, slice(0, None, 2)),
        ),
        narrow=BackendParity(
            onnx_vs_torch=_error_summary(onnx_depth, expected, slice(1, None, 2)),
            trt_vs_torch=_error_summary(trt_depth, expected, slice(1, None, 2)),
        ),
    )
    del model
    torch.cuda.empty_cache()
    return result


def benchmark_tensorrt(
    runtime: TensorRuntime,
    inputs: ParityInputs,
    warmup: int,
    iterations: int,
) -> TrtLatency:
    """Measure engine-call latency with CUDA events, including input copies."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    batch_size: int = int(inputs[IMAGE_INPUT_NAME].shape[0])
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
    return TrtLatency(
        batch_size=batch_size,
        warmup=warmup,
        iterations=iterations,
        median_ms_per_batch=median_ms,
        p95_ms_per_batch=float(np.percentile(elapsed_ms, 95)),
        median_ms_per_frame=median_ms / batch_size,
        frames_per_second=1000.0 * batch_size / median_ms,
    )


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
    from trtkit import TensorRtRuntime, TrtBuildConfig, ensure_engine, onnx_static_batch_size

    static_batch: int | None = onnx_static_batch_size(onnx_path)
    if static_batch is None or static_batch != config.batch_size:
        raise RuntimeError(f"ONNX batch must be static {config.batch_size}, got {static_batch}")
    engine_path: Path = ensure_engine(
        onnx_path,
        TrtBuildConfig(max_batch_size=config.batch_size, opt_batch_size=config.batch_size),
        cache_dir=config.cache_dir / "engines",
    )
    inputs: ParityInputs = _parity_inputs(config.batch_size, image_hw, torch.device("cuda"))
    trt_runtime: TensorRuntime = TensorRtRuntime(engine_path)
    parity: TrtParityReport = verify_three_backend_parity(
        checkpoint,
        onnx_path,
        image_hw,
        inputs,
        trt_runtime,
    )
    parity_cases: tuple[tuple[str, BackendParity], ...] = (("holey", parity.holey), ("narrow", parity.narrow))
    for case_name, backends in parity_cases:
        backend_errors: tuple[tuple[str, ParityError], ...] = (
            ("onnx_vs_torch", backends.onnx_vs_torch),
            ("trt_vs_torch", backends.trt_vs_torch),
        )
        for backend_name, errors in backend_errors:
            if errors.median_abs_m >= 0.003 or errors.max_abs_m >= 0.02:
                raise RuntimeError(f"parity failed for {backend_name}:{case_name}: {errors}")
    latency: TrtLatency = benchmark_tensorrt(
        trt_runtime,
        inputs,
        config.warmup,
        config.iterations,
    )
    report: TrtReport = TrtReport(
        config=TrtReportConfig(
            checkpoint=checkpoint,
            cache_dir=config.cache_dir,
            height=config.height,
            width=config.width,
            batch_size=config.batch_size,
            warmup=config.warmup,
            iterations=config.iterations,
            result_path=config.result_path,
        ),
        onnx_path=onnx_path,
        engine_path=engine_path,
        static_batch=static_batch,
        parity=parity,
        latency=latency,
        gpu=torch.cuda.get_device_name(),
    )
    report_json: str = to_json(report)
    config.result_path.parent.mkdir(parents=True, exist_ok=True)
    config.result_path.write_text(report_json + "\n", encoding="utf-8")
    print(report_json)
    return config.result_path
