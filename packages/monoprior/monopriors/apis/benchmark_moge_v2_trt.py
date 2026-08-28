"""Benchmark vendored, batched torch, and TensorRT MoGe v2 predictors."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float32, UInt8
from torch import Tensor

from monopriors.models.monoprior.moge_v2_trt import (
    ASPECT_BUCKETS,
    Encoder,
    MoGeV2GeometryGraphOutput,
    MoGeV2TorchGeometryPredictor,
    MoGeV2TrtGeometryPredictor,
    postprocess_moge_v2_geometry,
    select_aspect_bucket,
)
from monopriors.models.surface_normal.moge_v2 import MOGE_V2_NORMAL_CHECKPOINTS
from monopriors.third_party.moge.model.v2 import MoGeModel

GeometryPredictor: TypeAlias = MoGeV2TorchGeometryPredictor | MoGeV2TrtGeometryPredictor


@dataclass(frozen=True, slots=True)
class Config:
    """MoGe v2 batched predictor benchmark configuration."""

    encoder: Encoder = "vitl"
    """DINOv2 encoder size shared by every selected backend."""
    image_hw: tuple[int, int] = ASPECT_BUCKETS[0]
    """Primary caller height and width; defaults to the 4:3 network bucket."""
    widescreen_image_hw: tuple[int, int] = (1080, 1920)
    """Caller height and width for one 16:9 row per TensorRT backend."""
    resolution_level: int = 9
    """MoGe detail level from 0 through 9."""
    batch_sizes: tuple[int, ...] = (1, 4, 8)
    """Primary batch sizes; the torch backend also measures batch 16."""
    trt_max_batch_size: int = 8
    """Maximum batch in each TensorRT engine profile."""
    builder_optimization_level: int = 5
    """TensorRT builder optimization level from 0 through 5."""
    warmup_iters: int = 3
    """Untimed iterations before each backend and batch measurement."""
    timed_iters: int = 20
    """Synchronized iterations accumulated for each table row."""
    run_moge_infer: bool = True
    """Benchmark the current one-image-at-a-time vendored inference path."""
    run_torch_geometry: bool = True
    """Benchmark the fp16-autocast torch geometry twin."""
    run_trt_geometry: bool = True
    """Benchmark the full-head TensorRT geometry predictor."""
    run_trt_geometry_cudagraph: bool = True
    """Benchmark the same TensorRT engines with CUDA-graph replay."""


@dataclass(frozen=True, slots=True)
class BenchmarkImage:
    """One tiled CUDA input batch and its source label."""

    rgb_bhw3: UInt8[Tensor, "b h w 3"]
    """uint8 CUDA RGB frames at caller resolution."""
    source: str
    """Human-readable source image description."""


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    """One backend, input shape, bucket, and batch-size timing result."""

    backend: str
    """Stable backend label."""
    input_hw: tuple[int, int]
    """Caller input height and width."""
    bucket_hw: tuple[int, int]
    """Selected MoGe network height and width."""
    batch_size: int
    """Frames in each predictor call."""
    total_ms_per_batch: float
    """Synchronized total wall time per batch."""
    forward_ms_per_batch: float | None
    """Synchronized preprocess plus model or engine time, when split."""
    postprocess_ms_per_batch: float | None
    """Synchronized camera-recovery and output time, when split."""


def _benchmark_geometry(
    backend: str,
    predictor: GeometryPredictor,
    rgb_bhw3: UInt8[Tensor, "b h w 3"],
    warmup_iters: int,
    timed_iters: int,
) -> BenchmarkRow:
    """Measure staged geometry forward and postprocess wall time.

    Args:
        backend: Stable backend label.
        predictor: Torch or TensorRT staged geometry predictor.
        rgb_bhw3: uint8 CUDA RGB batch shaped ``b h w 3``.
        warmup_iters: Untimed iterations.
        timed_iters: Iterations accumulated in the result.

    Returns:
        Per-batch synchronized timing row with forward and postprocess split.
    """
    output_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
    bucket_hw: tuple[int, int] = select_aspect_bucket(output_hw, ASPECT_BUCKETS)
    for _warmup_index in range(warmup_iters):
        warmup_graph_output: MoGeV2GeometryGraphOutput = predictor.forward(rgb_bhw3)
        postprocess_moge_v2_geometry(warmup_graph_output, output_hw)
    torch.cuda.synchronize()

    forward_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    for _timed_index in range(timed_iters):
        torch.cuda.synchronize()
        forward_started: float = time.perf_counter()
        graph_output: MoGeV2GeometryGraphOutput = predictor.forward(rgb_bhw3)
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - forward_started

        postprocess_started: float = time.perf_counter()
        postprocess_moge_v2_geometry(graph_output, output_hw)
        torch.cuda.synchronize()
        postprocess_seconds += time.perf_counter() - postprocess_started

    scale_to_ms_per_batch: float = 1000.0 / timed_iters
    return BenchmarkRow(
        backend=backend,
        input_hw=output_hw,
        bucket_hw=bucket_hw,
        batch_size=rgb_bhw3.shape[0],
        total_ms_per_batch=(forward_seconds + postprocess_seconds) * scale_to_ms_per_batch,
        forward_ms_per_batch=forward_seconds * scale_to_ms_per_batch,
        postprocess_ms_per_batch=postprocess_seconds * scale_to_ms_per_batch,
    )


def _benchmark_moge_infer(
    model: MoGeModel,
    image_3hw: Float32[Tensor, "3 h w"],
    resolution_level: int,
    warmup_iters: int,
    timed_iters: int,
) -> BenchmarkRow:
    """Measure the current all-head vendored inference path on one image.

    Args:
        model: Pinned normal-capable MoGe v2 model on CUDA.
        image_3hw: Float32 CUDA RGB image shaped ``3 h w`` in ``[0, 1]``.
        resolution_level: MoGe detail level from 0 through 9.
        warmup_iters: Untimed iterations.
        timed_iters: Iterations accumulated in the result.

    Returns:
        One total-only timing row for the single-image baseline.
    """
    input_hw: tuple[int, int] = (image_3hw.shape[1], image_3hw.shape[2])
    bucket_hw: tuple[int, int] = select_aspect_bucket(input_hw, ASPECT_BUCKETS)
    for _warmup_index in range(warmup_iters):
        model.infer(image_3hw, resolution_level=resolution_level, use_fp16=True)
    torch.cuda.synchronize()

    started: float = time.perf_counter()
    for _timed_index in range(timed_iters):
        model.infer(image_3hw, resolution_level=resolution_level, use_fp16=True)
    torch.cuda.synchronize()
    total_seconds: float = time.perf_counter() - started
    return BenchmarkRow(
        backend="moge-infer",
        input_hw=input_hw,
        bucket_hw=bucket_hw,
        batch_size=1,
        total_ms_per_batch=total_seconds * 1000.0 / timed_iters,
        forward_ms_per_batch=None,
        postprocess_ms_per_batch=None,
    )


def _benchmark_input(image_hw: tuple[int, int], batch_size: int) -> BenchmarkImage:
    """Load the real example when available, otherwise make a CUDA gradient.

    Args:
        image_hw: Caller input height and width.
        batch_size: Number of tiled frames to return.

    Returns:
        Tiled uint8 CUDA RGB frames and a human-readable source label.
    """
    height: int = image_hw[0]
    width: int = image_hw[1]
    image_path: Path = Path("data/examples/single-image/room.jpg")
    bgr_hw3: UInt8[np.ndarray, "h0 w0 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path.exists() else None
    if bgr_hw3 is not None:
        resized_bgr_hw3: UInt8[np.ndarray, "h w 3"] = cv2.resize(bgr_hw3, (width, height), interpolation=cv2.INTER_AREA)
        rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(resized_bgr_hw3, cv2.COLOR_BGR2RGB)
        rgb_1hw3: UInt8[Tensor, "1 h w 3"] = torch.from_numpy(np.ascontiguousarray(rgb_hw3))[None].to(device="cuda")
        rgb_bhw3: UInt8[Tensor, "b h w 3"] = rgb_1hw3.expand(batch_size, -1, -1, -1).contiguous()
        return BenchmarkImage(rgb_bhw3=rgb_bhw3, source=str(image_path))

    x_w: Float32[Tensor, "w"] = torch.linspace(0, 255, width, device="cuda", dtype=torch.float32)
    y_h: Float32[Tensor, "h"] = torch.linspace(0, 255, height, device="cuda", dtype=torch.float32)
    red_hw: Float32[Tensor, "h w"] = x_w[None, :].expand(height, width)
    green_hw: Float32[Tensor, "h w"] = y_h[:, None].expand(height, width)
    blue_hw: Float32[Tensor, "h w"] = (red_hw + green_hw) / 2.0
    synthetic_hw3: UInt8[Tensor, "h w 3"] = torch.stack((red_hw, green_hw, blue_hw), dim=-1).to(torch.uint8)
    synthetic_bhw3: UInt8[Tensor, "b h w 3"] = synthetic_hw3[None].expand(batch_size, -1, -1, -1).contiguous()
    return BenchmarkImage(rgb_bhw3=synthetic_bhw3, source="synthetic CUDA gradient")


def _print_table(rows: list[BenchmarkRow]) -> None:
    """Print benchmark results as one Markdown table.

    Args:
        rows: Timing rows in display order.
    """
    print("| Backend | Input H×W | Bucket H×W | Batch | Forward ms/batch | Postprocess ms/batch | Total ms/batch | ms/frame | frames/s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        milliseconds_per_frame: float = row.total_ms_per_batch / row.batch_size
        frames_per_second: float = 1000.0 / milliseconds_per_frame
        forward_text: str = "—" if row.forward_ms_per_batch is None else f"{row.forward_ms_per_batch:.3f}"
        postprocess_text: str = "—" if row.postprocess_ms_per_batch is None else f"{row.postprocess_ms_per_batch:.3f}"
        print(
            f"| {row.backend} | {row.input_hw[0]}×{row.input_hw[1]} | {row.bucket_hw[0]}×{row.bucket_hw[1]} | {row.batch_size} | "
            f"{forward_text} | {postprocess_text} | {row.total_ms_per_batch:.3f} | {milliseconds_per_frame:.3f} | {frames_per_second:.2f} |"
        )


def _benchmark_trt_backend(
    backend: str,
    config: Config,
    primary_input: BenchmarkImage,
    widescreen_input: BenchmarkImage,
    *,
    use_cuda_graph: bool,
) -> list[BenchmarkRow]:
    """Measure one TensorRT mode across primary batches and one 16:9 row.

    Args:
        backend: Stable table label.
        config: Benchmark configuration.
        primary_input: Tiled input for configured batch sizes.
        widescreen_input: One-frame 16:9 input.
        use_cuda_graph: Whether to capture and replay engine launches.

    Returns:
        Timing rows for supported primary batches followed by the 16:9 row.
    """
    predictor: MoGeV2TrtGeometryPredictor = MoGeV2TrtGeometryPredictor(
        encoder=config.encoder,
        resolution_level=config.resolution_level,
        batch_size=config.trt_max_batch_size,
        builder_optimization_level=config.builder_optimization_level,
        use_cuda_graph=use_cuda_graph,
    )
    rows: list[BenchmarkRow] = []
    for batch_size in config.batch_sizes:
        if batch_size > config.trt_max_batch_size:
            print(f"Skipping {backend} batch {batch_size}: TensorRT engine profile max is {config.trt_max_batch_size}.")
            continue
        rows.append(
            _benchmark_geometry(
                backend,
                predictor,
                primary_input.rgb_bhw3[:batch_size],
                config.warmup_iters,
                config.timed_iters,
            )
        )
    rows.append(
        _benchmark_geometry(
            backend,
            predictor,
            widescreen_input.rgb_bhw3,
            config.warmup_iters,
            config.timed_iters,
        )
    )
    return rows


def main(config: Config) -> None:
    """Run selected MoGe v2 backends and print their timing table.

    Args:
        config: Benchmark configuration parsed by Tyro.

    Raises:
        RuntimeError: If CUDA is unavailable or no backend is selected.
        ValueError: If batch sizes, dimensions, profile size, or iteration counts are invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("The MoGe v2 TensorRT benchmark requires CUDA.")
    if not config.batch_sizes or any(batch_size < 1 for batch_size in config.batch_sizes):
        raise ValueError(f"batch_sizes must contain positive integers, got {config.batch_sizes}.")
    if config.image_hw[0] < 1 or config.image_hw[1] < 1 or config.widescreen_image_hw[0] < 1 or config.widescreen_image_hw[1] < 1:
        raise ValueError("Benchmark image dimensions must be positive.")
    if config.trt_max_batch_size < 1:
        raise ValueError(f"trt_max_batch_size must be positive, got {config.trt_max_batch_size}.")
    if not 0 <= config.builder_optimization_level <= 5:
        raise ValueError(f"builder_optimization_level must be within [0, 5], got {config.builder_optimization_level}.")
    if config.warmup_iters < 0 or config.timed_iters < 1:
        raise ValueError("warmup_iters must be non-negative and timed_iters must be positive.")
    if not (config.run_moge_infer or config.run_torch_geometry or config.run_trt_geometry or config.run_trt_geometry_cudagraph):
        raise RuntimeError("Select at least one benchmark backend.")

    torch_batch_sizes: tuple[int, ...] = config.batch_sizes if 16 in config.batch_sizes else (*config.batch_sizes, 16)
    max_primary_batch_size: int = max(torch_batch_sizes) if config.run_torch_geometry else max(config.batch_sizes)
    primary_input: BenchmarkImage = _benchmark_input(config.image_hw, max_primary_batch_size)
    widescreen_input: BenchmarkImage = _benchmark_input(config.widescreen_image_hw, 1)
    print(f"Primary input: {primary_input.source} resized to {config.image_hw[0]}x{config.image_hw[1]}")
    print(f"16:9 input: {widescreen_input.source} resized to {config.widescreen_image_hw[0]}x{config.widescreen_image_hw[1]}")
    rows: list[BenchmarkRow] = []

    if config.run_moge_infer:
        checkpoint: tuple[str, str] = MOGE_V2_NORMAL_CHECKPOINTS[config.encoder]
        model: MoGeModel = MoGeModel.from_pretrained(checkpoint[0], revision=checkpoint[1]).to("cuda").eval()
        image_3hw: Float32[Tensor, "3 h w"] = rearrange(primary_input.rgb_bhw3[0], "h w c -> c h w").float() / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive
        rows.append(
            _benchmark_moge_infer(
                model,
                image_3hw,
                config.resolution_level,
                config.warmup_iters,
                config.timed_iters,
            )
        )
        del model
        torch.cuda.empty_cache()

    if config.run_torch_geometry:
        torch_predictor: MoGeV2TorchGeometryPredictor = MoGeV2TorchGeometryPredictor(
            encoder=config.encoder,
            resolution_level=config.resolution_level,
        )
        for batch_size in torch_batch_sizes:
            rows.append(
                _benchmark_geometry(
                    "torch-geometry",
                    torch_predictor,
                    primary_input.rgb_bhw3[:batch_size],
                    config.warmup_iters,
                    config.timed_iters,
                )
            )
        del torch_predictor
        torch.cuda.empty_cache()

    if config.run_trt_geometry:
        standard_rows: list[BenchmarkRow] = _benchmark_trt_backend(
            "trt-geometry",
            config,
            primary_input,
            widescreen_input,
            use_cuda_graph=False,
        )
        rows.extend(standard_rows)
        torch.cuda.empty_cache()

    if config.run_trt_geometry_cudagraph:
        cuda_graph_rows: list[BenchmarkRow] = _benchmark_trt_backend(
            "trt-geometry-cudagraph",
            config,
            primary_input,
            widescreen_input,
            use_cuda_graph=True,
        )
        rows.extend(cuda_graph_rows)
        torch.cuda.empty_cache()

    _print_table(rows)
