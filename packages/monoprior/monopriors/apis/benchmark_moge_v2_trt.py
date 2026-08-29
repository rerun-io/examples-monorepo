"""Benchmark vendored, batched torch, and TensorRT MoGe v2 predictors."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float32, UInt8
from torch import Tensor

from monopriors.models.moge_v2 import (
    ASPECT_BUCKETS,
    Encoder,
    MoGeV2GeometryGraphOutput,
    MoGeV2TorchPredictor,
    MoGeV2TrtPredictor,
    load_pinned_moge_v2,
    postprocess_moge_v2_geometry,
    resolve_num_tokens,
    token_grid_hw,
)
from monopriors.third_party.moge.model.v2 import MoGeModel

Backend: TypeAlias = Literal["moge-infer", "torch-geometry", "trt-geometry", "trt-geometry-cudagraph"]
GeometryPredictor: TypeAlias = MoGeV2TorchPredictor | MoGeV2TrtPredictor

ALL_BACKENDS: tuple[Backend, ...] = ("moge-infer", "torch-geometry", "trt-geometry", "trt-geometry-cudagraph")


@dataclass(frozen=True, slots=True)
class Config:
    """MoGe v2 batched predictor benchmark configuration."""

    encoder: Encoder = "vitl"
    """DINOv2 encoder size shared by every selected backend."""
    image_hw: tuple[int, int] = ASPECT_BUCKETS[0]
    """Primary caller height and width; defaults to the 4:3 network bucket."""
    widescreen_image_hw: tuple[int, int] = (1080, 1920)
    """Caller height and width for one 16:9 row per backend."""
    resolution_level: int = 9
    """MoGe detail level from 0 through 9."""
    batch_sizes: tuple[int, ...] = (1, 4, 8)
    """Primary batch sizes measured for every backend."""
    warmup_iters: int = 3
    """Untimed iterations before each backend and batch measurement."""
    timed_iters: int = 20
    """Iterations accumulated for each table row."""
    backends: tuple[Backend, ...] = ALL_BACKENDS
    """Backends measured in display order."""


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    """One backend, input shape, bucket, and batch-size timing result."""

    backend: Backend
    """Stable backend label."""
    input_hw: tuple[int, int]
    """Caller input height and width."""
    bucket_hw: tuple[int, int]
    """MoGe network height and width observed during warmup."""
    batch_size: int
    """Frames in each predictor call."""
    total_ms_per_batch: float
    """Unsplit predictor wall time per batch with one trailing synchronization."""
    forward_ms_per_batch: float | None
    """Diagnostic synchronized preprocess plus model or engine time."""
    postprocess_ms_per_batch: float | None
    """Diagnostic synchronized camera-recovery and output time."""


def _benchmark_geometry(
    backend: Backend,
    predictor: GeometryPredictor,
    rgb_bhw3: UInt8[Tensor, "b h w 3"],
    warmup_iters: int,
    timed_iters: int,
) -> BenchmarkRow:
    """Measure split diagnostics and an unsplit predictor total.

    Args:
        backend: Stable backend label.
        predictor: Torch or TensorRT staged geometry predictor.
        rgb_bhw3: uint8 CUDA RGB batch shaped ``b h w 3``.
        warmup_iters: Untimed iterations.
        timed_iters: Iterations accumulated in the result.

    Returns:
        Per-batch timing row with diagnostic forward and postprocess columns.
    """
    output_hw: tuple[int, int] = (rgb_bhw3.shape[1], rgb_bhw3.shape[2])
    warmup_graph_output: MoGeV2GeometryGraphOutput = predictor.forward_geometry(rgb_bhw3)
    postprocess_moge_v2_geometry(warmup_graph_output, output_hw)
    bucket_hw: tuple[int, int] = (warmup_graph_output.points_bhw3.shape[1], warmup_graph_output.points_bhw3.shape[2])
    for _warmup_index in range(max(0, warmup_iters - 1)):
        warmup_graph_output = predictor.forward_geometry(rgb_bhw3)
        postprocess_moge_v2_geometry(warmup_graph_output, output_hw)
    torch.cuda.synchronize()

    forward_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    for _timed_index in range(timed_iters):
        torch.cuda.synchronize()
        forward_started: float = time.perf_counter()
        graph_output: MoGeV2GeometryGraphOutput = predictor.forward_geometry(rgb_bhw3)
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - forward_started

        postprocess_started: float = time.perf_counter()
        postprocess_moge_v2_geometry(graph_output, output_hw)
        torch.cuda.synchronize()
        postprocess_seconds += time.perf_counter() - postprocess_started

    total_started: float = time.perf_counter()
    for _timed_index in range(timed_iters):
        predictor.predict_geometry(rgb_bhw3)
    torch.cuda.synchronize()
    total_seconds: float = time.perf_counter() - total_started

    scale_to_ms_per_batch: float = 1000.0 / timed_iters
    return BenchmarkRow(
        backend=backend,
        input_hw=output_hw,
        bucket_hw=bucket_hw,
        batch_size=rgb_bhw3.shape[0],
        total_ms_per_batch=total_seconds * scale_to_ms_per_batch,
        forward_ms_per_batch=forward_seconds * scale_to_ms_per_batch,
        postprocess_ms_per_batch=postprocess_seconds * scale_to_ms_per_batch,
    )


def _benchmark_moge_infer(
    model: MoGeModel,
    image_b3hw: Float32[Tensor, "b 3 h w"],
    resolution_level: int,
    warmup_iters: int,
    timed_iters: int,
) -> BenchmarkRow:
    """Measure the vendored batched inference path.

    Args:
        model: Pinned normal-capable MoGe v2 model on CUDA.
        image_b3hw: Float32 CUDA RGB batch shaped ``b 3 h w`` in ``[0, 1]``.
        resolution_level: MoGe detail level from 0 through 9.
        warmup_iters: Untimed iterations.
        timed_iters: Iterations accumulated in the result.

    Returns:
        One total-only timing row for the vendored baseline.
    """
    input_hw: tuple[int, int] = (image_b3hw.shape[2], image_b3hw.shape[3])
    num_tokens: int = resolve_num_tokens(resolution_level)
    bucket_hw: tuple[int, int] = token_grid_hw(input_hw[1] / input_hw[0], num_tokens)
    for _warmup_index in range(warmup_iters):
        model.infer(image_b3hw, resolution_level=resolution_level, use_fp16=True)
    torch.cuda.synchronize()

    started: float = time.perf_counter()
    for _timed_index in range(timed_iters):
        model.infer(image_b3hw, resolution_level=resolution_level, use_fp16=True)
    torch.cuda.synchronize()
    total_seconds: float = time.perf_counter() - started
    return BenchmarkRow(
        backend="moge-infer",
        input_hw=input_hw,
        bucket_hw=bucket_hw,
        batch_size=image_b3hw.shape[0],
        total_ms_per_batch=total_seconds * 1000.0 / timed_iters,
        forward_ms_per_batch=None,
        postprocess_ms_per_batch=None,
    )


def _benchmark_input(image_hw: tuple[int, int], batch_size: int) -> UInt8[Tensor, "b h w 3"]:
    """Load and tile the required real benchmark image on CUDA.

    Args:
        image_hw: Caller input height and width.
        batch_size: Number of tiled frames to return.

    Returns:
        Tiled uint8 CUDA RGB frames at caller resolution.

    Raises:
        FileNotFoundError: If the Pixi-managed example image is unavailable.
    """
    height: int = image_hw[0]
    width: int = image_hw[1]
    image_path: Path = Path("data/examples/single-image/room.jpg")
    bgr_hw3: UInt8[np.ndarray, "h0 w0 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr_hw3 is None:
        raise FileNotFoundError(
            f"Could not read {image_path}; run `pixi run -e monoprior --frozen _monoprior-download-single-image`."
        )
    resized_bgr_hw3: UInt8[np.ndarray, "h w 3"] = cv2.resize(bgr_hw3, (width, height), interpolation=cv2.INTER_AREA)
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(resized_bgr_hw3, cv2.COLOR_BGR2RGB)
    rgb_1hw3: UInt8[Tensor, "1 h w 3"] = torch.from_numpy(np.ascontiguousarray(rgb_hw3))[None].to(device="cuda")
    return rgb_1hw3.expand(batch_size, -1, -1, -1).contiguous()


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


def main(config: Config) -> None:
    """Run selected MoGe v2 backends and print their timing table.

    Args:
        config: Benchmark configuration parsed by Tyro.

    Raises:
        RuntimeError: If CUDA is unavailable.
        ValueError: If no backend is selected or timed iterations are invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("The MoGe v2 TensorRT benchmark requires CUDA.")
    if config.timed_iters < 1:
        raise ValueError("timed_iters must be positive.")
    if not config.backends:
        raise ValueError("Select at least one benchmark backend.")

    max_batch_size: int = max(config.batch_sizes)
    primary_rgb_bhw3: UInt8[Tensor, "b h w 3"] = _benchmark_input(config.image_hw, max_batch_size)
    widescreen_rgb_bhw3: UInt8[Tensor, "1 h w 3"] = _benchmark_input(config.widescreen_image_hw, 1)
    rows: list[BenchmarkRow] = []

    for backend in config.backends:
        if backend == "moge-infer":
            loaded: tuple[MoGeModel, int] = load_pinned_moge_v2(config.encoder, config.resolution_level)
            model: MoGeModel = loaded[0]
            primary_image_b3hw: Float32[Tensor, "b 3 h w"] = rearrange(primary_rgb_bhw3, "b h w c -> b c h w").float() / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive
            widescreen_image_b3hw: Float32[Tensor, "1 3 h w"] = rearrange(widescreen_rgb_bhw3, "b h w c -> b c h w").float() / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive
            for batch_size in config.batch_sizes:
                rows.append(_benchmark_moge_infer(model, primary_image_b3hw[:batch_size], config.resolution_level, config.warmup_iters, config.timed_iters))
            rows.append(_benchmark_moge_infer(model, widescreen_image_b3hw, config.resolution_level, config.warmup_iters, config.timed_iters))
            del model
        else:
            predictor: GeometryPredictor
            if backend == "torch-geometry":
                predictor = MoGeV2TorchPredictor(
                    encoder=config.encoder,
                    heads="geometry",
                    network_hw_options=ASPECT_BUCKETS,
                    resolution_level=config.resolution_level,
                )
            else:
                predictor = MoGeV2TrtPredictor(
                    encoder=config.encoder,
                    heads="geometry",
                    resolution_level=config.resolution_level,
                    batch_size=max_batch_size,
                    use_cuda_graph=backend == "trt-geometry-cudagraph",
                )
            for batch_size in config.batch_sizes:
                rows.append(_benchmark_geometry(backend, predictor, primary_rgb_bhw3[:batch_size], config.warmup_iters, config.timed_iters))
            rows.append(_benchmark_geometry(backend, predictor, widescreen_rgb_bhw3, config.warmup_iters, config.timed_iters))
            del predictor
        torch.cuda.empty_cache()

    _print_table(rows)
