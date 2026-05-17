"""Benchmark PyTorch and TensorRT Sapiens2 pose heatmap inference."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import tyro
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image

from sapiens2_pose.api.image_pose import DeviceChoice, ModelSize, get_pose_model, resolve_device
from sapiens2_pose.api.pose_artifact import PosePredictionArtifact, load_pose_prediction_artifact
from sapiens2_pose.api.tensorrt_pose import TensorRtPoseHeatmapRunner
from sapiens2_pose.sapiens_lite.pose import MODEL_SPECS, ImagePreprocessor, prepare_pose_sample


@dataclass(frozen=True, slots=True)
class BenchmarkTensorRtPoseCli:
    """CLI arguments for pose heatmap backend benchmarking."""

    image_path: Path
    """Path to the input image."""
    baseline_artifact_path: Path
    """Path to the pristine baseline numeric artifact; its bboxes are reused for both backends."""
    engine_path: Path
    """Path to the TensorRT engine to benchmark."""
    output_json_path: Path | None = None
    """Optional path where the benchmark JSON summary should be written."""
    model_size: ModelSize = "0.4B"
    """Sapiens2 pose model size."""
    device: DeviceChoice = "cuda"
    """Compute device selection."""
    warmup_runs: int = 5
    """Number of warmup runs before timing."""
    timed_runs: int = 30
    """Number of timed runs."""


@dataclass(frozen=True, slots=True)
class PreparedPoseBatch:
    """Preprocessed Sapiens2 pose tensor batch."""

    inputs: torch.Tensor
    """Preprocessed NCHW pose input tensor."""
    bboxes: Float32[ndarray, "n 4"]
    """Person boxes used to create the input tensor."""


RunnerFn = Callable[[torch.Tensor], torch.Tensor]


def main(args: BenchmarkTensorRtPoseCli) -> None:
    """Benchmark PyTorch and TensorRT heatmap execution on the same pose crop."""
    if args.warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative.")
    if args.timed_runs < 1:
        raise ValueError("timed_runs must be at least 1.")
    resolved_device: str = resolve_device(args.device)
    if resolved_device != "cuda":
        raise ValueError("Benchmarking TensorRT requires CUDA.")

    baseline_artifact: PosePredictionArtifact = load_pose_prediction_artifact(args.baseline_artifact_path)
    image_pil: Image.Image = Image.open(args.image_path).convert("RGB")
    image_rgb: UInt8[ndarray, "h w 3"] = np.asarray(image_pil, dtype=np.uint8)
    batch: PreparedPoseBatch = _prepare_pose_batch(
        image_rgb=image_rgb,
        bboxes=baseline_artifact.bboxes,
        model_size=args.model_size,
        device=resolved_device,
    )

    pytorch_model: Any = get_pose_model(args.model_size, device="cuda")
    pytorch_runner: RunnerFn = _make_pytorch_runner(pytorch_model)
    tensorrt_runner: TensorRtPoseHeatmapRunner = TensorRtPoseHeatmapRunner(args.engine_path, device="cuda")

    pytorch_heatmaps: torch.Tensor = pytorch_runner(batch.inputs)
    tensorrt_heatmaps: torch.Tensor = tensorrt_runner(batch.inputs)
    heatmap_metrics: dict[str, float] = _heatmap_error_metrics(pytorch_heatmaps, tensorrt_heatmaps)

    pytorch_timing: dict[str, float] = _time_runner(
        pytorch_runner,
        batch.inputs,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    tensorrt_timing: dict[str, float] = _time_runner(
        tensorrt_runner,
        batch.inputs,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    speedup: float = pytorch_timing["median_ms"] / tensorrt_timing["median_ms"]
    batch_size: int = int(batch.inputs.shape[0])
    summary: dict[str, object] = {
        "image_path": str(args.image_path),
        "baseline_artifact_path": str(args.baseline_artifact_path),
        "engine_path": str(args.engine_path),
        "model_size": args.model_size,
        "batch_size": batch_size,
        "input_shape": list(batch.inputs.shape),
        "pytorch": pytorch_timing,
        "tensorrt": tensorrt_timing,
        "tensorrt_mode": "bf16_static_b1_cuda_graph",
        "speedup_median": speedup,
        "heatmap_error": heatmap_metrics,
    }
    summary_json: str = json.dumps(summary, indent=2, sort_keys=True)
    print(summary_json)
    if args.output_json_path is not None:
        args.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_json_path.write_text(summary_json + "\n")


def _make_pytorch_runner(model: Any) -> RunnerFn:
    def run(inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            heatmaps: torch.Tensor = model(inputs)
        return heatmaps

    return run


def _prepare_pose_batch(
    *,
    image_rgb: UInt8[ndarray, "h w 3"],
    bboxes: Float32[ndarray, "n 4"],
    model_size: ModelSize,
    device: str,
) -> PreparedPoseBatch:
    spec: Any = MODEL_SPECS[model_size]
    preprocessor: ImagePreprocessor = ImagePreprocessor().to(device)
    image_bgr: UInt8[ndarray, "h w 3"] = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    inputs_list: list[torch.Tensor] = []
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if bboxes_f32.shape[0] == 0:
        raise ValueError("Benchmarking requires at least one baseline person box.")
    bboxes_f32 = bboxes_f32[:1]
    for bbox in bboxes_f32:
        data: dict[str, Any] = prepare_pose_sample(image_bgr, bbox, input_size=spec.input_size)
        processed: dict[str, Any] = preprocessor(data)
        inputs_list.append(processed["inputs"])
    inputs: torch.Tensor = torch.cat(inputs_list, dim=0).to(device).contiguous()
    return PreparedPoseBatch(inputs=inputs, bboxes=bboxes_f32)


def _time_runner(
    runner: RunnerFn,
    inputs: torch.Tensor,
    *,
    warmup_runs: int,
    timed_runs: int,
) -> dict[str, float]:
    for _ in range(warmup_runs):
        _ = runner(inputs)
    torch.cuda.synchronize()

    durations_ms: list[float] = []
    for _ in range(timed_runs):
        start_time: float = time.perf_counter()
        _ = runner(inputs)
        torch.cuda.synchronize()
        end_time: float = time.perf_counter()
        duration_ms: float = (end_time - start_time) * 1000.0
        durations_ms.append(duration_ms)

    durations: Float32[ndarray, "..."] = np.asarray(durations_ms, dtype=np.float32)
    return {
        "mean_ms": float(durations.mean()),
        "median_ms": float(np.median(durations)),
        "min_ms": float(durations.min()),
        "max_ms": float(durations.max()),
    }


def _heatmap_error_metrics(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    baseline_np: Float32[ndarray, "n k h w"] = baseline.detach().float().cpu().numpy().astype(np.float32, copy=False)
    candidate_np: Float32[ndarray, "n k h w"] = candidate.detach().float().cpu().numpy().astype(np.float32, copy=False)
    difference: Float32[ndarray, "n k h w"] = np.abs(candidate_np - baseline_np).astype(np.float32, copy=False)
    squared_difference: Float32[ndarray, "n k h w"] = np.square(difference).astype(np.float32, copy=False)
    baseline_scale: float = float(np.max(np.abs(baseline_np)))
    relative_scale: float = max(baseline_scale, 1e-6)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(squared_difference.mean())),
        "max_abs_fraction_of_baseline_max": float(difference.max() / relative_scale),
        "rmse_fraction_of_baseline_max": float(np.sqrt(squared_difference.mean()) / relative_scale),
        "baseline_max_abs": baseline_scale,
    }


if __name__ == "__main__":
    main(tyro.cli(BenchmarkTensorRtPoseCli))
