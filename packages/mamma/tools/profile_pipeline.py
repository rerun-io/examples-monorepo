"""Kernel-level profiling of the steady-state streaming pipeline.

Wraps a warm steady-state slice in ``torch.profiler`` (CPU+CUDA) and reports:
  - per-stage rollup (``stage::*`` record_function ranges): CPU wall vs CUDA time
  - top CUDA kernels by self time (the real bottleneck list)
  - a chrome trace (open in https://ui.perfetto.dev) for flamegraph inspection

Per the TensorRT benchmarking guidance: profile only after warmup, measure a
window long enough to amortize periodic work (fit_stride cycles, a re-detect),
and attribute via profiler ranges rather than wall-clock around async launches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.video_io import TorchCodecMultiVideoReader

from mamma.calibration.npz_contract import CameraCalibration
from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import StreamingPipeline
from mamma.fitting.stage import FittingStage
from mamma.fitting.window_fitter import FitterConfig
from mamma.landmarks.estimator import LandmarkEstimator
from mamma.tracking.tracker import MultiViewTracker, TrackerConfig
from mamma.viz.stream_logger import StreamLogger


@dataclass
class ProfileConfig:
    rr_config: RerunTyroConfig
    """Rerun config (headless recommended; logging cost is part of the profile)."""
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(expected_subjects=1))
    """Tracker settings."""
    fitter: FitterConfig = field(default_factory=FitterConfig)
    """Fitter settings."""
    data_dir: Path = Path("data/inputs/indoors/crossing_arms")
    """Capture to profile on."""
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet weights."""
    resize_hw: tuple[int, int] = (720, 1280)
    """Engine resolution."""
    warmup_frames: int = 45
    """Steady-state warmup before the profiled window (must exceed bootstrap)."""
    profile_frames: int = 48
    """Profiled ticks (a multiple of fit_stride; spans >=1 re-detect-free cycle)."""
    trace_path: Path = Path("/tmp/mamma-profile/trace.json")
    """Chrome trace output (open in ui.perfetto.dev)."""
    top_kernels: int = 30
    """Kernel rows to print."""
    device: str = "cuda"
    """Compute device."""


def main(config: ProfileConfig) -> None:
    config.tracker.device = config.device
    config.fitter.device = config.device
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)
    scaled: list[CameraCalibration] = [
        cam.scaled_to(height=config.resize_hw[0], width=config.resize_hw[1]) for cam in sequence.cameras
    ]
    reader = TorchCodecMultiVideoReader(list(sequence.video_paths), device=config.device, resize_hw=config.resize_hw)
    logger = StreamLogger(sequence, resize_hw=config.resize_hw)
    pipeline = StreamingPipeline(
        sequence,
        reader,
        logger,
        tracker=MultiViewTracker(scaled, config.tracker),
        landmarks=LandmarkEstimator(config.mammanet_weights, device=config.device),
        fitting=FittingStage(scaled, config.fitter),
    )

    print(f"warmup: {config.warmup_frames} ticks")
    pipeline.run(max_frames=config.warmup_frames)
    torch.cuda.synchronize()

    print(f"profiling: {config.profile_frames} ticks")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        stats = pipeline.run(max_frames=config.profile_frames, start_frame=config.warmup_frames)
        torch.cuda.synchronize()

    ticks: int = stats.ticks
    print(f"\nprofiled {ticks} ticks in {stats.elapsed_s:.2f}s ({1000 * stats.elapsed_s / ticks:.1f} ms/tick wall)")

    # ── Per-stage rollup from record_function ranges ──
    events = prof.key_averages()
    print(f"\n{'stage':<22}{'cpu_ms/tick':>13}{'cuda_ms/tick':>14}{'calls':>7}")
    for evt in sorted(events, key=lambda e: -e.self_device_time_total):
        if evt.key.startswith("stage::"):
            cpu_ms: float = evt.cpu_time_total / 1000.0 / ticks
            cuda_ms: float = evt.device_time_total / 1000.0 / ticks
            print(f"{evt.key:<22}{cpu_ms:>13.2f}{cuda_ms:>14.2f}{evt.count:>7}")

    # ── Top CUDA kernels by self device time ──
    print(f"\ntop {config.top_kernels} CUDA kernels by self time (per profiled window):")
    print(f"{'kernel':<72}{'cuda_ms':>9}{'calls':>8}")
    kernel_events = [e for e in events if e.self_device_time_total > 0 and not e.key.startswith("stage::")]
    for evt in sorted(kernel_events, key=lambda e: -e.self_device_time_total)[: config.top_kernels]:
        name: str = evt.key if len(evt.key) <= 70 else evt.key[:67] + "..."
        print(f"{name:<72}{evt.self_device_time_total / 1000.0:>9.1f}{evt.count:>8}")

    total_cuda_ms: float = sum(e.self_device_time_total for e in kernel_events) / 1000.0
    print(f"\ntotal CUDA kernel time: {total_cuda_ms:.0f} ms over {ticks} ticks = {total_cuda_ms / ticks:.1f} ms/tick")
    print(f"wall {1000 * stats.elapsed_s / ticks:.1f} ms/tick -> CPU/launch overhead = {1000 * stats.elapsed_s / ticks - total_cuda_ms / ticks:.1f} ms/tick")

    config.trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(config.trace_path))
    print(f"\nchrome trace: {config.trace_path} (open in ui.perfetto.dev)")


if __name__ == "__main__":
    main(tyro.cli(ProfileConfig))
