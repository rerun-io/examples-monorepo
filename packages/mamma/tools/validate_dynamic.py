"""Dynamic golden gate: run the streaming pipeline on the full running_jumping
clip (478 frames, runs + jumps) and gate per-frame PVE against the frozen
reference (``data/golden/ma_3d_dynamic/outdoors/running_jumping/reference.rrd``,
see ``mamma.eval.dynamic_reference``). Unlike validate_golden's 30-frame
quasi-static slice, this covers fast motion — the failure mode the strides
regressions hid.

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from jaxtyping import Float32
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import ResultCollector, StreamingPipeline, build_streaming_pipeline
from mamma.engine.types import CameraTracks
from mamma.eval.dynamic_reference import load_reference
from mamma.fitting.stage import TickFitOutput
from mamma.fitting.window_fitter import FitterConfig
from mamma.landmarks.estimator import CameraLandmarks
from mamma.tracking.tracker import TrackerConfig


class _FitsOnlyCollector(ResultCollector):
    """Collector that drops tracks/landmarks references.

    The full collector pins every tick's GPU mask tensors for the whole run —
    ~33 MB/tick at 4K (15 GB over the clip). This gate only reads fits.
    """

    def collect(
        self,
        frame_idx: int,
        tracks: list[CameraTracks] | None,
        landmarks: list[CameraLandmarks] | None,
        fit_output: TickFitOutput | None = None,
    ) -> None:
        super().collect(frame_idx, None, None, fit_output)


@dataclass
class ValidateDynamicConfig:
    rr_config: RerunTyroConfig
    """Rerun behavior (defaults headless-friendly; pass --rr-config.save to keep an RRD)."""
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(expected_subjects=1))
    """Tracker settings; running_jumping is single-subject."""
    fitter: FitterConfig = field(default_factory=lambda: FitterConfig(emit_stride=1))
    """Window fitter settings (emit every frame — the comparison needs fresh meshes)."""
    data_dir: Path = Path("data/inputs/outdoors/running_jumping")
    """Dynamic capture inputs (MAMMA NPZ layout)."""
    reference_rrd: Path = Path("data/golden/ma_3d_dynamic/outdoors/running_jumping/reference.rrd")
    """Frozen reference vertices (``mamma.eval.dynamic_reference``): the dataset's MAMMA prediction through SMPL-X, frozen
    on 2026-11-21 because the original-DAG output was lost. A missing file fails the gate."""
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet weights."""
    resize_hw: tuple[int, int] = (720, 1280)
    """Engine resolution."""
    trt_engine: Path | None = None
    """Optional MammaNet TensorRT engine plan."""
    pve_p95_tolerance_mm: float = 30.0
    """PASS bound on the 95th-percentile per-frame PVE vs the reference."""
    pve_max_tolerance_mm: float = 30.0
    """PASS bound on the worst single frame's PVE vs the reference."""
    skip_first_frames: int = 8
    """Frames excluded from the gate at clip start (window fill: no fit exists yet)."""
    hires_crops: bool = True
    """Decode native 4K and sample landmark crops from it (tracking/logging
    stay at resize_hw). Closes most of the crop-quality gap vs the original's
    4K-fed landmarks at a fraction of full-4K cost."""
    chunk_size: int = 32
    """Frames decoded per camera per chunk (threaded path). Lower at high
    resolutions: a 32-frame 4-cam 4K chunk is ~3 GB of VRAM, two in flight."""
    device: str = "cuda"
    """Compute device."""


def main(config: ValidateDynamicConfig) -> int:
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)
    try:
        reference_frames, golden_verts, reference = load_reference(config.reference_rrd)
    except (FileNotFoundError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1
    if reference_frames.tolist() != list(range(sequence.frame_count)):
        print(f"FAIL: {config.reference_rrd} must hold every frame 0..{sequence.frame_count - 1} of {sequence.name} in order")
        return 1
    print(f"sequence {sequence.name}: {sequence.frame_count} frames; golden verts {golden_verts.shape}")

    collector: ResultCollector = _FitsOnlyCollector()
    pipeline: StreamingPipeline = build_streaming_pipeline(
        sequence,
        resize_hw=config.resize_hw,
        device=config.device,
        tracker_config=config.tracker,
        fitter_config=config.fitter,
        mammanet_weights=config.mammanet_weights,
        trt_engine=config.trt_engine,
        collector=collector,
        hires_crops=config.hires_crops,
    )
    stats = pipeline.run(chunk_size=config.chunk_size)
    if pipeline.fitting is not None:
        # Fixed-lag emission holds the last emit_lag frames; flush them.
        for tail in pipeline.fitting.drain():
            collector.collect(-1, None, None, tail)
    pipeline.close()
    print(f"pipeline: {stats.ticks} ticks in {stats.elapsed_s:.1f}s ({stats.ticks_per_s:.1f} ticks/s)")
    print(f"decode: {stats.decode_backend}")

    # Per-frame PVE over every emitted frame inside the golden range.
    pve_by_frame: dict[int, float] = {}
    for fit_out in collector.fits:
        if fit_out is None or 0 not in fit_out.fits:
            continue
        fit = fit_out.fits[0]
        f: int = fit.frame_idx
        if f < config.skip_first_frames or f >= golden_verts.shape[0]:
            continue
        pve_by_frame[f] = float(np.linalg.norm(fit.vertices - golden_verts[f], axis=-1).mean()) * 1000.0

    expected: int = golden_verts.shape[0] - config.skip_first_frames
    if len(pve_by_frame) < expected:
        print(f"FAIL: only {len(pve_by_frame)}/{expected} frames emitted a fit")
        return 1

    pve: Float32[ndarray, "f"] = np.array([pve_by_frame[f] for f in sorted(pve_by_frame)], dtype=np.float32)
    p95: float = float(np.percentile(pve, 95))
    pmax: float = float(pve.max())
    worst: list[int] = [f for f, _ in sorted(pve_by_frame.items(), key=lambda kv: -kv[1])[:5]]
    print(f"\nDYNAMIC GOLDEN GATE ({len(pve)} frames {sorted(pve_by_frame)[0]}..{sorted(pve_by_frame)[-1]} vs {reference}):")
    print(f"  PVE mean {pve.mean():6.1f} mm   median {np.median(pve):6.1f} mm")
    print(f"  PVE p95  {p95:6.1f} mm  (tolerance {config.pve_p95_tolerance_mm} mm)  {'PASS' if p95 <= config.pve_p95_tolerance_mm else 'FAIL'}")
    print(f"  PVE max  {pmax:6.1f} mm  (tolerance {config.pve_max_tolerance_mm} mm)  {'PASS' if pmax <= config.pve_max_tolerance_mm else 'FAIL'}")
    print(f"  worst frames: {', '.join(f'f{f}(t={f / sequence.fps:.1f}s)={pve_by_frame[f]:.0f}mm' for f in worst)}")
    ok: bool = p95 <= config.pve_p95_tolerance_mm and pmax <= config.pve_max_tolerance_mm
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(ValidateDynamicConfig)))
