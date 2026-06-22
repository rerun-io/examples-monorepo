"""Dynamic golden gate: run the streaming pipeline on the full running_jumping
clip (478 frames, runs + jumps) and gate per-frame PVE against the original
5-stage DAG's SMPL-X output. Unlike validate_golden's 30-frame quasi-static
slice, this covers fast motion — the failure mode the strides regressions hid.

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
    golden_npz: Path = Path("data/golden/ma_3d_dynamic/outdoors/running_jumping/verts_joints_body_id-00.npz")
    """Frozen original-DAG verts/joints for this clip (ma_3d output)."""
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet weights."""
    resize_hw: tuple[int, int] = (720, 1280)
    """Engine resolution."""
    trt_engine: Path | None = None
    """Optional MammaNet TensorRT engine plan."""
    pve_p95_tolerance_mm: float = 30.0
    """PASS bound on the 95th-percentile per-frame PVE vs the original DAG."""
    pve_max_tolerance_mm: float = 30.0
    """PASS bound on the worst single frame's PVE vs the original DAG."""
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
    golden = np.load(config.golden_npz, allow_pickle=True)
    golden_verts: Float32[ndarray, "f v 3"] = golden["pred_vertices"]
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
    print(f"\nDYNAMIC GOLDEN GATE ({len(pve)} frames vs original DAG, {config.golden_npz.name}):")
    print(f"  PVE mean {pve.mean():6.1f} mm   median {np.median(pve):6.1f} mm")
    print(f"  PVE p95  {p95:6.1f} mm  (tolerance {config.pve_p95_tolerance_mm} mm)  {'PASS' if p95 <= config.pve_p95_tolerance_mm else 'FAIL'}")
    print(f"  PVE max  {pmax:6.1f} mm  (tolerance {config.pve_max_tolerance_mm} mm)  {'PASS' if pmax <= config.pve_max_tolerance_mm else 'FAIL'}")
    print(f"  worst frames: {', '.join(f'f{f}(t={f / sequence.fps:.1f}s)={pve_by_frame[f]:.0f}mm' for f in worst)}")
    ok: bool = p95 <= config.pve_p95_tolerance_mm and pmax <= config.pve_max_tolerance_mm
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(ValidateDynamicConfig)))
