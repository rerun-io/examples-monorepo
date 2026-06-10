"""Golden gate: run the streaming pipeline on the crossing_arms golden slice and
gate final-3D agreement (MPJPE/PVE) against the frozen original-pipeline output.

Exit code 0 = PASS, 1 = FAIL. 2D landmark deltas are printed as diagnostics.
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
from simplecv.video_io import TorchCodecMultiVideoReader

from mamma.calibration.npz_contract import CameraCalibration
from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import ResultCollector, StreamingPipeline
from mamma.eval.golden import GOLDEN_FRAME_END, GOLDEN_FRAME_START, fit3d_comparison, landmarks_diagnostics
from mamma.fitting.stage import FittingStage
from mamma.fitting.window_fitter import FitResult, FitterConfig
from mamma.landmarks.estimator import LandmarkEstimator
from mamma.tracking.tracker import MultiViewTracker, TrackerConfig
from mamma.viz.stream_logger import StreamLogger


@dataclass
class ValidateConfig:
    rr_config: RerunTyroConfig
    """Rerun behavior (defaults headless-friendly; pass --rr-config.save to keep an RRD)."""
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(expected_subjects=1))
    """Tracker settings; crossing_arms is single-subject."""
    fitter: FitterConfig = field(default_factory=FitterConfig)
    """Window fitter settings."""
    data_dir: Path = Path("data/inputs/indoors/crossing_arms")
    """Golden capture inputs."""
    golden_dir: Path = Path("data/golden")
    """Golden outputs root (ma_2d/ma_3d baseline trees)."""
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet weights."""
    resize_hw: tuple[int, int] = (720, 1280)
    """Engine resolution."""
    trt_engine: Path | None = None
    """Optional MammaNet TensorRT engine plan."""
    mpjpe_tolerance_mm: float = 30.0
    """PASS bound on MPJPE vs golden. Justification: streaming defaults deliver
    ~20mm; the golden is itself a 0.15-iter-scale run and the streaming rewrite
    is a different operating point (720p, causal) — see implementation-notes.html."""
    pve_tolerance_mm: float = 30.0
    """PASS bound on PVE vs golden."""
    device: str = "cuda"
    """Compute device."""


def main(config: ValidateConfig) -> int:
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)
    scaled_cams: list[CameraCalibration] = [
        cam.scaled_to(height=config.resize_hw[0], width=config.resize_hw[1]) for cam in sequence.cameras
    ]
    config.tracker.device = config.device
    config.fitter.device = config.device

    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(
        list(sequence.video_paths), device=config.device, resize_hw=config.resize_hw
    )
    logger: StreamLogger = StreamLogger(sequence, resize_hw=config.resize_hw)
    tracker: MultiViewTracker = MultiViewTracker(scaled_cams, config.tracker)
    estimator: LandmarkEstimator = LandmarkEstimator(config.mammanet_weights, device=config.device, engine_path=config.trt_engine)
    fitting: FittingStage = FittingStage(scaled_cams, config.fitter)
    collector: ResultCollector = ResultCollector()
    pipeline: StreamingPipeline = StreamingPipeline(
        sequence, reader, logger, tracker=tracker, landmarks=estimator, fitting=fitting, collector=collector
    )

    # Start one window early so the fitter has emitted every golden-slice frame.
    start_frame: int = GOLDEN_FRAME_START - config.fitter.window_size
    n_frames: int = GOLDEN_FRAME_END - start_frame
    stats = pipeline.run(max_frames=n_frames, start_frame=start_frame)
    print(f"pipeline: {stats.ticks} ticks in {stats.elapsed_s:.1f}s ({stats.ticks_per_s:.1f} ticks/s)")

    # Collect per-frame outputs for the golden slice.
    golden_frames: range = range(GOLDEN_FRAME_START, GOLDEN_FRAME_END)
    fits_by_frame: dict[int, FitResult] = {}
    landmarks_by_frame: dict[int, list] = {}
    for frame_idx, lms, fit_out in zip(collector.frame_indices, collector.landmarks, collector.fits, strict=True):
        if fit_out is not None and 0 in fit_out.fits:
            fits_by_frame[fit_out.fits[0].frame_idx] = fit_out.fits[0]
        if lms is not None:
            landmarks_by_frame[frame_idx] = lms

    missing: list[int] = [f for f in golden_frames if f not in fits_by_frame]
    if missing:
        print(f"FAIL: no fit emitted for golden frames {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return 1

    # --- Diagnostics: 2D landmarks vs golden ma_2d (report-only) ---
    cam_names: list[str] = sequence.camera_names
    pred_lm: dict[str, Float32[ndarray, "f j 3"]] = {}
    pred_vis: dict[str, Float32[ndarray, "f j"]] = {}
    for cam_idx, cam in enumerate(cam_names):
        per_frame: list[ndarray] = []
        per_vis: list[ndarray] = []
        for f in golden_frames:
            lms = landmarks_by_frame.get(f)
            result = lms[cam_idx].get(0) if lms is not None else None
            if result is None:
                per_frame.append(np.zeros((512, 3), np.float32))
                per_vis.append(np.zeros(512, np.float32))
            else:
                per_frame.append(result.joints2d.cpu().numpy())
                per_vis.append(result.visibility.cpu().numpy())
        pred_lm[cam] = np.stack(per_frame)
        pred_vis[cam] = np.stack(per_vis)
    scale: float = sequence.cameras[0].width / config.resize_hw[1]
    diags = landmarks_diagnostics(
        config.golden_dir / "ma_2d/baseline/indoors/crossing_arms", cam_names, pred_lm, pred_vis, pred_scale_to_golden=scale
    )
    print("\n2D landmark diagnostics vs golden ma_2d (report-only, golden px space):")
    for d in diags:
        print(f"  {d.camera}: px_p50={d.px_p50:.2f} px_p99={d.px_p99:.2f} vis_p99={d.vis_p99:.3f} ({d.frames_compared} frames)")

    # --- PRIMARY GATE: final 3D vs golden ma_3d ---
    pred_joints: Float32[ndarray, "f j 3"] = np.stack([fits_by_frame[f].joints for f in golden_frames])
    pred_vertices: Float32[ndarray, "f v 3"] = np.stack([fits_by_frame[f].vertices for f in golden_frames])
    cmp = fit3d_comparison(
        config.golden_dir / "ma_3d/baseline/indoors/crossing_arms/verts_joints_body_id-00.npz", pred_joints, pred_vertices
    )
    mpjpe_ok: bool = cmp.mpjpe_mm <= config.mpjpe_tolerance_mm
    pve_ok: bool = cmp.pve_mm <= config.pve_tolerance_mm
    print(f"\nGOLDEN GATE (frames {GOLDEN_FRAME_START}:{GOLDEN_FRAME_END}, {cmp.frames_compared} frames):")
    print(f"  MPJPE {cmp.mpjpe_mm:6.1f} mm  (tolerance {config.mpjpe_tolerance_mm} mm)  {'PASS' if mpjpe_ok else 'FAIL'}")
    print(f"  PVE   {cmp.pve_mm:6.1f} mm  (tolerance {config.pve_tolerance_mm} mm)  {'PASS' if pve_ok else 'FAIL'}")
    overall: bool = mpjpe_ok and pve_ok
    print(f"\nRESULT: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(ValidateConfig)))
