"""Dump every per-tick artifact of one streaming run to NPZs for offline comparison.

The quality/fast preset gates compare our pipeline against the original DAG's
saved artifacts (ma_2d landmarks, ma_3d triangulated points + SMPL-X params,
ma_masks PNGs). The streaming engine never writes to disk, so this tool runs
the pipeline once with a collector that eagerly moves each tick's outputs to
CPU and saves them at the end — one canonical run that analysis scripts can
slice repeatedly without re-paying GPU time.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import ResultCollector, StreamingPipeline, build_streaming_pipeline
from mamma.engine.presets import PresetName, get_preset
from mamma.engine.types import CameraTracks
from mamma.fitting.stage import TickFitOutput
from mamma.fitting.window_fitter import FitResult, FitterConfig
from mamma.landmarks.estimator import CameraLandmarks
from mamma.tracking.tracker import TrackerConfig


class _DumpCollector(ResultCollector):
    """Moves each tick's outputs to CPU numpy immediately (no GPU pinning)."""

    def __init__(self, n_cams: int, obj_id: int) -> None:
        super().__init__()
        self.n_cams: int = n_cams
        self.obj_id: int = obj_id
        self.mask_frames: list[int] = []
        self.masks_packed: list[list[ndarray]] = []  # [tick][cam] packbits of (h w) bool
        self.mask_hw: tuple[int, int] | None = None
        self.lm_frames: list[int] = []
        self.lm_joints2d: list[ndarray] = []  # (c, 512, 3) per tick
        self.lm_vis: list[ndarray] = []  # (c, 512) per tick
        self.tri_frames: list[int] = []
        self.tri_points: list[ndarray] = []  # (512, 3) per tick
        self.tri_valid: list[ndarray] = []  # (512,) per tick
        self.fit_by_frame: dict[int, FitResult] = {}

    def collect(
        self,
        frame_idx: int,
        tracks: list[CameraTracks] | None,
        landmarks: list[CameraLandmarks] | None,
        fit_output: TickFitOutput | None = None,
    ) -> None:
        if tracks is not None and all(self.obj_id in cam for cam in tracks):
            per_cam: list[ndarray] = []
            for cam in tracks:
                mask: ndarray = cam[self.obj_id].mask.cpu().numpy()
                self.mask_hw = (mask.shape[0], mask.shape[1])
                per_cam.append(np.packbits(mask))
            self.mask_frames.append(frame_idx)
            self.masks_packed.append(per_cam)
        if landmarks is not None and all(self.obj_id in cam for cam in landmarks):
            self.lm_frames.append(frame_idx)
            self.lm_joints2d.append(np.stack([cam[self.obj_id].joints2d.cpu().numpy() for cam in landmarks]))
            self.lm_vis.append(np.stack([cam[self.obj_id].visibility.cpu().numpy() for cam in landmarks]))
        if frame_idx >= 0 and frame_idx % 100 == 0:
            print(f"tick {frame_idx}", flush=True)
        if fit_output is not None:
            if self.obj_id in fit_output.triangulated:
                cloud, valid = fit_output.triangulated[self.obj_id]
                self.tri_frames.append(frame_idx)
                self.tri_points.append(cloud.cpu().numpy())
                self.tri_valid.append(valid.cpu().numpy())
            if self.obj_id in fit_output.fits:
                fit: FitResult = fit_output.fits[self.obj_id]
                self.fit_by_frame[fit.frame_idx] = fit


@dataclass
class DumpConfig:
    rr_config: RerunTyroConfig
    """Rerun behavior (headless by default; the dump itself needs no viewer)."""
    out_dir: Path = Path("/tmp/qfdig/current_run")
    """Where the artifact NPZs land."""
    preset: PresetName | None = None
    """When set ('quality'|'fast'), overrides tracker/fitter/resize/hires_crops
    with that preset's operating point (tracker/fitter flags below are ignored)."""
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(expected_subjects=1))
    """Tracker settings; running_jumping is single-subject."""
    fitter: FitterConfig = field(default_factory=lambda: FitterConfig(emit_stride=1))
    """Window fitter settings (emit every frame for full-coverage comparison)."""
    data_dir: Path = Path("data/inputs/outdoors/running_jumping")
    """Capture inputs (MAMMA NPZ layout)."""
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet weights."""
    trt_engine: Path | None = None
    """Optional MammaNet TensorRT engine plan."""
    resize_hw: tuple[int, int] = (720, 1280)
    """Engine resolution (masks/landmarks are dumped at this resolution)."""
    obj_id: int = 0
    """Person id to dump (golden artifacts are body_id 00)."""
    proxy_dir: Path | None = None
    """720p proxy video dir (tools/make_proxies.py). Decodes proxies instead of
    4K source (forces hires_crops=False); the runtime lever for the presets."""
    seg_stride: int = 1
    """Log full-res masks every Nth tick in the saved RRD (1 = per-frame, so the
    mask tracks the person with no lag — for showcase recordings)."""
    mp_decode: bool = True
    """Multiprocess decode workers (disable at high engine resolutions)."""
    hires_crops: bool = True
    """Decode native res; sample landmark crops from it."""
    chunk_size: int = 32
    """Frames decoded per camera per chunk."""
    device: str = "cuda"
    """Compute device."""


def main(config: DumpConfig) -> int:
    sequence: MultiViewSequence = load_mamma_sequence(config.data_dir)
    tracker_cfg: TrackerConfig = config.tracker
    fitter_cfg: FitterConfig = config.fitter
    resize_hw: tuple[int, int] = config.resize_hw
    hires_crops: bool = config.hires_crops
    if config.preset is not None:
        preset = get_preset(config.preset)
        tracker_cfg, fitter_cfg, resize_hw, hires_crops = preset.tracker, preset.fitter, preset.resize_hw, preset.hires_crops
        print(f"preset={config.preset}: tracker={tracker_cfg.sam2_config} redetect={tracker_cfg.redetect_interval} tick_iters={fitter_cfg.tick_iters}")
    collector = _DumpCollector(n_cams=len(sequence.camera_names), obj_id=config.obj_id)
    pipeline: StreamingPipeline = build_streaming_pipeline(
        sequence,
        resize_hw=resize_hw,
        device=config.device,
        tracker_config=tracker_cfg,
        fitter_config=fitter_cfg,
        mammanet_weights=config.mammanet_weights,
        trt_engine=config.trt_engine,
        collector=collector,
        use_mp_decode=config.mp_decode,
        hires_crops=hires_crops,
        proxy_dir=config.proxy_dir,
        seg_stride=config.seg_stride,
    )
    stats = pipeline.run(chunk_size=config.chunk_size, timing_doc=True)
    if pipeline.fitting is not None:
        faces = pipeline.fitting.faces
        # Head backfill: the bootstrap solves the first window but push() only
        # emits from window[-1-emit_lag] onward, so the first emit_lag-ish
        # frames have a cloud but no mesh. Log + collect them so the mesh
        # covers the clip from frame 0 (matches the original DAG, which fits
        # every frame). Mirror of the tail drain below.
        for head in pipeline.fitting.drain_head():
            collector.collect(-1, None, None, head)
            frame_head: int = next(iter(head.fits.values())).frame_idx
            pipeline.logger.log_tick_fit(frame_head, head.fits, {}, faces)
        for tail in pipeline.fitting.drain():
            collector.collect(-1, None, None, tail)
            # Also log the drained fixed-lag tail meshes to the RRD so the mesh
            # covers the final emit_lag frames (otherwise the cloud advances to
            # the clip end while the mesh stays frozen on the last emitted frame).
            if tail.fits:
                frame: int = next(iter(tail.fits.values())).frame_idx
                pipeline.logger.log_tick_fit(frame, tail.fits, {}, faces)
    # Per-stage timing TextDocument (matches the original relog's timing panel).
    pipeline.logger.log_timing_summary(
        dict(stats.profiler.totals), dict(stats.profiler.counts), stats.elapsed_s, stats.ticks, label=config.preset or ""
    )
    pipeline.logger.flush()
    pipeline.close()
    print(f"pipeline: {stats.ticks} ticks in {stats.elapsed_s:.1f}s ({stats.ticks_per_s:.1f} ticks/s)")
    print(stats.profiler.report())

    config.out_dir.mkdir(parents=True, exist_ok=True)
    # Each artifact list is only appended when the subject is visible enough that
    # tick (masks: all cameras; landmarks: all cameras; tri/fits: >=2 cameras +
    # the fitter bootstrapped). On a clip too short to bootstrap, or one where the
    # subject is never co-visible, these stay empty and the np.stack/fits[0] saves
    # below would raise an opaque ValueError/IndexError after a full GPU run.
    if not (collector.mask_hw and collector.lm_frames and collector.tri_frames and collector.fit_by_frame):
        print(
            f"incomplete capture — masks {len(collector.mask_frames)}f, landmarks {len(collector.lm_frames)}f, "
            f"tri {len(collector.tri_frames)}f, fits {len(collector.fit_by_frame)}f; subject not visible long enough to dump. Skipping save."
        )
        return 1
    assert collector.mask_hw is not None, "no masks collected"
    masks_arr: ndarray = np.stack([np.stack(per_cam) for per_cam in collector.masks_packed])
    np.savez_compressed(
        config.out_dir / "masks.npz",
        packed=masks_arr,  # (f, c, packed) uint8; unpack -> bool (h, w)
        frame_indices=np.array(collector.mask_frames, dtype=np.int64),
        mask_hw=np.array(collector.mask_hw, dtype=np.int64),
        camera_names=np.array(sequence.camera_names),
    )
    np.savez_compressed(
        config.out_dir / "landmarks.npz",
        joints2d=np.stack(collector.lm_joints2d),  # (f, c, 512, 3) [x_px, y_px, logvar] @ resize_hw
        visibility=np.stack(collector.lm_vis),  # (f, c, 512)
        frame_indices=np.array(collector.lm_frames, dtype=np.int64),
        camera_names=np.array(sequence.camera_names),
    )
    np.savez_compressed(
        config.out_dir / "triangulated.npz",
        points=np.stack(collector.tri_points),  # (f, 512, 3) world meters
        valid=np.stack(collector.tri_valid),  # (f, 512) bool
        frame_indices=np.array(collector.tri_frames, dtype=np.int64),
    )
    frames_sorted: list[int] = sorted(collector.fit_by_frame)
    fits: list[FitResult] = [collector.fit_by_frame[f] for f in frames_sorted]
    np.savez_compressed(
        config.out_dir / "smplx_fits.npz",
        frame_indices=np.array(frames_sorted, dtype=np.int64),
        pose=np.stack([fit.pose for fit in fits]),  # (f, 165) axis-angle
        betas=fits[0].betas,
        trans=np.stack([fit.trans for fit in fits]),  # (f, 3) meters
        joints=np.stack([fit.joints for fit in fits]),  # (f, j, 3) meters
        vertices=np.stack([fit.vertices for fit in fits]),  # (f, v, 3) meters
        rest_joints=fits[0].rest_joints,
    )
    timing: dict[str, object] = {
        "ticks": stats.ticks,
        "elapsed_s": stats.elapsed_s,
        "ticks_per_s": stats.ticks_per_s,
        "clip_seconds": sequence.frame_count / sequence.fps,
        "stage_totals_s": stats.profiler.totals,
        "stage_counts": stats.profiler.counts,
        "preset": config.preset,
        "resize_hw": list(resize_hw),
        "hires_crops": hires_crops,
        "sam2_config": tracker_cfg.sam2_config,
        "redetect_interval": tracker_cfg.redetect_interval,
        "tick_iters": fitter_cfg.tick_iters,
        "trt_engine": str(config.trt_engine) if config.trt_engine else None,
    }
    (config.out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    print(f"masks {masks_arr.shape} | landmarks {len(collector.lm_frames)}f | tri {len(collector.tri_frames)}f | fits {len(frames_sorted)}f")
    print(f"dumped to {config.out_dir}")
    return 0


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    sys.exit(main(tyro.cli(DumpConfig)))
