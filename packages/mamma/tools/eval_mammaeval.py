"""MammaEval accuracy in the MAMMA paper protocol (arXiv 2506.13040, Table 4), plus the realtime stats of the same run.

Runs the SAME streaming pipeline the benchmark builds (decode -> track -> landmarks -> fit -> Rerun logging) on a
prepared MammaEval sequence (``~/data/mamma-eval/<split>/<seq>/{meta,videos,gt}``), collects every frame's emitted
SMPL-X, and scores it against the MoSh++ GT: MPJPE over the 55 SMPL-X joints and PVE over all 10475 vertices, mm,
world frame, NO alignment (PA-MPJPE alongside for context). Paper reference (16 cams): MPJPE 12.96 mm / PVE 17.18 mm.

Output capture: the fitting stage's ``step`` is wrapped so each tick's fits are recorded on the fit worker thread;
unlike a ``ResultCollector`` this keeps the fit overlapped with the next tick, so the reported ticks/s is the
benchmark's operating point. The fitter's bootstrap-window head (``drain_head``) and fixed-lag tail (``drain``) are
flushed after the loop, so every frame of ``--frames`` is emitted. ``emit_stride`` is forced to 1 (emission is a pure
read-out of the fit; with stride 2 every other frame would repeat the previous fit). The JSON is written after the loop.

Deviations from the paper: lens distortion is ignored (prepared videos/calibrations are pinhole), Singles run at half
resolution with portrait cameras rotated to landscape, and the camera count is whatever ``--cameras`` selects.
"""

from __future__ import annotations

import dataclasses
import gc
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from scipy.optimize import linear_sum_assignment
from serde import serde
from serde.json import to_json
from simplecv.rerun_log_utils import RerunTyroConfig

from mamma.datasets.mamma_npz import load_mamma_sequence
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import PipelineStats, ResultCollector, StreamingPipeline, build_streaming_pipeline
from mamma.engine.presets import PresetName, get_preset
from mamma.engine.types import CameraTracks
from mamma.eval.mosh_gt import NUM_KINEMATIC_JOINTS, MoshSequence, PosedBody, load_mosh_gt, pose_mosh_person
from mamma.eval.paper_metrics import AggregateMetrics, SequenceMetrics, aggregate, sequence_metrics
from mamma.fitting.stage import FittingStage, TickFitOutput
from mamma.fitting.window_fitter import FitResult, FitterConfig
from mamma.landmarks.estimator import CameraLandmarks
from mamma.tracking.tracker import TrackerConfig

PAPER_MPJPE_MM: float = 12.96
"""MAMMA, MammaEval-S, 16 cameras (Table 4)."""
PAPER_PVE_MM: float = 17.18
"""MAMMA, MammaEval-S, 16 cameras (Table 4)."""


@dataclass
class EvalConfig:
    seq_dir: Path
    """Prepared sequence dir: meta/{IOI_NN,global}.npz, videos/IOI_NN.mp4, gt/global.npz."""
    rr_config: RerunTyroConfig = field(default_factory=lambda: RerunTyroConfig(headless=True))
    """Rerun behavior; default headless with no sink (in-memory), so the full logging path still runs, as in the
    benchmark. Pass --rr-config.no-headless to spawn a viewer or --rr-config.save to keep an RRD."""
    cameras: list[str] | None = None
    """Explicit camera names (e.g. IOI_01 IOI_05); default all cameras in meta/."""
    num_cameras: int | None = None
    """Keep only the first N of the selected cameras (name order)."""
    frames: tuple[int, int] | None = None
    """Video frame range START END (end exclusive); default the whole sequence."""
    resize_hw: tuple[int, int] = (752, 1024)
    """Engine resolution (h, w). Default ~half of the prepared 1504x2056 landscape videos: 752 = 1504/2 exactly, width
    1024 instead of 1028 because the Rerun video encoder needs a width that is a multiple of 16 (0.4% anisotropy,
    absorbed by the per-axis K scaling). Always applied, also with --preset (the presets' 720x1280 would squash the
    1.37:1 MammaEval frames)."""
    preset: PresetName | None = None
    """When set, overrides tracker/fitter/hires_crops with that preset's operating point (resize_hw stays ours)."""
    tracker: TrackerConfig = field(default_factory=lambda: TrackerConfig(expected_subjects=None))
    """Tracker settings. expected_subjects defaults to None (track everyone the bootstrap finds): MammaEval-S has
    non-mocapped bystanders (e.g. the ball thrower in CatchBall), and with 1 the bootstrap can lock onto one of them;
    the GT person is then matched to a track by pelvis distance."""
    fitter: FitterConfig = field(default_factory=FitterConfig)
    """Fitter settings (emit_stride is forced to 1)."""
    hires_crops: bool = True
    """Decode native res; sample MammaNet crops from it."""
    mammanet_weights: Path = Path("data/weights/ma_2d/mamma_mask_full_cvpr.safetensors")
    """Converted MammaNet weights."""
    trt_engine: Path | None = None
    """Optional MammaNet TensorRT engine plan."""
    body_models_dir: Path = Path("data/body_models")
    """Body-models root with smplx_lockedhead/ and smplx_v1_1/ (GT models)."""
    chunk_size: int = 32
    """Frames decoded per camera per chunk."""
    observe_tracked_only: bool = False
    """With --tracker.track-stride N, run landmarks + fit only on tracked ticks (body observed at fps/N)."""
    hold_missing: bool = False
    """Score a frame without its own fit with the latest earlier fit of that track (what the viewer shows when the
    body is observed below the frame rate, e.g. with --observe-tracked-only); the count is reported as frames_held."""
    sync_fit: bool = False
    """Await each tick's fit inline (as validate_golden's collector does) instead of overlapping it with the next
    tick. Slower, but avoids a race in the overlapped path: the fit worker's CUDA-graph capture (bootstrap, first
    graphed tick) fails when the tracker's YOLO re-detect calls torch.cuda.synchronize on the main thread at the same
    moment ("operation not permitted when stream is capturing")."""
    warmup_frames: int = 0
    """Frames run once on a throwaway pipeline first (absorbs load/autotune, like the benchmark's 33); 0 = cold run,
    so ticks/s includes bootstrap + autotune."""
    out: Path | None = None
    """JSON summary path; default /tmp/mamma-rt/eval/<seq>_<N>cams_<start>-<end>.json."""
    dump_npz: Path | None = None
    """Optional NPZ of per-frame predicted and GT joints/vertices of every matched person (diagnostics)."""
    device: str = "cuda"
    """Compute device."""


@serde
@dataclass(frozen=True, slots=True)
class PersonReport:
    """Scores of one GT person."""

    subject: str
    """GT subject id."""
    gender: str
    """GT gender (selects the GT SMPL-X model)."""
    obj_id: int | None
    """Pipeline track id matched to this person; None when no track overlaps."""
    frames_scored: int
    """Frames in range with an emitted fit."""
    frames_missing: list[int]
    """Frames in range without an emitted fit (not scored)."""
    frames_low_visibility: int
    """Scored frames where is_body_in_img holds in fewer than 2 of the used cameras (still scored)."""
    mpjpe_mm: float | None
    """Mean MPJPE over scored frames (55 joints, no alignment)."""
    pve_mm: float | None
    """Mean PVE over scored frames (10475 vertices, no alignment)."""
    pa_mpjpe_mm: float | None
    """Mean PA-MPJPE over scored frames (context)."""
    steady_mpjpe_mm: float | None
    """Mean MPJPE over scored frames from start + window_size on (excludes the bootstrap window; not the paper's
    number, which counts every frame)."""
    steady_pve_mm: float | None
    """Mean PVE over the same steady-state frames."""
    mpjpe_p95_mm: float | None
    """95th-percentile per-frame MPJPE."""
    pve_p95_mm: float | None
    """95th-percentile per-frame PVE."""
    per_frame_index: list[int]
    """Video frame of each per-frame entry."""
    per_frame_mpjpe_mm: list[float]
    """Per-frame MPJPE."""
    per_frame_pve_mm: list[float]
    """Per-frame PVE."""
    per_frame_pa_mpjpe_mm: list[float]
    """Per-frame PA-MPJPE."""


@serde
@dataclass(frozen=True, slots=True)
class RealtimeReport:
    """Throughput of the scored run."""

    ticks: int
    """Synchronized frame sets processed."""
    elapsed_s: float
    """Loop wall time (excludes model construction)."""
    ticks_per_s: float
    """Sustained throughput."""
    realtime_fraction: float
    """ticks_per_s / fps (1.0 = realtime)."""
    cam_fps: float
    """ticks_per_s x cameras."""
    decode_backend: str
    """Decoder behind the frames."""
    warmup_frames: int
    """Warmup frames run before (0 = cold; bootstrap + autotune inside elapsed_s)."""
    sync_fit: bool
    """Fit awaited inline (no fit/track overlap): ticks/s is then below the benchmark's operating point."""
    stage_ms_per_call: dict[str, float]
    """Per-stage mean milliseconds per call."""


@serde
@dataclass(frozen=True, slots=True)
class EvalReport:
    """JSON summary of one eval run."""

    sequence: str
    """Sequence name."""
    seq_dir: str
    """Prepared sequence directory."""
    cameras: list[str]
    """Cameras used, in pipeline order."""
    frame_start: int
    """First video frame run."""
    frame_end: int
    """One past the last video frame run."""
    resize_hw: list[int]
    """Engine resolution (h, w)."""
    preset: str | None
    """Preset name or None (explicit configs)."""
    hires_crops: bool
    """Native-res landmark crops."""
    trt_engine: str | None
    """TensorRT plan used for MammaNet."""
    emit_lag: int
    """Fitter fixed-lag frames (output latency emit_lag/fps)."""
    protocol: str
    """What the numbers mean."""
    people: list[PersonReport]
    """Per-person scores."""
    pooled_mpjpe_mm: float | None
    """Mean over all scored frames of all people."""
    pooled_pve_mm: float | None
    """Mean over all scored frames of all people."""
    pooled_pa_mpjpe_mm: float | None
    """Mean over all scored frames of all people."""
    seqmean_mpjpe_mm: float | None
    """Mean of per-person-sequence means."""
    seqmean_pve_mm: float | None
    """Mean of per-person-sequence means."""
    realtime: RealtimeReport
    """Throughput of the same run."""
    paper_mpjpe_mm: float = PAPER_MPJPE_MM
    """Paper reference, 16 cameras."""
    paper_pve_mm: float = PAPER_PVE_MM
    """Paper reference, 16 cameras."""


def select_cameras(seq_dir: Path, cameras: list[str] | None, num_cameras: int | None) -> list[str]:
    """Camera names to run: explicit list or all in ``meta/``, then the first ``num_cameras``.

    Args:
        seq_dir: Prepared sequence dir.
        cameras: Explicit names, or None for all.
        num_cameras: Keep the first N, or None for all.
    """
    names: list[str] = cameras if cameras is not None else sorted(p.stem for p in (seq_dir / "meta").glob("*.npz") if p.stem != "global")
    if num_cameras is not None:
        names = names[:num_cameras]
    if len(names) < 2:
        raise ValueError(f"need >= 2 cameras, got {names}")
    return names


def load_body_in_img(seq_dir: Path, cameras: list[str], frames_len: int) -> Bool[ndarray, "c f p"] | None:
    """Per-camera ``is_body_in_img`` from ``meta/<cam>.npz`` or ``gt/<cam>.npz``; None if any camera lacks it.

    Rotating a portrait camera to landscape does not change which frames see the body, so the flag is copied as-is.
    """
    flags: list[Bool[ndarray, "f p"]] = []
    for cam in cameras:
        found: Bool[ndarray, "f p"] | None = None
        for sub in ("meta", "gt"):
            path: Path = seq_dir / sub / f"{cam}.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as npz:
                    if "is_body_in_img" in npz.files:
                        found = npz["is_body_in_img"].astype(bool)
                        break
        if found is None or found.shape[0] != frames_len:
            return None
        flags.append(found)
    return np.stack(flags)


def attach_fit_recorder(fitting: FittingStage) -> list[TickFitOutput]:
    """Record every ``fitting.step`` output in call order without forcing a sync.

    The pipeline calls ``self.fitting.step`` on its single fit worker, so the shadowing instance attribute sees
    every tick, in order, while the fit stays overlapped with the next tick's tracking (a ``ResultCollector``
    would await each fit inline and change the measured throughput).

    Returns:
        The list the outputs are appended to.
    """
    outputs: list[TickFitOutput] = []
    inner = fitting.step

    def step(frame_idx: int, landmarks: list[CameraLandmarks]) -> TickFitOutput:
        out: TickFitOutput = inner(frame_idx, landmarks)
        outputs.append(out)
        return out

    fitting.step = step  # type: ignore[method-assign]
    return outputs


@dataclass(slots=True)
class PredictedTrack:
    """One pipeline track's emitted SMPL-X, by video frame."""

    joints: dict[int, Float32[ndarray, "55 3"]] = field(default_factory=dict)
    """Kinematic joints per frame, world metres."""
    vertices: dict[int, Float32[ndarray, "10475 3"]] = field(default_factory=dict)
    """Vertices per frame, world metres."""
    duplicates: int = 0
    """Frames emitted more than once (first emission kept: the causal output)."""


def hold_previous_fit(track: PredictedTrack, frames: range) -> int:
    """Fill frames without a fit with the latest earlier fit (sample-and-hold); returns how many were filled."""
    filled: int = 0
    last: int | None = None
    for frame in frames:
        if frame in track.joints:
            last = frame
        elif last is not None:
            track.joints[frame] = track.joints[last]
            track.vertices[frame] = track.vertices[last]
            filled += 1
    return filled


def collect_tracks(outputs: list[TickFitOutput]) -> dict[int, PredictedTrack]:
    """Group emitted fits by track id and video frame (``FitResult.frame_idx``, so emit lag is already undone)."""
    tracks: dict[int, PredictedTrack] = {}
    for out in outputs:
        for obj_id, fit in out.fits.items():
            track: PredictedTrack = tracks.setdefault(obj_id, PredictedTrack())
            result: FitResult = fit
            if result.frame_idx in track.joints:
                track.duplicates += 1
                continue
            track.joints[result.frame_idx] = result.joints[:NUM_KINEMATIC_JOINTS].astype(np.float32)
            track.vertices[result.frame_idx] = result.vertices.astype(np.float32)
    return tracks


def match_people(gt_bodies: list[PosedBody], tracks: dict[int, PredictedTrack]) -> list[int | None]:
    """Assign each GT person the track with the smallest mean pelvis distance over shared frames (Hungarian).

    Args:
        gt_bodies: Posed GT per person over the run's frames.
        tracks: Emitted fits per track id.

    Returns:
        Track id per GT person, None when unmatched.
    """
    obj_ids: list[int] = sorted(tracks)
    if not obj_ids:
        return [None] * len(gt_bodies)
    cost: Float64[ndarray, "p t"] = np.full((len(gt_bodies), len(obj_ids)), 1e6)
    for i, body in enumerate(gt_bodies):
        for j, obj_id in enumerate(obj_ids):
            rows: list[int] = [k for k, f in enumerate(body.frames.tolist()) if f in tracks[obj_id].joints]
            if rows:
                pred_pelvis: Float32[ndarray, "n 3"] = np.stack([tracks[obj_id].joints[int(body.frames[k])][0] for k in rows])
                cost[i, j] = float(np.linalg.norm(pred_pelvis - body.joints[rows, 0], axis=-1).mean())
    assignment: tuple[Int64[ndarray, "m"], Int64[ndarray, "m"]] = linear_sum_assignment(cost)
    matched: list[int | None] = [None] * len(gt_bodies)
    for i, j in zip(assignment[0].tolist(), assignment[1].tolist(), strict=True):
        if cost[i, j] < 1e6:
            matched[i] = obj_ids[j]
    return matched


def _p95(values: Float64[ndarray, "f"]) -> float:
    return float(np.percentile(values, 95))


class _InlineFitCollector(ResultCollector):
    """Keeps nothing; its presence alone makes the pipeline await every fit inline (``--sync-fit``)."""

    def collect(
        self,
        frame_idx: int,
        tracks: list[CameraTracks] | None,
        landmarks: list[CameraLandmarks] | None,
        fit_output: TickFitOutput | None = None,
    ) -> None:
        return None


def build_pipeline(config: EvalConfig, sequence: MultiViewSequence) -> StreamingPipeline:
    """The benchmark's pipeline (``tools/benchmark.py::build_pipeline``) with our resize_hw and emit_stride=1."""
    tracker_cfg: TrackerConfig = config.tracker
    fitter_cfg: FitterConfig = config.fitter
    hires_crops: bool = config.hires_crops
    if config.preset is not None:
        preset = get_preset(config.preset, expected_subjects=config.tracker.expected_subjects)
        tracker_cfg, fitter_cfg, hires_crops = preset.tracker, preset.fitter, preset.hires_crops
    return build_streaming_pipeline(
        sequence,
        resize_hw=config.resize_hw,
        device=config.device,
        tracker_config=dataclasses.replace(tracker_cfg),
        fitter_config=dataclasses.replace(fitter_cfg, emit_stride=1),
        mammanet_weights=config.mammanet_weights,
        trt_engine=config.trt_engine,
        hires_crops=hires_crops,
        collector=_InlineFitCollector() if config.sync_fit else None,
        observe_tracked_only=config.observe_tracked_only,
    )


def run_warmup(config: EvalConfig, sequence: MultiViewSequence, start: int) -> None:
    """Benchmark-style warmup on a throwaway, sink-less recording; frees the pipeline afterwards."""
    import rerun as rr
    import torch

    main_recording = rr.get_global_data_recording()
    rr.set_global_data_recording(rr.RecordingStream(application_id="mamma-eval-warmup"))
    warmup: StreamingPipeline = build_pipeline(config, sequence)
    try:
        warmup.run(chunk_size=config.chunk_size, max_frames=config.warmup_frames, start_frame=start)
    finally:
        warmup.close()
        if main_recording is not None:
            rr.set_global_data_recording(main_recording)
    del warmup
    gc.collect()
    torch.cuda.empty_cache()


def main(config: EvalConfig) -> int:
    if config.resize_hw[1] % 16 != 0:
        raise ValueError(f"resize_hw width {config.resize_hw[1]} must be a multiple of 16 (StreamLogger video encoder plane size)")
    config = dataclasses.replace(config, seq_dir=config.seq_dir.expanduser())
    cameras: list[str] = select_cameras(config.seq_dir, config.cameras, config.num_cameras)
    sequence: MultiViewSequence = load_mamma_sequence(config.seq_dir, camera_names=cameras)
    gt: MoshSequence = load_mosh_gt(config.seq_dir / "gt" / "global.npz")
    start: int = 0 if config.frames is None else config.frames[0]
    end: int = gt.frames_len if config.frames is None else min(config.frames[1], gt.frames_len)
    if not 0 <= start < end:
        raise ValueError(f"bad frame range {start}..{end} (GT has {gt.frames_len} frames)")
    sizes: set[tuple[int, int]] = {(cam.width, cam.height) for cam in sequence.cameras}
    print(f"sequence {gt.name}: GT {gt.frames_len} frames, {len(gt.people)} people ({gt.method}); cameras {cameras} {sizes}")
    print(f"frames [{start}, {end}), resize_hw {config.resize_hw}, preset {config.preset}, trt {config.trt_engine}")

    if config.warmup_frames > 0:
        run_warmup(config, sequence, start)
        print(f"warmup done ({config.warmup_frames} frames)")

    pipeline: StreamingPipeline = build_pipeline(config, sequence)
    assert pipeline.fitting is not None, "eval needs the full pipeline (tracker + landmarks + fitter)"
    outputs: list[TickFitOutput] = attach_fit_recorder(pipeline.fitting)
    try:
        stats: PipelineStats = pipeline.run(chunk_size=config.chunk_size, max_frames=end - start, start_frame=start)
    finally:
        pipeline.close()
    head: list[TickFitOutput] = pipeline.fitting.drain_head()
    tail: list[TickFitOutput] = pipeline.fitting.drain()
    emit_lag: int = pipeline.fitting.config.emit_lag
    window_size: int = pipeline.fitting.config.window_size
    tracks: dict[int, PredictedTrack] = collect_tracks(head + outputs + tail)
    if config.hold_missing:
        held: int = sum(hold_previous_fit(track, range(start, end)) for track in tracks.values())
        print(f"held {held} frames without their own fit (latest earlier fit)")
    fps: float = sequence.fps
    realtime = RealtimeReport(
        ticks=stats.ticks,
        elapsed_s=stats.elapsed_s,
        ticks_per_s=stats.ticks_per_s,
        realtime_fraction=stats.ticks_per_s / fps,
        cam_fps=stats.ticks_per_s * len(cameras),
        decode_backend=stats.decode_backend,
        warmup_frames=config.warmup_frames,
        sync_fit=config.sync_fit,
        stage_ms_per_call={k: 1000.0 * v / max(stats.profiler.counts[k], 1) for k, v in stats.profiler.totals.items()},
    )
    del pipeline
    gc.collect()

    # --- GT + scoring (after the streaming loop) ---
    run_frames: Int64[ndarray, "f"] = np.arange(start, end, dtype=np.int64)
    gt_bodies: list[PosedBody] = [pose_mosh_person(p, run_frames, config.body_models_dir) for p in gt.people]
    matched: list[int | None] = match_people(gt_bodies, tracks)
    body_in_img: Bool[ndarray, "c f p"] | None = load_body_in_img(config.seq_dir, cameras, gt.frames_len)

    reports: list[PersonReport] = []
    scored: list[SequenceMetrics] = []
    for p_idx, (person, body, obj_id) in enumerate(zip(gt.people, gt_bodies, matched, strict=True)):
        track: PredictedTrack | None = tracks.get(obj_id) if obj_id is not None else None
        have: list[int] = [] if track is None else [k for k, f in enumerate(run_frames.tolist()) if f in track.joints]
        have_set: set[int] = set(have)
        missing: list[int] = [int(f) for k, f in enumerate(run_frames.tolist()) if k not in have_set]
        if track is None or not have:
            reports.append(
                PersonReport(person.subject, person.gender, obj_id, 0, missing, 0, None, None, None, None, None, None, None, [], [], [], [])
            )
            continue
        frames: Int64[ndarray, "n"] = run_frames[have]
        metrics: SequenceMetrics = sequence_metrics(
            f"{gt.name}/{person.subject}",
            frames,
            np.stack([track.joints[int(f)] for f in frames]),
            np.stack([track.vertices[int(f)] for f in frames]),
            body.joints[have],
            body.vertices[have],
        )
        scored.append(metrics)
        steady: Bool[ndarray, "n"] = frames >= start + window_size
        low_vis: int = 0 if body_in_img is None else int((body_in_img[:, frames, p_idx].sum(axis=0) < 2).sum())
        reports.append(
            PersonReport(
                subject=person.subject,
                gender=person.gender,
                obj_id=obj_id,
                frames_scored=len(frames),
                frames_missing=missing,
                frames_low_visibility=low_vis,
                mpjpe_mm=metrics.mean_mpjpe_mm,
                pve_mm=metrics.mean_pve_mm,
                pa_mpjpe_mm=metrics.mean_pa_mpjpe_mm,
                steady_mpjpe_mm=float(metrics.mpjpe_mm[steady].mean()) if steady.any() else None,
                steady_pve_mm=float(metrics.pve_mm[steady].mean()) if steady.any() else None,
                mpjpe_p95_mm=_p95(metrics.mpjpe_mm),
                pve_p95_mm=_p95(metrics.pve_mm),
                per_frame_index=frames.tolist(),
                per_frame_mpjpe_mm=metrics.mpjpe_mm.tolist(),
                per_frame_pve_mm=metrics.pve_mm.tolist(),
                per_frame_pa_mpjpe_mm=metrics.pa_mpjpe_mm.tolist(),
            )
        )
    agg: AggregateMetrics | None = aggregate(scored) if scored else None
    if config.dump_npz is not None:
        arrays: dict[str, ndarray] = {}
        for p_idx, (body, obj_id) in enumerate(zip(gt_bodies, matched, strict=True)):
            if obj_id is None:
                continue
            frames_p: Int64[ndarray, "n"] = np.array([f for f in run_frames.tolist() if f in tracks[obj_id].joints], dtype=np.int64)
            arrays[f"p{p_idx}_frames"] = frames_p
            arrays[f"p{p_idx}_pred_joints"] = np.stack([tracks[obj_id].joints[int(f)] for f in frames_p])
            arrays[f"p{p_idx}_pred_vertices"] = np.stack([tracks[obj_id].vertices[int(f)] for f in frames_p])
            arrays[f"p{p_idx}_gt_joints"] = body.joints[frames_p - start]
            arrays[f"p{p_idx}_gt_vertices"] = body.vertices[frames_p - start]
        config.dump_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(config.dump_npz, **arrays)

    report = EvalReport(
        sequence=gt.name,
        seq_dir=str(config.seq_dir),
        cameras=cameras,
        frame_start=start,
        frame_end=end,
        resize_hw=list(config.resize_hw),
        preset=config.preset,
        hires_crops=config.hires_crops if config.preset is None else get_preset(config.preset).hires_crops,
        trt_engine=None if config.trt_engine is None else str(config.trt_engine),
        emit_lag=emit_lag,
        protocol="MPJPE: 55 SMPL-X joints; PVE: 10475 vertices; mm; world frame; no alignment; GT = MoSh++ (release SMPL-X "
        "variant per gender); frames without an emitted fit are listed, not scored; low-visibility frames are scored.",
        people=reports,
        pooled_mpjpe_mm=None if agg is None else agg.pooled_mpjpe_mm,
        pooled_pve_mm=None if agg is None else agg.pooled_pve_mm,
        pooled_pa_mpjpe_mm=None if agg is None else agg.pooled_pa_mpjpe_mm,
        seqmean_mpjpe_mm=None if agg is None else agg.seqmean_mpjpe_mm,
        seqmean_pve_mm=None if agg is None else agg.seqmean_pve_mm,
        realtime=realtime,
    )

    print(f"\nMAMMAEVAL ({gt.name}, {len(cameras)} cams, frames {start}:{end}, no alignment):")
    print(f"  {'subject':<8}{'track':>6}{'frames':>8}{'missing':>8}{'MPJPE':>9}{'PVE':>9}{'PA-MPJPE':>10}{'MPJPE95':>9}{'PVE95':>9}")
    for r in reports:
        if r.mpjpe_mm is None:
            print(f"  {r.subject:<8}{str(r.obj_id):>6}{0:>8}{len(r.frames_missing):>8}   (no fit emitted)")
            continue
        print(
            f"  {r.subject:<8}{r.obj_id:>6}{r.frames_scored:>8}{len(r.frames_missing):>8}{r.mpjpe_mm:>9.2f}{r.pve_mm:>9.2f}"
            f"{r.pa_mpjpe_mm:>10.2f}{r.mpjpe_p95_mm:>9.2f}{r.pve_p95_mm:>9.2f}"
        )
    if agg is not None:
        print(f"  pooled   MPJPE {agg.pooled_mpjpe_mm:.2f} mm  PVE {agg.pooled_pve_mm:.2f} mm  PA-MPJPE {agg.pooled_pa_mpjpe_mm:.2f} mm")
        print(f"  seq-mean MPJPE {agg.seqmean_mpjpe_mm:.2f} mm  PVE {agg.seqmean_pve_mm:.2f} mm")
    for r in reports:
        if r.steady_mpjpe_mm is not None:
            print(f"  {r.subject}: steady state (frames >= {start + window_size}): MPJPE {r.steady_mpjpe_mm:.2f} mm  PVE {r.steady_pve_mm:.2f} mm")
    print(f"  paper (16 cams, MammaEval-S): MPJPE {PAPER_MPJPE_MM} mm  PVE {PAPER_PVE_MM} mm")
    print(
        f"\nREALTIME: {stats.ticks} ticks in {stats.elapsed_s:.2f} s = {stats.ticks_per_s:.2f} ticks/s ({realtime.realtime_fraction:.0%} of {fps:.0f} fps, {realtime.cam_fps:.0f} cam-fps); decode {stats.decode_backend}"
    )
    print(stats.profiler.report())

    out: Path = config.out or Path("/tmp/mamma-rt/eval") / f"{gt.name}_{len(cameras)}cams_{start}-{end}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_json(report))
    print(f"\nwrote {out}")
    return 0 if agg is not None else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(EvalConfig)))
