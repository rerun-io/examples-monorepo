"""The resident streaming loop: decode -> track -> (landmarks/fit, M3+) -> log.

No disk writes happen anywhere in this loop; outputs exist only as Rerun logs
(and in-memory results returned to the caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from jaxtyping import UInt8
from simplecv.video_io import TorchCodecMultiVideoReader

from mamma.calibration.npz_contract import CameraCalibration
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.profiler import StageProfiler
from mamma.engine.types import CameraTracks
from mamma.fitting.stage import FittingStage, TickFitOutput
from mamma.fitting.window_fitter import FitterConfig
from mamma.landmarks.estimator import CameraLandmarks, LandmarkEstimator
from mamma.tracking.tracker import MultiViewTracker, TrackerConfig
from mamma.viz.stream_logger import StreamLogger


class ResultCollector:
    """In-memory accumulator of per-tick outputs (for validation/benchmarks).

    Keeps everything in RAM — the streaming loop itself never writes to disk.
    """

    def __init__(self) -> None:
        self.frame_indices: list[int] = []
        self.tracks: list[list[CameraTracks] | None] = []
        self.landmarks: list[list[CameraLandmarks] | None] = []
        self.fits: list[TickFitOutput | None] = []

    def collect(
        self,
        frame_idx: int,
        tracks: list[CameraTracks] | None,
        landmarks: list[CameraLandmarks] | None,
        fit_output: TickFitOutput | None = None,
    ) -> None:
        self.frame_indices.append(frame_idx)
        self.tracks.append(tracks)
        self.landmarks.append(landmarks)
        self.fits.append(fit_output)


@dataclass(slots=True)
class PipelineStats:
    """Wall-clock summary of one streaming run."""

    ticks: int
    """Synchronized frame sets processed."""
    elapsed_s: float
    """Total wall time of the loop (excluding model construction)."""
    profiler: StageProfiler = field(repr=False)
    """Per-stage timing breakdown."""

    @property
    def ticks_per_s(self) -> float:
        """Sustained throughput."""
        return self.ticks / self.elapsed_s if self.elapsed_s > 0 else 0.0


class StreamingPipeline:
    """Owns the per-tick loop over a multiview sequence."""

    def __init__(
        self,
        sequence: MultiViewSequence,
        reader: TorchCodecMultiVideoReader,
        logger: StreamLogger,
        tracker: MultiViewTracker | None = None,
        landmarks: LandmarkEstimator | None = None,
        fitting: FittingStage | None = None,
        collector: ResultCollector | None = None,
        engine_hw: tuple[int, int] | None = None,
    ) -> None:
        self.sequence: MultiViewSequence = sequence
        self.reader: TorchCodecMultiVideoReader = reader
        self.logger: StreamLogger = logger
        self.tracker: MultiViewTracker | None = tracker
        self.landmarks: LandmarkEstimator | None = landmarks
        self.fitting: FittingStage | None = fitting
        self.collector: ResultCollector | None = collector
        # Dual-resolution mode: the reader decodes at NATIVE resolution and
        # each tick derives an engine_hw copy for tracking/logging, while
        # landmark crops sample the native frame (sharper MammaNet input —
        # the original fed it 4K crops; 720p crops cost ~20mm PVE on
        # self-occluded poses).
        self.engine_hw: tuple[int, int] | None = engine_hw
        self._hires_scale: float = 1.0 if engine_hw is None else reader.width / engine_hw[1]

    def run(self, chunk_size: int = 32, max_frames: int | None = None, start_frame: int | None = None, timing_doc: bool = False) -> PipelineStats:
        """Stream the sequence tick by tick; returns timing stats.

        Decode runs one chunk ahead on background threads (one per camera)
        so NVDEC overlaps with model compute; wall time approaches
        ``max(decode, compute)`` instead of their sum.

        By default the run covers the sequence's calibrated window
        ``[frame_start, frame_start + frame_count)`` so frames outside the
        reference interval are never fit; an explicit ``start_frame`` /
        ``max_frames`` (golden, profiling) overrides it. Either way the bound is
        clamped to ``reader.frame_cnt`` so a short video can't overrun.
        """
        import time
        from concurrent.futures import Future, ThreadPoolExecutor
        from itertools import repeat

        profiler: StageProfiler = StageProfiler()
        self.logger.setup(timing_doc=timing_doc)

        if start_frame is None:
            start_frame = self.sequence.frame_start
        start: float = time.perf_counter()
        frame_idx: int = start_frame
        if max_frames is None:
            window_end: int = self.sequence.frame_start + self.sequence.frame_count
            frame_count: int = min(window_end, self.reader.frame_cnt)
        else:
            frame_count = min(start_frame + max_frames, self.reader.frame_cnt)
        chunk_ranges: list[tuple[int, int]] = [
            (s, min(s + chunk_size, frame_count)) for s in range(start_frame, frame_count, chunk_size)
        ]

        def decode_chunk(bounds: tuple[int, int]) -> list[UInt8[torch.Tensor, "b 3 h w"]]:
            with ThreadPoolExecutor(max_workers=len(self.reader.video_readers)) as cam_pool:
                return list(
                    cam_pool.map(
                        lambda reader, b: reader.get_frames_in_range(b[0], b[1]),
                        self.reader.video_readers,
                        repeat(bounds),
                    )
                )

        def chunk_iter():
            prefetcher = ThreadPoolExecutor(max_workers=1)
            try:
                pending_chunk: Future | None = prefetcher.submit(decode_chunk, chunk_ranges[0]) if chunk_ranges else None
                for chunk_idx in range(len(chunk_ranges)):
                    assert pending_chunk is not None
                    chunk = pending_chunk.result()
                    pending_chunk = (
                        prefetcher.submit(decode_chunk, chunk_ranges[chunk_idx + 1])
                        if chunk_idx + 1 < len(chunk_ranges)
                        else None
                    )
                    yield chunk_ranges[chunk_idx][0], chunk
            finally:
                # finally (not fallthrough) so a consumer exception closing this
                # generator still releases the prefetch thread.
                prefetcher.shutdown(wait=False, cancel_futures=True)

        # Fit + its logging run one tick deep on a worker: fit(t) overlaps
        # track/landmarks(t+1) — fit feeds nothing upstream. With a collector
        # (validation) the tail is awaited inline to keep results exact-ordered.
        fit_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fit")
        pending_fit: Future | None = None
        try:
            for _chunk_start, chunk in chunk_iter():
                chunk_len: int = min(int(video.shape[0]) for video in chunk)
                for local_idx in range(chunk_len):
                    frames: list[UInt8[torch.Tensor, "3 h w"]] = [video[local_idx] for video in chunk]
                    frames_hires: list[UInt8[torch.Tensor, "3 hh ww"]] | None = None
                    if self.engine_hw is not None:
                        frames_hires = frames
                        lo: UInt8[torch.Tensor, "c 3 h w"] = (
                            torch.nn.functional.interpolate(torch.stack(frames).float(), size=self.engine_hw, mode="area")
                            .round_()
                            .clamp_(0, 255)
                            .to(torch.uint8)
                        )
                        frames = [lo[i] for i in range(lo.shape[0])]
                    tracks: list[CameraTracks] | None = None
                    landmarks: list[CameraLandmarks] | None = None
                    if self.tracker is not None:
                        with profiler.stage("track"):
                            tracks = self.tracker.step(frame_idx, frames)
                    if self.landmarks is not None and tracks is not None:
                        with profiler.stage("landmarks"):
                            landmarks = self.landmarks.estimate(frames, tracks, frames_hires=frames_hires, hires_scale=self._hires_scale)
                    if self.fitting is not None and landmarks is not None:
                        with profiler.stage("fit_wait"):
                            if pending_fit is not None:
                                pending_fit.result()

                        def fit_tail(idx: int, lms: list[CameraLandmarks]) -> TickFitOutput:
                            assert self.fitting is not None
                            out: TickFitOutput = self.fitting.step(idx, lms)
                            self.logger.log_tick_fit(idx, out.fits, out.triangulated, self.fitting.faces)
                            if out.metrics:
                                self.logger.log_tick_metrics(idx, {}, out.metrics)
                            return out

                        pending_fit = fit_worker.submit(fit_tail, frame_idx, landmarks)
                        if self.collector is not None:
                            with profiler.stage("fit_wait"):
                                fit_output: TickFitOutput | None = pending_fit.result()
                                pending_fit = None
                        else:
                            fit_output = None
                    else:
                        fit_output = None
                    with profiler.stage("log_video"):
                        self.logger.log_tick_video(frame_idx, frames)
                    if tracks is not None:
                        with profiler.stage("log_tracks"):
                            self.logger.log_tick_tracks(frame_idx, tracks, seg_stride=self.logger.seg_stride)
                    if landmarks is not None and frame_idx % 2 == 0:
                        # Dense 512-point overlays at 15 Hz halve their D2H cost;
                        # boxes/masks/mesh cadences are unchanged.
                        with profiler.stage("log_landmarks"):
                            self.logger.log_tick_landmarks(frame_idx, landmarks)
                    if self.collector is not None:
                        self.collector.collect(frame_idx, tracks, landmarks, fit_output)
                    with profiler.stage("log_metrics"):
                        self.logger.log_tick_metrics(frame_idx, dict(profiler.last_ms), {})
                    frame_idx += 1
            with profiler.stage("log_video"):
                if pending_fit is not None:
                    pending_fit.result()
                self.logger.flush()
        finally:
            # The in-process prefetcher releases its own pool in chunk_iter's
            # finally; here we only join the fit worker. wait=True: an in-flight
            # fit finishes in ~ms; a non-daemon worker left running would
            # otherwise block interpreter exit.
            fit_worker.shutdown(wait=True, cancel_futures=True)

        elapsed: float = time.perf_counter() - start
        return PipelineStats(ticks=frame_idx - start_frame, elapsed_s=elapsed, profiler=profiler)

    def close(self) -> None:
        """No-op retained for API symmetry.

        The in-process prefetcher owns its ``ThreadPoolExecutor`` and releases it
        in ``chunk_iter``'s ``finally`` block, so the pipeline holds no persistent
        decode resources to free. Kept so existing callers can still ``close()``.
        """


def build_streaming_pipeline(
    sequence: MultiViewSequence,
    resize_hw: tuple[int, int],
    device: str,
    tracker_config: TrackerConfig | None = None,
    fitter_config: FitterConfig | None = None,
    mammanet_weights: Path | None = None,
    trt_engine: Path | None = None,
    collector: ResultCollector | None = None,
    hires_crops: bool = True,
    proxy_dir: Path | None = None,
    seg_stride: int = 5,
) -> StreamingPipeline:
    """Assemble the standard decode->track->landmarks->fit->log pipeline.

    The single construction path shared by every tool (demos, benchmark,
    golden validation). Stages are optional and cumulative: tracking needs
    ``tracker_config``; landmarks additionally need ``mammanet_weights``;
    fitting additionally needs ``fitter_config``. ``device`` is propagated
    into the stage configs here so callers never mutate them.

    ``proxy_dir`` (from ``tools/make_proxies.py``) swaps the decode source to
    pre-transcoded ``<proxy_dir>/<cam>.mp4`` videos already at ``resize_hw``,
    sidestepping the 4K-NVDEC decode floor. It forces ``hires_crops=False``
    (no native frame exists to crop from); calibration still scales from the
    sequence's native intrinsics, so geometry is unchanged.
    """
    import dataclasses

    decode_sequence: MultiViewSequence = sequence
    if proxy_dir is not None:
        proxy_paths: list[Path] = [proxy_dir / f"{name}.mp4" for name in sequence.camera_names]
        missing: list[Path] = [p for p in proxy_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing proxies (run tools/make_proxies.py): {missing}")
        decode_sequence = dataclasses.replace(sequence, video_paths=proxy_paths)
        hires_crops = False

    scaled_cams: list[CameraCalibration] = sequence.scaled_cameras(resize_hw)
    # hires_crops: decode at native resolution; the pipeline downscales to
    # resize_hw per tick for tracking/logging and crops landmarks from native.
    reader = TorchCodecMultiVideoReader(list(decode_sequence.video_paths), device=device, resize_hw=None if hires_crops else resize_hw)
    logger = StreamLogger(sequence, resize_hw=resize_hw, seg_stride=seg_stride)

    tracker: MultiViewTracker | None = None
    if tracker_config is not None:
        tracker_config.device = device
        tracker = MultiViewTracker(scaled_cams, tracker_config)

    landmarks: LandmarkEstimator | None = None
    if tracker is not None and mammanet_weights is not None:
        landmarks = LandmarkEstimator(mammanet_weights, device=device, engine_path=trt_engine)

    fitting: FittingStage | None = None
    if landmarks is not None and fitter_config is not None:
        fitter_config.device = device
        fitting = FittingStage(scaled_cams, fitter_config)

    return StreamingPipeline(
        decode_sequence,  # proxy paths drive the reader; logger keeps the native sequence
        reader,
        logger,
        tracker=tracker,
        landmarks=landmarks,
        fitting=fitting,
        collector=collector,
        engine_hw=resize_hw if hires_crops else None,
    )
