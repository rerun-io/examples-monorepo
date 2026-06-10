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
        use_mp_decode: bool = True,
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
        self.mp_decoder = None
        if use_mp_decode:
            from mamma.engine.mp_decode import MultiprocessDecoder

            # Persistent decode workers (~400 vs 140 cam-fps in-process on
            # torchcodec 0.10); spawned here so run() never pays startup.
            self.mp_decoder = MultiprocessDecoder(
                list(sequence.video_paths), resize_hw=(reader.height, reader.width)
            )

    def run(self, chunk_size: int = 32, max_frames: int | None = None, start_frame: int = 0) -> PipelineStats:
        """Stream the sequence tick by tick; returns timing stats.

        Decode runs one chunk ahead on background threads (one per camera)
        so NVDEC overlaps with model compute; wall time approaches
        ``max(decode, compute)`` instead of their sum.
        """
        import time
        from concurrent.futures import Future, ThreadPoolExecutor
        from itertools import repeat

        profiler: StageProfiler = StageProfiler()
        self.logger.setup()

        start: float = time.perf_counter()
        frame_idx: int = start_frame
        frame_count: int = self.reader.frame_cnt if max_frames is None else min(start_frame + max_frames, self.reader.frame_cnt)
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
            if self.mp_decoder is not None:
                yield from self.mp_decoder.iter_chunks(start_frame, frame_count)
                return
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
                            self.logger.log_tick_tracks(frame_idx, tracks)
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
        except BaseException:
            # Callers rarely close() on error paths — release the decode
            # workers here so a failed run can't leak NVDEC/CUDA contexts.
            self.close()
            raise
        finally:
            # wait=True: an in-flight fit finishes in ~ms; a non-daemon worker
            # left running would otherwise block interpreter exit.
            fit_worker.shutdown(wait=True, cancel_futures=True)

        elapsed: float = time.perf_counter() - start
        return PipelineStats(ticks=frame_idx - start_frame, elapsed_s=elapsed, profiler=profiler)

    def close(self) -> None:
        """Terminate persistent decode workers (call before discarding the pipeline)."""
        if self.mp_decoder is not None:
            self.mp_decoder.close()
            self.mp_decoder = None


def build_streaming_pipeline(
    sequence: MultiViewSequence,
    resize_hw: tuple[int, int],
    device: str,
    tracker_config: TrackerConfig | None = None,
    fitter_config: FitterConfig | None = None,
    mammanet_weights: Path | None = None,
    trt_engine: Path | None = None,
    collector: ResultCollector | None = None,
    use_mp_decode: bool = True,
    hires_crops: bool = True,
) -> StreamingPipeline:
    """Assemble the standard decode->track->landmarks->fit->log pipeline.

    The single construction path shared by every tool (demos, benchmark,
    golden validation). Stages are optional and cumulative: tracking needs
    ``tracker_config``; landmarks additionally need ``mammanet_weights``;
    fitting additionally needs ``fitter_config``. ``device`` is propagated
    into the stage configs here so callers never mutate them.
    """
    scaled_cams: list[CameraCalibration] = sequence.scaled_cameras(resize_hw)
    # hires_crops: decode at native resolution; the pipeline downscales to
    # resize_hw per tick for tracking/logging and crops landmarks from native.
    reader = TorchCodecMultiVideoReader(list(sequence.video_paths), device=device, resize_hw=None if hires_crops else resize_hw)
    logger = StreamLogger(sequence, resize_hw=resize_hw)

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
        sequence,
        reader,
        logger,
        tracker=tracker,
        landmarks=landmarks,
        fitting=fitting,
        collector=collector,
        use_mp_decode=use_mp_decode,
        engine_hw=resize_hw if hires_crops else None,
    )
