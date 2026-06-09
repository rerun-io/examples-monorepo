"""The resident streaming loop: decode -> track -> (landmarks/fit, M3+) -> log.

No disk writes happen anywhere in this loop; outputs exist only as Rerun logs
(and in-memory results returned to the caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from jaxtyping import UInt8
from simplecv.video_io import TorchCodecMultiVideoReader

from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.profiler import StageProfiler
from mamma.engine.types import CameraTracks
from mamma.fitting.stage import FittingStage, TickFitOutput
from mamma.landmarks.estimator import CameraLandmarks, LandmarkEstimator
from mamma.tracking.tracker import MultiViewTracker
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
        collector: "ResultCollector | None" = None,
    ) -> None:
        self.sequence: MultiViewSequence = sequence
        self.reader: TorchCodecMultiVideoReader = reader
        self.logger: StreamLogger = logger
        self.tracker: MultiViewTracker | None = tracker
        self.landmarks: LandmarkEstimator | None = landmarks
        self.fitting: FittingStage | None = fitting
        self.collector: ResultCollector | None = collector

    def run(self, chunk_size: int = 32, max_frames: int | None = None, start_frame: int = 0) -> PipelineStats:
        """Stream the sequence tick by tick; returns timing stats."""
        import time

        profiler: StageProfiler = StageProfiler()
        self.logger.setup()

        start: float = time.perf_counter()
        frame_idx: int = start_frame
        frame_count: int = self.reader.frame_cnt if max_frames is None else min(start_frame + max_frames, self.reader.frame_cnt)
        for chunk_start in range(start_frame, frame_count, chunk_size):
            chunk_stop: int = min(chunk_start + chunk_size, frame_count)
            with profiler.stage("decode"):
                chunk: list[UInt8[torch.Tensor, "b 3 h w"]] = [
                    reader.get_frames_in_range(chunk_start, chunk_stop) for reader in self.reader.video_readers
                ]
            chunk_len: int = min(int(video.shape[0]) for video in chunk)
            for local_idx in range(chunk_len):
                frames: list[UInt8[torch.Tensor, "3 h w"]] = [video[local_idx] for video in chunk]
                tracks: list[CameraTracks] | None = None
                landmarks: list[CameraLandmarks] | None = None
                if self.tracker is not None:
                    with profiler.stage("track"):
                        tracks = self.tracker.step(frame_idx, frames)
                if self.landmarks is not None and tracks is not None:
                    with profiler.stage("landmarks"):
                        landmarks = self.landmarks.estimate(frames, tracks)
                fit_output: TickFitOutput | None = None
                if self.fitting is not None and landmarks is not None:
                    with profiler.stage("fit"):
                        fit_output = self.fitting.step(frame_idx, landmarks)
                with profiler.stage("log_video"):
                    self.logger.log_tick_video(frame_idx, frames)
                if tracks is not None:
                    with profiler.stage("log_tracks"):
                        self.logger.log_tick_tracks(frame_idx, tracks)
                if landmarks is not None:
                    with profiler.stage("log_landmarks"):
                        self.logger.log_tick_landmarks(frame_idx, landmarks)
                if fit_output is not None and self.fitting is not None:
                    with profiler.stage("log_fit"):
                        self.logger.log_tick_fit(frame_idx, fit_output.fits, fit_output.triangulated, self.fitting.faces)
                if self.collector is not None:
                    self.collector.collect(frame_idx, tracks, landmarks, fit_output)
                frame_idx += 1
        with profiler.stage("log_video"):
            self.logger.flush()

        elapsed: float = time.perf_counter() - start
        return PipelineStats(ticks=frame_idx - start_frame, elapsed_s=elapsed, profiler=profiler)
