"""Public construction contracts for the streaming pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from simplecv.video_io import TorchCodecMultiVideoReader

from mamma.calibration.npz_contract import CameraCalibration
from mamma.datasets.sequence import MultiViewSequence
from mamma.engine.pipeline import StreamingPipeline, build_streaming_pipeline
from mamma.tracking.tracker import MultiViewTracker, TrackerConfig


def test_decode_device_is_independent_from_compute_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CPU decoding must not move model stages off CUDA."""
    captured_devices: list[str] = []

    def fake_reader_init(
        reader: TorchCodecMultiVideoReader,
        video_sources: list[Path | bytes],
        device: str | None = None,
        num_workers: int | None = None,
        num_ffmpeg_threads: int = 0,
        seek_mode: str = "approximate",
        resize_hw: tuple[int, int] | None = None,
    ) -> None:
        del num_workers, num_ffmpeg_threads, seek_mode
        captured_devices.append(str(device))
        reader.device = str(device)
        reader.resize_hw = resize_hw
        reader._video_paths = [Path(source) for source in video_sources]
        reader._video_readers = []
        reader._height = 480
        reader._width = 640
        reader._fps = 25.0
        reader._frame_cnt = 121

    monkeypatch.setattr(TorchCodecMultiVideoReader, "__init__", fake_reader_init)
    camera: CameraCalibration = CameraCalibration(
        name="0",
        k_matrix=np.eye(3, dtype=np.float64),
        world_to_cam=np.eye(4, dtype=np.float64),
        width=640,
        height=480,
    )
    sequence: MultiViewSequence = MultiViewSequence(
        name="decode-device-test",
        cameras=[camera],
        video_paths=[tmp_path / "0.mp4"],
        fps=25.0,
        frame_count=121,
    )

    pipeline: StreamingPipeline = build_streaming_pipeline(
        sequence,
        resize_hw=(480, 640),
        device="cuda",
        decode_device="cpu",
        hires_crops=True,
    )
    try:
        assert captured_devices == ["cpu"]
        assert pipeline.reader.device == "cpu"
        assert pipeline.engine_hw == (480, 640)
    finally:
        pipeline.logger._video_worker.shutdown(wait=True)


def test_cuda_decoder_is_constructed_after_cuda_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Model initialization must finish before the NVDEC context is created."""
    construction_order: list[str] = []

    def fake_tracker_init(
        tracker: MultiViewTracker,
        cameras: list[CameraCalibration],
        config: TrackerConfig,
    ) -> None:
        construction_order.append("tracker")
        tracker.cameras = cameras
        tracker.config = config

    def fake_reader_init(
        reader: TorchCodecMultiVideoReader,
        video_sources: list[Path | bytes],
        device: str | None = None,
        num_workers: int | None = None,
        num_ffmpeg_threads: int = 0,
        seek_mode: str = "approximate",
        resize_hw: tuple[int, int] | None = None,
    ) -> None:
        del num_workers, num_ffmpeg_threads, seek_mode
        construction_order.append("reader")
        reader.device = str(device)
        reader.resize_hw = resize_hw
        reader._video_paths = [Path(source) for source in video_sources]
        reader._video_readers = []
        reader._height = 480
        reader._width = 640
        reader._fps = 25.0
        reader._frame_cnt = 121

    monkeypatch.setattr(MultiViewTracker, "__init__", fake_tracker_init)
    monkeypatch.setattr(TorchCodecMultiVideoReader, "__init__", fake_reader_init)
    camera: CameraCalibration = CameraCalibration(
        name="0",
        k_matrix=np.eye(3, dtype=np.float64),
        world_to_cam=np.eye(4, dtype=np.float64),
        width=640,
        height=480,
    )
    sequence: MultiViewSequence = MultiViewSequence(
        name="decoder-order-test",
        cameras=[camera],
        video_paths=[tmp_path / "0.mp4"],
        fps=25.0,
        frame_count=121,
    )

    pipeline: StreamingPipeline = build_streaming_pipeline(
        sequence,
        resize_hw=(480, 640),
        device="cuda",
        tracker_config=TrackerConfig(),
        hires_crops=True,
    )
    try:
        assert construction_order == ["tracker", "reader"]
        assert pipeline.reader.device == "cuda"
    finally:
        pipeline.logger._video_worker.shutdown(wait=True)
