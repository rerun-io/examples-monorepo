from __future__ import annotations

import runpy
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import UInt8

import simplecv.apis.benchmark_torchcodec_cuda_singleview as benchmark_module


def test_benchmark_config_defaults_to_chunked_approximate_decode(tmp_path: Path) -> None:
    """The singleview benchmark defaults to the simple chunked CUDA path."""
    config = benchmark_module.BenchmarkConfig(video_path=tmp_path / "sample.mp4")

    assert config.chunk_size == 32
    assert config.seek_mode == "approximate"


def test_existing_tensor_benchmark_reports_gpu_ready_rgb_nchw_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenCV tensor baseline reports GPU-ready RGB NCHW uint8 batches."""
    first_bgr: UInt8[np.ndarray, "h w 3"] = np.array([[[1, 2, 3]]], dtype=np.uint8)
    second_bgr: UInt8[np.ndarray, "h w 3"] = np.array([[[4, 5, 6]]], dtype=np.uint8)

    class FakeVideoReader:
        instances: list["FakeVideoReader"] = []

        def __init__(self, _video_path: Path) -> None:
            self.frames: list[UInt8[np.ndarray, "h w 3"]] = [first_bgr, second_bgr]
            self.next_calls: int = 0
            self.instances.append(self)

        def __len__(self) -> int:
            return len(self.frames)

        def __iter__(self):
            return self

        def __next__(self):
            if self.next_calls >= len(self.frames):
                raise StopIteration
            frame: UInt8[np.ndarray, "h w 3"] = self.frames[self.next_calls]
            self.next_calls += 1
            return frame

    monkeypatch.setattr(benchmark_module, "VideoReader", FakeVideoReader)

    result = benchmark_module.benchmark_existing_tensor(
        Path("sample.mp4"),
        max_frames=1,
        device="cpu",
        chunk_size=1,
    )

    assert result.label == "VideoReader -> cpu"
    assert result.frames == 1
    assert "RGB NCHW uint8 tensor" in result.detail
    assert "checksum=3" in result.detail
    assert FakeVideoReader.instances[0].next_calls == 1


def test_chunked_benchmark_helpers_validate_chunk_size_before_decoding() -> None:
    """Chunked benchmark helpers reject invalid chunks before opening decoders."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        benchmark_module.benchmark_torchcodec_chunked(
            Path("missing.mp4"),
            device="cpu",
            max_frames=None,
            chunk_size=0,
            num_ffmpeg_threads=0,
            seek_mode="approximate",
        )

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        benchmark_module.benchmark_existing_tensor(
            Path("missing.mp4"),
            max_frames=None,
            device="cpu",
            chunk_size=0,
        )


def test_torchcodec_chunk_progress_reports_frame_rate_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunked singleview progress reports frames, while still showing chunk size."""
    class FakeTqdm:
        instances: list["FakeTqdm"] = []

        def __init__(self, iterable: Iterable[int] | None = None, **kwargs: object) -> None:
            self.iterable: Iterable[int] | None = iterable
            self.kwargs: dict[str, object] = kwargs
            self.updates: list[int] = []
            self.instances.append(self)

        def __iter__(self) -> Iterator[int]:
            assert self.iterable is not None
            return iter(self.iterable)

        def __enter__(self) -> "FakeTqdm":
            return self

        def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
            return None

        def update(self, n: int) -> None:
            self.updates.append(n)

    class FakeTorchCodecVideoReader:
        def __init__(self, _video_path: Path, *, device: str, num_ffmpeg_threads: int, seek_mode: str) -> None:
            self.frame_count: int = 3

        def __len__(self) -> int:
            return self.frame_count

        def iter_chunks(
            self,
            chunk_size: int,
            max_frames: int | None = None,
        ) -> Iterator[UInt8[torch.Tensor, "b 3 h w"]]:
            frame_count: int = self.frame_count if max_frames is None else min(max_frames, self.frame_count)
            for start in range(0, frame_count, chunk_size):
                stop: int = min(start + chunk_size, frame_count)
                yield self.get_frames_in_range(start, stop)

        def get_frames_in_range(self, start: int, stop: int) -> UInt8[torch.Tensor, "b 3 h w"]:
            frame_count: int = stop - start
            video: UInt8[torch.Tensor, "b 3 h w"] = torch.ones((frame_count, 3, 1, 1), dtype=torch.uint8)
            return video

    monkeypatch.setattr(benchmark_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(benchmark_module, "TorchCodecVideoReader", FakeTorchCodecVideoReader)

    result = benchmark_module.benchmark_torchcodec_chunked(
        Path("sample.mp4"),
        device="cpu",
        max_frames=None,
        chunk_size=2,
        num_ffmpeg_threads=0,
        seek_mode="approximate",
    )

    progress_bar: FakeTqdm = FakeTqdm.instances[0]
    assert result.frames == 3
    assert progress_bar.kwargs["total"] == 3
    assert progress_bar.kwargs["desc"] == "TorchCodec chunks (2 frames/chunk)"
    assert progress_bar.kwargs["unit"] == " frames"
    assert progress_bar.updates == [2, 1]


def test_tool_is_minimal_tyro_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI tool delegates parsing to tyro and execution to the API main."""
    tool_path: Path = Path(__file__).parent.parent / "tools" / "benchmark_torchcodec_cuda_singleview.py"
    config: object = object()
    main_calls: list[object] = []

    def fake_cli(config_cls: type, description: str) -> object:
        assert config_cls is benchmark_module.BenchmarkConfig
        assert "singleview" in description
        return config

    def fake_main(parsed_config: object) -> None:
        main_calls.append(parsed_config)

    monkeypatch.setattr("tyro.cli", fake_cli)
    monkeypatch.setattr(benchmark_module, "main", fake_main)

    runpy.run_path(str(tool_path), run_name="__main__")

    assert main_calls == [config]
