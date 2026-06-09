from __future__ import annotations

import runpy
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import UInt8

import simplecv.apis.benchmark_torchcodec_cuda_multiview as benchmark_module


def test_find_video_paths_searches_directory_in_sorted_order(tmp_path: Path) -> None:
    """Benchmark inputs are discovered as sorted mp4s from a directory."""
    video_dir: Path = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "cam_b.mp4").touch()
    (video_dir / "notes.txt").touch()
    (video_dir / "cam_a.mp4").touch()

    video_paths: list[Path] = benchmark_module.find_video_paths(video_dir)

    assert [path.name for path in video_paths] == ["cam_a.mp4", "cam_b.mp4"]


def test_find_video_paths_prefers_rgb_low_videos(tmp_path: Path) -> None:
    """Assembly101-style RGB exo videos are selected when mixed mp4s exist."""
    video_dir: Path = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "C10095_rgb_low.mp4").touch()
    (video_dir / "C10115_rgb_low.mp4").touch()
    (video_dir / "HMC_84346135_mono10bit_low.mp4").touch()

    video_paths: list[Path] = benchmark_module.find_video_paths(video_dir)

    assert [path.name for path in video_paths] == ["C10095_rgb_low.mp4", "C10115_rgb_low.mp4"]


def test_benchmark_config_defaults_to_chunked_approximate_decode(tmp_path: Path) -> None:
    """The benchmark defaults to the chosen simple chunked CUDA path."""
    config = benchmark_module.BenchmarkConfig(video_dir=tmp_path)

    assert config.chunk_size == 32
    assert config.seek_mode == "approximate"


def test_existing_tensor_benchmark_reports_gpu_ready_rgb_nchw_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenCV multiview tensor baseline reports GPU-ready RGB NCHW uint8 batches."""
    cam0_frame0: UInt8[np.ndarray, "h w 3"] = np.array([[[1, 2, 3]]], dtype=np.uint8)
    cam1_frame0: UInt8[np.ndarray, "h w 3"] = np.array([[[4, 5, 6]]], dtype=np.uint8)
    cam0_frame1: UInt8[np.ndarray, "h w 3"] = np.array([[[7, 8, 9]]], dtype=np.uint8)
    cam1_frame1: UInt8[np.ndarray, "h w 3"] = np.array([[[10, 11, 12]]], dtype=np.uint8)

    class FakeMultiVideoReader:
        instances: list["FakeMultiVideoReader"] = []

        def __init__(self, _video_paths: list[Path]) -> None:
            self.frames: list[list[UInt8[np.ndarray, "h w 3"]]] = [
                [cam0_frame0, cam1_frame0],
                [cam0_frame1, cam1_frame1],
            ]
            self.next_calls: int = 0
            self.instances.append(self)

        def __len__(self) -> int:
            return len(self.frames)

        def __iter__(self):
            return self

        def __next__(self):
            if self.next_calls >= len(self.frames):
                raise StopIteration
            frames: list[UInt8[np.ndarray, "h w 3"]] = self.frames[self.next_calls]
            self.next_calls += 1
            return frames

    monkeypatch.setattr(benchmark_module, "MultiVideoReader", FakeMultiVideoReader)

    result = benchmark_module.benchmark_existing_tensor(
        [Path("cam0.mp4"), Path("cam1.mp4")],
        max_frames=1,
        device="cpu",
        chunk_size=1,
    )

    assert result.label == "MultiVideoReader -> cpu"
    assert result.frames == 2
    assert "RGB NCHW uint8 tensors" in result.detail
    assert "checksum=9" in result.detail
    assert FakeMultiVideoReader.instances[0].next_calls == 1


def test_chunked_benchmark_helpers_validate_chunk_size_before_decoding() -> None:
    """Chunked benchmark helpers reject invalid chunks before opening decoders."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        benchmark_module.benchmark_torchcodec_chunked(
            [Path("cam0.mp4"), Path("cam1.mp4")],
            device="cpu",
            max_frames=None,
            num_workers=None,
            chunk_size=0,
            num_ffmpeg_threads=0,
            seek_mode="approximate",
        )

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        benchmark_module.benchmark_existing_tensor(
            [Path("cam0.mp4"), Path("cam1.mp4")],
            max_frames=None,
            device="cpu",
            chunk_size=0,
        )


def test_multiview_tensor_progress_reports_camera_frame_rate_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunked multiview progress reports camera-frames and shows the chunk shape."""
    cam0_frame0: UInt8[np.ndarray, "h w 3"] = np.array([[[1, 2, 3]]], dtype=np.uint8)
    cam1_frame0: UInt8[np.ndarray, "h w 3"] = np.array([[[4, 5, 6]]], dtype=np.uint8)
    cam0_frame1: UInt8[np.ndarray, "h w 3"] = np.array([[[7, 8, 9]]], dtype=np.uint8)
    cam1_frame1: UInt8[np.ndarray, "h w 3"] = np.array([[[10, 11, 12]]], dtype=np.uint8)

    class FakeTqdm:
        instances: list["FakeTqdm"] = []

        def __init__(self, iterable: Iterable[object] | None = None, **kwargs: object) -> None:
            self.iterable: Iterable[object] | None = iterable
            self.kwargs: dict[str, object] = kwargs
            self.updates: list[int] = []
            self.instances.append(self)

        def __iter__(self) -> Iterator[object]:
            assert self.iterable is not None
            return iter(self.iterable)

        def __enter__(self) -> "FakeTqdm":
            return self

        def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
            return None

        def update(self, n: int) -> None:
            self.updates.append(n)

    class FakeMultiVideoReader:
        def __init__(self, _video_paths: list[Path]) -> None:
            self.frames: list[list[UInt8[np.ndarray, "h w 3"]]] = [
                [cam0_frame0, cam1_frame0],
                [cam0_frame1, cam1_frame1],
            ]
            self.next_calls: int = 0

        def __len__(self) -> int:
            return len(self.frames)

        def __iter__(self):
            return self

        def __next__(self):
            if self.next_calls >= len(self.frames):
                raise StopIteration
            frames: list[UInt8[np.ndarray, "h w 3"]] = self.frames[self.next_calls]
            self.next_calls += 1
            return frames

    monkeypatch.setattr(benchmark_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(benchmark_module, "MultiVideoReader", FakeMultiVideoReader)

    result = benchmark_module.benchmark_existing_tensor(
        [Path("cam0.mp4"), Path("cam1.mp4")],
        max_frames=None,
        device="cpu",
        chunk_size=2,
    )

    progress_bar: FakeTqdm = FakeTqdm.instances[0]
    assert result.frames == 4
    assert progress_bar.kwargs["total"] == 4
    assert progress_bar.kwargs["desc"] == "MultiVideoReader -> cpu (2 frames/chunk x 2 cameras)"
    assert progress_bar.kwargs["unit"] == " camera-frames"
    assert progress_bar.updates == [4]


def test_tool_is_minimal_tyro_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI tool delegates parsing to tyro and execution to the API main."""
    tool_path: Path = Path(__file__).parent.parent / "tools" / "benchmark_torchcodec_cuda_multiview.py"
    config: object = object()
    main_calls: list[object] = []

    def fake_cli(config_cls: type, description: str) -> object:
        assert config_cls is benchmark_module.BenchmarkConfig
        assert "multiview" in description
        return config

    def fake_main(parsed_config: object) -> None:
        main_calls.append(parsed_config)

    monkeypatch.setattr("tyro.cli", fake_cli)
    monkeypatch.setattr(benchmark_module, "main", fake_main)

    runpy.run_path(str(tool_path), run_name="__main__")

    assert main_calls == [config]
