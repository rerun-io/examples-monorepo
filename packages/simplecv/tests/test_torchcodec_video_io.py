"""Tests for TorchCodec-based video readers.

These tests use the street_dance videos which are synced multi-camera recordings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import UInt8

from simplecv.video_io import (
    TorchCodecMultiVideoReader,
    TorchCodecVideoReader,
    VideoReader,
    rgb_chw_tensor_to_bgr_hwc,
)

# Test data directory
STREET_DANCE_DIR: Path = Path(__file__).parent.parent / "data" / "street_dance" / "videos"


def test_rgb_chw_tensor_to_bgr_hwc_converts_layout_and_channels() -> None:
    """Convert RGB CHW tensors to BGR HWC numpy images for legacy APIs."""
    rgb_chw: UInt8[torch.Tensor, "3 h w"] = torch.tensor(
        [
            [[1, 2]],
            [[3, 4]],
            [[5, 6]],
        ],
        dtype=torch.uint8,
    )

    bgr_hwc: np.ndarray = rgb_chw_tensor_to_bgr_hwc(rgb_chw)
    expected_bgr_hwc: np.ndarray = np.array([[[5, 3, 1], [6, 4, 2]]], dtype=np.uint8)

    assert bgr_hwc.flags.c_contiguous
    assert np.array_equal(bgr_hwc, expected_bgr_hwc)


@pytest.fixture
def sample_video_path() -> Path:
    """Get path to first street_dance video."""
    video_path: Path = STREET_DANCE_DIR / "01.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")
    return video_path


@pytest.fixture
def multi_video_paths() -> list[Path]:
    """Get paths to first 3 street_dance videos for multi-camera tests."""
    paths: list[Path] = [STREET_DANCE_DIR / f"{i:02d}.mp4" for i in [1, 3, 5]]
    for p in paths:
        if not p.exists():
            pytest.skip(f"Test video not found: {p}")
    return paths


class TestTorchCodecVideoReader:
    """Tests for TorchCodecVideoReader with file path input."""

    def test_metadata_properties(self, sample_video_path: Path) -> None:
        """Test that video metadata is correctly extracted."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path, device="cpu")

        assert reader.width > 0, "Width should be positive"
        assert reader.height > 0, "Height should be positive"
        assert reader.fps > 0, "FPS should be positive"
        assert reader.frame_cnt > 0, "Frame count should be positive"
        assert reader.resolution == (reader.width, reader.height)

    def test_frame_count_matches_opencv(self, sample_video_path: Path) -> None:
        """Verify frame count matches OpenCV VideoReader."""
        tc_reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path, device="cpu")
        cv_reader: VideoReader = VideoReader(sample_video_path)

        # Allow small difference (TorchCodec might be more accurate)
        assert abs(tc_reader.frame_cnt - cv_reader.frame_cnt) <= 1, (
            f"Frame count mismatch: TorchCodec={tc_reader.frame_cnt}, OpenCV={cv_reader.frame_cnt}"
        )

    def test_frame_shape_and_dtype(self, sample_video_path: Path) -> None:
        """Test that decoded frames have correct shape and dtype."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path, device="cpu")
        frame: UInt8[torch.Tensor, "3 h w"] = reader.get_frame(0)

        assert frame.ndim == 3, "Frame should be 3D (C, H, W)"
        assert frame.shape[0] == 3, "Frame should have 3 channels (RGB)"
        assert frame.shape[1] == reader.height, "Frame height mismatch"
        assert frame.shape[2] == reader.width, "Frame width mismatch"
        assert frame.dtype == torch.uint8, "Frame should be uint8"

    def test_rgb_output_format(self, sample_video_path: Path) -> None:
        """Verify output is RGB tensor format."""
        tc_reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path, device="cpu")
        cv_reader: VideoReader = VideoReader(sample_video_path)

        tc_frame: UInt8[torch.Tensor, "3 h w"] = tc_reader.get_frame(0)
        cv_frame_raw = cv_reader[0]
        assert cv_frame_raw is not None
        assert isinstance(cv_frame_raw, np.ndarray)
        cv_rgb_chw_np: np.ndarray = cv_frame_raw[:, :, ::-1].transpose(2, 0, 1).copy()
        cv_rgb_chw: UInt8[torch.Tensor, "3 h w"] = torch.from_numpy(cv_rgb_chw_np)

        diff: float = float((tc_frame.to(torch.float32) - cv_rgb_chw.to(torch.float32)).abs().mean().item())
        assert diff < 5.0, f"Frames differ too much (mean diff={diff}), RGB conversion may be wrong"

    def test_sequential_iteration(self, sample_video_path: Path) -> None:
        """Test sequential iteration through video."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path, device="cpu")

        # Only decode first 10 frames for speed
        frame_count: int = 0
        max_frames: int = 10
        for frame in reader:
            assert frame.shape[0] == 3, "Frame should have 3 channels"
            frame_count += 1
            if frame_count >= max_frames:
                break

        assert frame_count == max_frames, f"Expected {max_frames} frames, got {frame_count}"

    def test_random_access(self, sample_video_path: Path) -> None:
        """Test random access to frames."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path, device="cpu")

        # Access frames in non-sequential order
        indices: list[int] = [0, 50, 25, 75, 10]
        for idx in indices:
            if idx < reader.frame_cnt:
                frame: UInt8[torch.Tensor, "3 h w"] = reader.get_frame(idx)
                assert frame.shape[0] == 3, f"Frame {idx} should have 3 channels"

    def test_negative_indexing(self, sample_video_path: Path) -> None:
        """Test negative indexing."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path, device="cpu")

        last_frame_result = reader[-1]
        assert isinstance(last_frame_result, torch.Tensor)
        last_frame: UInt8[torch.Tensor, "3 h w"] = last_frame_result
        explicit_last: UInt8[torch.Tensor, "3 h w"] = reader.get_frame(reader.frame_cnt - 1)

        assert torch.equal(last_frame, explicit_last), "Negative indexing should work"

    def test_slicing(self, sample_video_path: Path) -> None:
        """Test slice access."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path, device="cpu")

        frames_result = reader[0:5]
        assert isinstance(frames_result, torch.Tensor)
        frames: UInt8[torch.Tensor, "b 3 h w"] = frames_result
        assert frames.shape[0] == 5, "Slice should return 5 frames"
        assert frames.shape[1] == 3, "Each frame should have 3 channels"

    def test_context_manager(self, sample_video_path: Path) -> None:
        """Test context manager protocol."""
        with TorchCodecVideoReader(sample_video_path, device="cpu") as reader:
            frame: UInt8[torch.Tensor, "3 h w"] = reader.get_frame(0)
            assert frame.shape[0] == 3


class TestTorchCodecVideoReaderBytes:
    """Tests for TorchCodecVideoReader with bytes input."""

    def test_bytes_input(self, sample_video_path: Path) -> None:
        """Test creating reader from bytes."""
        video_bytes: bytes = sample_video_path.read_bytes()
        reader: TorchCodecVideoReader = TorchCodecVideoReader(video_bytes, device="cpu")

        assert reader.width > 0
        assert reader.height > 0
        assert reader.frame_cnt > 0

    def test_bytes_matches_path(self, sample_video_path: Path) -> None:
        """Verify bytes and path inputs produce identical frames."""
        video_bytes: bytes = sample_video_path.read_bytes()

        path_reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path, device="cpu")
        bytes_reader: TorchCodecVideoReader = TorchCodecVideoReader(video_bytes, device="cpu")

        # Metadata should match
        assert path_reader.frame_cnt == bytes_reader.frame_cnt
        assert path_reader.width == bytes_reader.width
        assert path_reader.height == bytes_reader.height

        # First few frames should be identical
        for i in range(min(5, path_reader.frame_cnt)):
            path_frame: UInt8[torch.Tensor, "3 h w"] = path_reader.get_frame(i)
            bytes_frame: UInt8[torch.Tensor, "3 h w"] = bytes_reader.get_frame(i)
            assert torch.equal(path_frame, bytes_frame), f"Frame {i} mismatch"


class TestTorchCodecMultiVideoReader:
    """Tests for TorchCodecMultiVideoReader."""

    def test_multi_video_creation(self, multi_video_paths: list[Path]) -> None:
        """Test creating multi-video reader."""
        sources: list[Path | bytes] = list(multi_video_paths)
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources, device="cpu")

        assert len(reader.video_readers) == len(multi_video_paths)
        assert len(reader.video_paths) == len(multi_video_paths)

    def test_multi_video_properties(self, multi_video_paths: list[Path]) -> None:
        """Test multi-video reader properties."""
        sources: list[Path | bytes] = list(multi_video_paths)
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources, device="cpu")

        assert reader.height > 0
        assert reader.width > 0
        assert len(reader) > 0

    def test_multi_video_iteration(self, multi_video_paths: list[Path]) -> None:
        """Test iterating through synchronized frames."""
        sources: list[Path | bytes] = list(multi_video_paths)
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources, device="cpu")

        frame_count: int = 0
        max_frames: int = 5
        for frame_list in reader:
            assert frame_list is not None
            assert len(frame_list) == len(multi_video_paths), "Should have one frame per video"
            for frame in frame_list:
                assert frame.shape[0] == 3, "Each frame should have 3 channels"
            frame_count += 1
            if frame_count >= max_frames:
                break

        assert frame_count == max_frames

    def test_multi_video_indexing(self, multi_video_paths: list[Path]) -> None:
        """Test random access to synchronized frames."""
        sources: list[Path | bytes] = list(multi_video_paths)
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources, device="cpu")

        frame_list: list = reader[0]
        assert len(frame_list) == len(multi_video_paths)
        assert frame_list[0].shape[0] == 3

    def test_multi_video_with_bytes(self, multi_video_paths: list[Path]) -> None:
        """Test multi-video reader with mixed bytes and paths."""
        # First video as bytes, rest as paths
        sources: list[Path | bytes] = [
            multi_video_paths[0].read_bytes(),
            multi_video_paths[1],
            multi_video_paths[2],
        ]
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources, device="cpu")

        assert len(reader) > 0
        frame_list: list = reader[0]
        assert len(frame_list) == 3


class TestTorchCodecMultiVideoReaderChunks:
    """Tests for chunked TorchCodec multiview decode."""

    def test_reader_exposes_chunked_decode_only(self) -> None:
        """The tensor reader should not expose a full materialization API."""
        assert not hasattr(TorchCodecMultiVideoReader, "read_all")

    def test_iter_chunks_decodes_full_sequence_in_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Iterate over the full sequence without materializing every frame at once."""
        import sys
        import types

        import torch

        class _Metadata:
            width: int = 4
            height: int = 2
            num_frames: int = 5
            average_fps: float = 60.0

        class _FrameBatch:
            def __init__(self, data: torch.Tensor) -> None:
                self.data: torch.Tensor = data

        class _FakeVideoDecoder:
            def __init__(self, source: str | Path, **_kwargs: object) -> None:
                self.source: str | Path = source
                self.metadata: _Metadata = _Metadata()

            def get_frames_in_range(self, start: int, stop: int, step: int = 1) -> _FrameBatch:
                frames: torch.Tensor = torch.arange(start, stop, step, dtype=torch.uint8)[:, None, None, None]
                return _FrameBatch(frames.expand(-1, 3, self.metadata.height, self.metadata.width).contiguous())

        fake_decoders_module = types.SimpleNamespace(VideoDecoder=_FakeVideoDecoder)
        monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake_decoders_module)

        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(
            [Path("cam0.mp4"), Path("cam1.mp4")],
            device="cpu",
            num_workers=1,
        )
        chunks = list(reader.iter_chunks(chunk_size=2))

        assert [chunk[0].shape[0] for chunk in chunks] == [2, 2, 1]
        assert int(chunks[0][0][0, 0, 0, 0].item()) == 0
        assert int(chunks[1][0][0, 0, 0, 0].item()) == 2
        assert int(chunks[-1][0][0, 0, 0, 0].item()) == 4

    def test_reader_defaults_to_approximate_seek(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The multiview tensor reader defaults to the selected approximate seek mode."""
        import sys
        import types

        import torch

        class _Metadata:
            width: int = 4
            height: int = 2
            num_frames: int = 6
            average_fps: float = 60.0

        class _FrameBatch:
            def __init__(self, data: torch.Tensor) -> None:
                self.data: torch.Tensor = data

        class _FakeVideoDecoder:
            def __init__(self, source: str | Path, **_kwargs: object) -> None:
                self.source: str | Path = source
                self.kwargs: dict[str, object] = dict(_kwargs)
                self.metadata: _Metadata = _Metadata()
                decoder_kwargs.append(self.kwargs)

            def get_frames_in_range(self, start: int, stop: int, step: int = 1) -> _FrameBatch:
                frames: torch.Tensor = torch.arange(start, stop, step, dtype=torch.uint8)[:, None, None, None]
                return _FrameBatch(frames.expand(-1, 3, self.metadata.height, self.metadata.width).contiguous())

        decoder_kwargs: list[dict[str, object]] = []
        fake_decoders_module = types.SimpleNamespace(VideoDecoder=_FakeVideoDecoder)
        monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake_decoders_module)

        TorchCodecMultiVideoReader(
            [Path("cam0.mp4"), Path("cam1.mp4")],
            device="cpu",
            num_workers=1,
        )

        assert {kwargs["seek_mode"] for kwargs in decoder_kwargs} == {"approximate"}
