"""Tests for TorchCodec-based video readers.

These tests use the street_dance videos which are synced multi-camera recordings.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from jaxtyping import UInt8
from numpy import ndarray

from simplecv.video_io import (
    TorchCodecMultiVideoReader,
    TorchCodecVideoReader,
    VideoReader,
)

# Test data directory
STREET_DANCE_DIR: Path = Path(__file__).parent.parent / "data" / "street_dance" / "videos"


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
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path)

        assert reader.width > 0, "Width should be positive"
        assert reader.height > 0, "Height should be positive"
        assert reader.fps > 0, "FPS should be positive"
        assert reader.frame_cnt > 0, "Frame count should be positive"
        assert reader.resolution == (reader.width, reader.height)

    def test_frame_count_matches_opencv(self, sample_video_path: Path) -> None:
        """Verify frame count matches OpenCV VideoReader."""
        tc_reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path)
        cv_reader: VideoReader = VideoReader(sample_video_path)

        # Allow small difference (TorchCodec might be more accurate)
        assert abs(tc_reader.frame_cnt - cv_reader.frame_cnt) <= 1, (
            f"Frame count mismatch: TorchCodec={tc_reader.frame_cnt}, OpenCV={cv_reader.frame_cnt}"
        )

    def test_frame_shape_and_dtype(self, sample_video_path: Path) -> None:
        """Test that decoded frames have correct shape and dtype."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path)
        frame: UInt8[ndarray, "h w 3"] = reader.get_frame(0)

        assert frame.ndim == 3, "Frame should be 3D (H, W, C)"
        assert frame.shape[2] == 3, "Frame should have 3 channels (BGR)"
        assert frame.shape[0] == reader.height, "Frame height mismatch"
        assert frame.shape[1] == reader.width, "Frame width mismatch"
        assert frame.dtype == np.uint8, "Frame should be uint8"

    def test_bgr_output_format(self, sample_video_path: Path) -> None:
        """Verify output is in BGR format (OpenCV convention)."""
        tc_reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path)
        cv_reader: VideoReader = VideoReader(sample_video_path)

        tc_frame: UInt8[ndarray, "h w 3"] = tc_reader.get_frame(0)
        cv_frame_raw = cv_reader[0]
        assert cv_frame_raw is not None
        cv_frame: UInt8[ndarray, "h w 3"] = cv_frame_raw  # type: ignore[assignment]

        # Frames should be very similar (both BGR)
        diff: float = float(np.abs(tc_frame.astype(float) - cv_frame.astype(float)).mean())
        assert diff < 5.0, f"Frames differ too much (mean diff={diff}), BGR conversion may be wrong"

    def test_sequential_iteration(self, sample_video_path: Path) -> None:
        """Test sequential iteration through video."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path)

        # Only decode first 10 frames for speed
        frame_count: int = 0
        max_frames: int = 10
        for frame in reader:
            assert frame.shape[2] == 3, "Frame should have 3 channels"
            frame_count += 1
            if frame_count >= max_frames:
                break

        assert frame_count == max_frames, f"Expected {max_frames} frames, got {frame_count}"

    def test_random_access(self, sample_video_path: Path) -> None:
        """Test random access to frames."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path)

        # Access frames in non-sequential order
        indices: list[int] = [0, 50, 25, 75, 10]
        for idx in indices:
            if idx < reader.frame_cnt:
                frame: UInt8[ndarray, "h w 3"] = reader.get_frame(idx)
                assert frame.shape[2] == 3, f"Frame {idx} should have 3 channels"

    def test_negative_indexing(self, sample_video_path: Path) -> None:
        """Test negative indexing."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path)

        last_frame: UInt8[ndarray, "h w 3"] = reader.get_frame(reader.frame_cnt - 1)
        explicit_last: UInt8[ndarray, "h w 3"] = reader.get_frame(reader.frame_cnt - 1)

        assert np.array_equal(last_frame, explicit_last), "Negative indexing should work"

    def test_slicing(self, sample_video_path: Path) -> None:
        """Test slice access."""
        reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path)

        frames_result = reader[0:5]
        frames: list[UInt8[ndarray, "h w 3"]] = cast(list, frames_result)
        assert len(frames) == 5, "Slice should return 5 frames"
        for frame in frames:
            assert frame.shape[2] == 3, "Each frame should have 3 channels"

    def test_context_manager(self, sample_video_path: Path) -> None:
        """Test context manager protocol."""
        with TorchCodecVideoReader(sample_video_path) as reader:
            frame: UInt8[ndarray, "h w 3"] = reader.get_frame(0)
            assert frame.shape[2] == 3


class TestTorchCodecVideoReaderBytes:
    """Tests for TorchCodecVideoReader with bytes input."""

    def test_bytes_input(self, sample_video_path: Path) -> None:
        """Test creating reader from bytes."""
        video_bytes: bytes = sample_video_path.read_bytes()
        reader: TorchCodecVideoReader = TorchCodecVideoReader(video_bytes)

        assert reader.width > 0
        assert reader.height > 0
        assert reader.frame_cnt > 0

    def test_bytes_matches_path(self, sample_video_path: Path) -> None:
        """Verify bytes and path inputs produce identical frames."""
        video_bytes: bytes = sample_video_path.read_bytes()

        path_reader: TorchCodecVideoReader = TorchCodecVideoReader(sample_video_path)
        bytes_reader: TorchCodecVideoReader = TorchCodecVideoReader(video_bytes)

        # Metadata should match
        assert path_reader.frame_cnt == bytes_reader.frame_cnt
        assert path_reader.width == bytes_reader.width
        assert path_reader.height == bytes_reader.height

        # First few frames should be identical
        for i in range(min(5, path_reader.frame_cnt)):
            path_frame: UInt8[ndarray, "h w 3"] = path_reader.get_frame(i)
            bytes_frame: UInt8[ndarray, "h w 3"] = bytes_reader.get_frame(i)
            assert np.array_equal(path_frame, bytes_frame), f"Frame {i} mismatch"


class TestTorchCodecMultiVideoReader:
    """Tests for TorchCodecMultiVideoReader."""

    def test_multi_video_creation(self, multi_video_paths: list[Path]) -> None:
        """Test creating multi-video reader."""
        sources: list[Path | bytes] = list(multi_video_paths)
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources)

        assert len(reader.video_readers) == len(multi_video_paths)
        assert len(reader.video_paths) == len(multi_video_paths)

    def test_multi_video_properties(self, multi_video_paths: list[Path]) -> None:
        """Test multi-video reader properties."""
        sources: list[Path | bytes] = list(multi_video_paths)
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources)

        assert reader.height > 0
        assert reader.width > 0
        assert len(reader) > 0

    def test_multi_video_iteration(self, multi_video_paths: list[Path]) -> None:
        """Test iterating through synchronized frames."""
        sources: list[Path | bytes] = list(multi_video_paths)
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources)

        frame_count: int = 0
        max_frames: int = 5
        for frame_list in reader:
            assert frame_list is not None
            assert len(frame_list) == len(multi_video_paths), "Should have one frame per video"
            for frame in frame_list:
                assert frame.shape[2] == 3, "Each frame should have 3 channels"
            frame_count += 1
            if frame_count >= max_frames:
                break

        assert frame_count == max_frames

    def test_multi_video_indexing(self, multi_video_paths: list[Path]) -> None:
        """Test random access to synchronized frames."""
        sources: list[Path | bytes] = list(multi_video_paths)
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources)

        frame_list: list = reader[0]
        assert len(frame_list) == len(multi_video_paths)

    def test_multi_video_with_bytes(self, multi_video_paths: list[Path]) -> None:
        """Test multi-video reader with mixed bytes and paths."""
        # First video as bytes, rest as paths
        sources: list[Path | bytes] = [
            multi_video_paths[0].read_bytes(),
            multi_video_paths[1],
            multi_video_paths[2],
        ]
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(sources)

        assert len(reader) > 0
        frame_list: list = reader[0]
        assert len(frame_list) == 3
