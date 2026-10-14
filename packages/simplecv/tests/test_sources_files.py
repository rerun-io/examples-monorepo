"""Whole-stride sampling and the local file sources (unit lane: synthetic files, CPU decode)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("torchcodec", reason="the video file source decodes with torchcodec")

from simplecv.sources.catalog_video import sample_indices  # noqa: E402
from simplecv.sources.files import ImageDirSource, VideoFileSource  # noqa: E402
from simplecv.types.frames import MultiViewFrames  # noqa: E402


@settings(max_examples=100, deadline=None)
@given(n=st.integers(1, 300), stride=st.integers(1, 7), start=st.integers(0, 200), length=st.integers(1, 300))
def test_stride_keeps_every_nth_window_frame_in_whole_strides(n: int, stride: int, start: int, length: int) -> None:
    times = np.arange(n, dtype=np.int64) * 10
    window = (start * 10, (start + length) * 10)
    inside = np.flatnonzero((times >= window[0]) & (times < window[1]))
    expected = inside[: inside.size // stride * stride : stride]
    if expected.size == 0:
        with pytest.raises(ValueError, match="no frames"):
            sample_indices(times, "all", window, stride)
        return
    np.testing.assert_array_equal(sample_indices(times, "all", window, stride), expected)
    assert expected.size == inside.size // stride  # a decoder reading i * stride for i < m // stride


def frames_of(source: ImageDirSource | VideoFileSource) -> list[MultiViewFrames]:
    events = list(source.events())
    assert all(e.port == "frames" and isinstance(e.value, MultiViewFrames) and e.t_ns == e.value.t_ns for e in events)
    return [e.value for e in events]  # type: ignore[misc]


def test_image_dir_source_sorts_naturally_strides_and_converts_to_rgb(tmp_path: Path) -> None:
    for i in (1, 2, 10):
        cv2.imwrite(str(tmp_path / f"img{i}.png"), np.full((4, 6, 3), (i, 0, 200), dtype=np.uint8))  # BGR
    (tmp_path / "notes.txt").write_text("not an image")
    frames = frames_of(ImageDirSource(tmp_path, stride=2, fps=10.0))
    assert [f.frame_index for f in frames] == [0, 2] and [f.t_ns for f in frames] == [0, 200_000_000]
    assert frames[1].rgb.shape == (1, 3, 4, 6) and frames[1].rgb[0, :, 0, 0].tolist() == [200, 0, 10]  # img10, as RGB
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no .png"):
        ImageDirSource(tmp_path / "empty")


def test_video_file_source_reads_whole_strides_on_the_stream_clock(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 16))
    for i in range(7):
        writer.write(np.full((16, 32, 3), 30 * i, dtype=np.uint8))
    writer.release()
    frames = frames_of(VideoFileSource(path, stride=2))
    assert [f.frame_index for f in frames] == [0, 2, 4]  # 7 // 2 whole strides
    np.testing.assert_allclose([f.t_ns for f in frames], [0, 200_000_000, 400_000_000], atol=1_000_000)
    assert frames[0].rgb.shape == (1, 3, 16, 32) and abs(int(frames[2].rgb.float().mean()) - 120) <= 3
    assert len(frames_of(VideoFileSource(path, stride=1, max_frames=2))) == 2
