"""Whole-stride sampling and the local file sources (unit lane: synthetic files, CPU decode)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import ExifTags, Image

pytest.importorskip("torchcodec", reason="the video file source decodes with torchcodec")

from simplecv.sources import files  # noqa: E402
from simplecv.sources.catalog_video import sample_indices  # noqa: E402
from simplecv.sources.files import ImageDirSource, VideoFileSource, WebcamSource  # noqa: E402
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


@pytest.mark.parametrize("orientation", [1, 6])
def test_image_dir_resolution_comes_from_the_header_turned_as_the_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orientation: int) -> None:
    """EXIF orientation 6 (a phone held upright): cv2.imread turns the pixels, so the resolution turns with them."""
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    Image.new("RGB", (6, 4)).save(tmp_path / "img.jpg", exif=exif)
    source = ImageDirSource(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(files.cv2, "imread", lambda *_: pytest.fail("decoded the pixels"))
        width, height = source.resolution_wh()
    assert (1, 3, height, width) == frames_of(source)[0].rgb.shape == ((1, 3, 4, 6) if orientation == 1 else (1, 3, 6, 4))


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
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr("torchcodec.decoders.VideoDecoder.get_frame_at", lambda *_: pytest.fail("decoded a frame"))
        assert VideoFileSource(path, stride=2).resolution_wh() == (32, 16)  # the stream header


def test_webcam_source_converts_bgr_counts_frames_and_releases_the_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[int] = []
    reads: list[int] = []

    class Camera:
        def __init__(self, device: int) -> None:
            self.device, self.frames = device, 0

        def read(self) -> tuple[bool, np.ndarray]:
            self.frames += 1
            reads.append(self.device)
            return True, np.full((4, 6, 3), (10 * self.frames, 0, 255), dtype=np.uint8)  # BGR: blue

        def get(self, prop: int) -> float:
            return {files.cv2.CAP_PROP_FRAME_WIDTH: 6.0, files.cv2.CAP_PROP_FRAME_HEIGHT: 4.0}[prop]

        def release(self) -> None:
            released.append(self.device)

    monkeypatch.setattr(files.cv2, "VideoCapture", Camera)
    assert WebcamSource(2).resolution_wh() == (6, 4) and released == [2] and reads == []  # capture properties: no frame read
    released.clear()
    events = list(WebcamSource(2, max_frames=3).events())
    frames: list[MultiViewFrames] = [e.value for e in events]  # type: ignore[misc]
    assert [f.frame_index for f in frames] == [0, 1, 2] and frames[0].views == ("webcam",) and released == [2]
    assert [e.t_ns for e in events] == sorted(e.t_ns for e in events) and frames[0].rgb.shape == (1, 3, 4, 6)
    assert frames[1].rgb[0, :, 0, 0].tolist() == [255, 0, 20]  # RGB
