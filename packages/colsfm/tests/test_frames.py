"""The frame source: the catalog manifest, the backend an input directory selects, the bounded video store, the folder decode."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from colsfm import frames as frames_module
from colsfm.frame_manifest import FRAME_SOURCE_FILE, FrameSourceManifest, StreamKeyframes, read_manifest, write_manifest
from colsfm.frames import ImageFolderFrames, VideoStreamFrames, open_frame_source


def _manifest() -> FrameSourceManifest:
    return FrameSourceManifest(
        catalog_url="rerun+http://127.0.0.1:51235",
        dataset="msd-g2",
        segment="segment",
        timeline="timestamp",
        streams=[StreamKeyframes(entity="world/rig_00/cam_00/pinhole/video", names=["cam0/10", "cam0/20"], t_ns=[10, 20])],
    )


def test_a_manifest_round_trips_through_its_file(tmp_path: Path) -> None:
    """What the catalog door writes is what the run reads back."""
    path = tmp_path / FRAME_SOURCE_FILE
    write_manifest(path, _manifest())
    assert read_manifest(path) == _manifest()


def test_a_manifest_with_an_unknown_field_is_refused_naming_the_file(tmp_path: Path) -> None:
    """The manifest is a file colsfm owns, so a stray field is an error, not silently dropped."""
    path = tmp_path / FRAME_SOURCE_FILE
    path.write_text('{"catalog_url": "x", "dataset": "d", "segment": "s", "timeline": "t", "streams": [], "extra": 1}')
    with pytest.raises(ValueError, match=FRAME_SOURCE_FILE):
        read_manifest(path)


def test_names_and_timestamps_must_pair_up() -> None:
    """A stream's two lists are parallel; a length mismatch would key frames to the wrong names."""
    with pytest.raises(ValueError, match="names for"):
        StreamKeyframes(entity="e", names=["a"], t_ns=[1, 2])


def test_an_input_directory_with_a_manifest_reads_the_video_streams(tmp_path: Path) -> None:
    """The backend follows the directory: a manifest means the catalog's streams, none means the files."""
    assert isinstance(open_frame_source(tmp_path), ImageFolderFrames)
    write_manifest(tmp_path / FRAME_SOURCE_FILE, _manifest())
    assert isinstance(open_frame_source(tmp_path), VideoStreamFrames)


@pytest.mark.integration
def test_the_image_folder_decodes_jpeg_and_png_on_the_gpu_like_opencv(tmp_path: Path) -> None:
    """nvJPEG and the OpenCV fallback give RGB `[3, h, w]` frames matching OpenCV's own decode."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (nvJPEG)")
    rng = np.random.default_rng(0)
    bgr = cv2.GaussianBlur(rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8), (5, 5), 0)
    (tmp_path / "cam").mkdir()
    # 4:4:4 so the comparison isolates the IDCT: decoders upsample 4:2:0 chroma differently (libjpeg's "fancy"
    # upsampling vs nvJPEG's), which moves single noisy pixels by tens of levels and is no fault of either.
    cv2.imwrite(
        str(tmp_path / "cam" / "a.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    )
    cv2.imwrite(str(tmp_path / "cam" / "b.png"), bgr)
    frames = dict(ImageFolderFrames(tmp_path).frames(["cam/a.jpg", "cam/b.png"]))
    for name in ("cam/a.jpg", "cam/b.png"):
        rgb = frames[name]
        assert rgb.is_cuda and rgb.dtype == torch.uint8 and tuple(rgb.shape) == (3, 48, 64)
        decoded = cv2.imread(str(tmp_path / name), cv2.IMREAD_COLOR)
        assert decoded is not None
        reference = decoded[:, :, ::-1].transpose(2, 0, 1).astype(np.int16)
        difference = np.abs(rgb.cpu().numpy().astype(np.int16) - reference)
        # PNG is lossless; two JPEG decoders may round the IDCT differently by a level or two.
        assert difference.max() <= (0 if name.endswith(".png") else 3)


SAMPLES: int = 100
"""Samples in the fake stream; keyframe `k` is sample `k` at `10 k` ns, and its pixels are all `k`."""


class _FakeVideo:
    """Stands in for one stream's NVDEC decoder: records what was decoded, frames whose pixels name their sample."""

    def __init__(self) -> None:
        self.requested: list[int] = []

    def get_frames_at(self, indices: Sequence[int]) -> torch.Tensor:
        self.requested.extend(indices)
        return torch.stack([torch.full((3, 4, 4), index, dtype=torch.uint8) for index in indices])


def _stream_manifest(camera: str = "cam0") -> FrameSourceManifest:
    names = [f"{camera}/{10 * k}" for k in range(SAMPLES)]
    return FrameSourceManifest(
        catalog_url="rerun+http://127.0.0.1:51235",
        dataset="msd-g2",
        segment="segment",
        timeline="timestamp",
        streams=[StreamKeyframes(entity=f"world/rig_00/{camera}/pinhole/video", names=names, t_ns=[10 * k for k in range(SAMPLES)])],
    )


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[_FakeVideo]:
    """Every decoder `_open_streams` hands out, in opening order."""
    videos: list[_FakeVideo] = []

    def fake_open(manifest: FrameSourceManifest, device: torch.device) -> list[tuple[np.ndarray, _FakeVideo]]:
        opened: list[tuple[np.ndarray, _FakeVideo]] = []
        for _ in manifest.streams:
            videos.append(_FakeVideo())
            opened.append((np.arange(SAMPLES, dtype=np.int64) * 10, videos[-1]))
        return opened

    monkeypatch.setattr(frames_module, "_open_streams", fake_open)
    return videos


FRAME_BYTES: int = 3 * 4 * 4


def test_a_request_decodes_only_its_names_and_yields_them_in_its_order(opened: list[_FakeVideo]) -> None:
    """Three keyframes of a hundred: three decodes, in packet order, handed back in the order asked."""
    source = VideoStreamFrames(_stream_manifest(), device=torch.device("cpu"), budget_bytes=SAMPLES * FRAME_BYTES)
    got = list(source.frames(["cam0/700", "cam0/30", "cam0/500"]))
    assert [name for name, _ in got] == ["cam0/700", "cam0/30", "cam0/500"]
    assert [int(rgb[0, 0, 0]) for _, rgb in got] == [70, 3, 50]
    assert opened[0].requested == [3, 50, 70]


def test_an_evicted_frame_is_decoded_again_from_the_video(opened: list[_FakeVideo]) -> None:
    """The store never holds more than its budget; asking for an evicted frame decodes it again, same pixels."""
    source = VideoStreamFrames(_stream_manifest(), device=torch.device("cpu"), budget_bytes=2 * FRAME_BYTES)
    names = [f"cam0/{10 * k}" for k in range(4)]
    assert [int(rgb[0, 0, 0]) for _, rgb in source.frames(names)] == [0, 1, 2, 3]
    assert source.stored_bytes <= 2 * FRAME_BYTES
    assert [int(rgb[0, 0, 0]) for _, rgb in source.frames(names[:1])] == [0]
    assert source.decoded_frames == 5


def test_a_window_whose_fresh_frames_evict_its_own_held_frames_still_yields_them(opened: list[_FakeVideo]) -> None:
    """Storing the window's decoded frames can evict the frames it already held; those still come back, unchanged."""
    source = VideoStreamFrames(_stream_manifest(), device=torch.device("cpu"), budget_bytes=2 * FRAME_BYTES)
    list(source.frames(["cam0/0", "cam0/10"]))
    got = list(source.frames(["cam0/0", "cam0/20", "cam0/30"]))
    assert [(name, int(rgb[0, 0, 0])) for name, rgb in got] == [("cam0/0", 0), ("cam0/20", 2), ("cam0/30", 3)]
    assert source.decoded_frames == 4
    assert source.stored_bytes <= 2 * FRAME_BYTES


def test_close_releases_every_frame_and_decoder(opened: list[_FakeVideo]) -> None:
    """After `close` nothing is held; a later read starts from scratch with a fresh decoder."""
    source = VideoStreamFrames(_stream_manifest(), device=torch.device("cpu"), budget_bytes=SAMPLES * FRAME_BYTES)
    list(source.frames(["cam0/0", "cam0/10"]))
    source.close()
    assert not source.store and source.stored_bytes == 0 and not source.decoders
    list(source.frames(["cam0/0"]))
    assert len(opened) == 2


def test_each_input_directory_is_read_fresh(tmp_path: Path, opened: list[_FakeVideo]) -> None:
    """No process-wide cache: a manifest rewritten at the same path, and a second input, are what a new source reads."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    write_manifest(first / FRAME_SOURCE_FILE, _stream_manifest("cam0"))
    write_manifest(second / FRAME_SOURCE_FILE, _stream_manifest("cam1"))
    source_a = open_frame_source(first, torch.device("cpu"))
    source_b = open_frame_source(second, torch.device("cpu"))
    assert isinstance(source_a, VideoStreamFrames) and isinstance(source_b, VideoStreamFrames)
    assert source_a.manifest.streams[0].names[0] == "cam0/0" and source_b.manifest.streams[0].names[0] == "cam1/0"
    write_manifest(first / FRAME_SOURCE_FILE, _stream_manifest("cam2"))
    rewritten = open_frame_source(first, torch.device("cpu"))
    assert isinstance(rewritten, VideoStreamFrames) and rewritten.manifest.streams[0].names[0] == "cam2/0"


def test_the_folder_reads_a_bounded_window_ahead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ThreadPoolExecutor.map` submits the whole request at once; the folder keeps `READ_AHEAD` files in flight."""
    (tmp_path / "cam").mkdir()
    names = [f"cam/{k}.png" for k in range(40)]
    for name in names:
        cv2.imwrite(str(tmp_path / name), np.zeros((4, 4, 3), dtype=np.uint8))
    submitted: list[int] = []
    original = frames_module.ThreadPoolExecutor.submit

    def counting_submit(self, fn, /, *args, **kwargs):  # noqa: ANN001, ANN202
        submitted.append(1)
        return original(self, fn, *args, **kwargs)

    monkeypatch.setattr(frames_module.ThreadPoolExecutor, "submit", counting_submit)
    stream = ImageFolderFrames(tmp_path, torch.device("cpu")).frames(names)
    next(stream)
    assert len(submitted) == frames_module.READ_AHEAD
    assert len(list(stream)) == len(names) - 1
