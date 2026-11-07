"""Thread affinity, CPU-fallback reporting and batched access of the TorchCodec readers.

The unit tests swap torchcodec's ``VideoDecoder`` for fakes; the integration tests
decode real H.264 clips that ffmpeg writes into ``tmp_path``.
"""

import shutil
import subprocess
import sys
import threading
import types
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from jaxtyping import UInt8

from simplecv.video_io import TorchCodecMultiVideoReader, TorchCodecVideoReader

_bound_device: threading.local = threading.local()
"""The CUDA device index the fake ``torch.cuda.set_device`` bound on each thread."""


class _Metadata:
    width: int = 4
    height: int = 2
    num_frames: int = 6
    average_fps: float = 30.0


class _FrameBatch:
    def __init__(self, data: torch.Tensor) -> None:
        self.data: torch.Tensor = data


class _FallbackStatus:
    def __init__(self, fell_back: bool) -> None:
        self.fell_back: bool = fell_back

    def __bool__(self) -> bool:
        return self.fell_back

    def __str__(self) -> str:
        return "[CUDA] Fallback status: Falling back due to: Video not supported" if self.fell_back else "No fallback required"


def _frames(indices: Sequence[int]) -> UInt8[torch.Tensor, "b 3 h w"]:
    """Frames whose every pixel holds the frame index."""
    values: torch.Tensor = torch.tensor(list(indices), dtype=torch.uint8)[:, None, None, None]
    return values.expand(-1, 3, _Metadata.height, _Metadata.width).contiguous()


class _FakeCudaDecoder:
    """torchcodec's CUDA decoder fails on a thread that has no CUDA device bound; so does this fake."""

    fell_back: bool = False

    def __init__(self, _source: Path, **_kwargs: object) -> None:
        self._check()
        self.metadata: _Metadata = _Metadata()
        self.cpu_fallback: _FallbackStatus = _FallbackStatus(self.fell_back)
        self.decode_threads: list[int] = []

    def _check(self) -> None:
        assert getattr(_bound_device, "index", None) == 0, "decode on a thread without a bound CUDA device"

    def get_frame_at(self, frame_id: int) -> _FrameBatch:
        self._check()
        self.decode_threads.append(threading.get_ident())
        return _FrameBatch(_frames([frame_id])[0])

    def get_frames_in_range(self, start: int, stop: int) -> _FrameBatch:
        self._check()
        return _FrameBatch(_frames(range(start, stop)))


class _FakeFallbackDecoder(_FakeCudaDecoder):
    fell_back = True


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake CUDA decoders plus a ``torch.cuda.set_device`` that records the binding per thread."""
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", types.SimpleNamespace(VideoDecoder=_FakeCudaDecoder))
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: setattr(_bound_device, "index", index))


def test_reader_driven_from_new_threads_binds_cuda_device(fake_cuda: None) -> None:
    """mamma's decode_chunk advances a reader from a new thread per chunk; every decode must still find a CUDA device."""
    reader: TorchCodecVideoReader = TorchCodecVideoReader(Path("cam0.mp4"), device="cuda")
    for start in range(0, 6, 2):
        with ThreadPoolExecutor(max_workers=1) as pool:
            chunk: UInt8[torch.Tensor, "b 3 h w"] = pool.submit(reader.get_frames_in_range, start, start + 2).result()
        assert chunk[:, 0, 0, 0].tolist() == [start, start + 1]


def test_multi_reader_thread_owned_keeps_each_decoder_on_its_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """``thread_owned`` passes through to every camera reader: one owner thread per decoder."""

    class _OwnedDecoder:
        def __init__(self, _source: Path, **_kwargs: object) -> None:
            self.metadata: _Metadata = _Metadata()
            self.owner_thread: int = threading.get_ident()

        def get_frames_in_range(self, start: int, stop: int) -> _FrameBatch:
            assert threading.get_ident() == self.owner_thread
            return _FrameBatch(_frames(range(start, stop)))

    monkeypatch.setitem(sys.modules, "torchcodec.decoders", types.SimpleNamespace(VideoDecoder=_OwnedDecoder))
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader([Path("cam0.mp4"), Path("cam1.mp4")], device="cpu", thread_owned=True)
    try:
        for start in range(0, 6, 3):
            with ThreadPoolExecutor(max_workers=2) as pool:
                chunks: list[UInt8[torch.Tensor, "b 3 h w"]] = list(
                    pool.map(lambda cam, s=start: cam.get_frames_in_range(s, s + 3), reader.video_readers)
                )
            assert [chunk[:, 0, 0, 0].tolist() for chunk in chunks] == [[start, start + 1, start + 2]] * 2
    finally:
        reader.close()


def test_cpu_fallback_is_reported_once(fake_cuda: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream NVDEC rejects (HEVC 4:4:4) decodes on the CPU; the reader says so instead of staying silent."""
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", types.SimpleNamespace(VideoDecoder=_FakeFallbackDecoder))
    with pytest.warns(RuntimeWarning, match="Video not supported"):
        reader: TorchCodecVideoReader = TorchCodecVideoReader(Path("hevc444.mp4"), device="cuda")
    assert reader.cpu_fallback


def test_nvdec_stream_reports_no_fallback(fake_cuda: None) -> None:
    reader: TorchCodecVideoReader = TorchCodecVideoReader(Path("h264.mp4"), device="cuda")
    assert not reader.cpu_fallback


def test_multi_reader_getitem_decodes_cameras_on_the_worker_pool(fake_cuda: None) -> None:
    """``reader[idx]`` decodes the cameras in parallel, like ``get_frames_at``, not one after another on the caller."""
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader([Path("cam0.mp4"), Path("cam1.mp4")], device="cuda")
    try:
        frames: list[UInt8[torch.Tensor, "3 h w"]] = reader[4]
        assert [int(frame[0, 0, 0]) for frame in frames] == [4, 4]
        decode_threads: set[int] = {ident for cam in reader.video_readers for ident in cam._decoder.decode_threads}
        assert threading.get_ident() not in decode_threads
    finally:
        reader.close()


def test_stepped_slice_decodes_in_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``reader[a:b:step]`` asks the decoder for all indices at once instead of one seek per frame."""

    class _BatchOnlyDecoder:
        def __init__(self, _source: Path, **_kwargs: object) -> None:
            self.metadata: _Metadata = _Metadata()

        def get_frame_at(self, frame_id: int) -> _FrameBatch:
            raise AssertionError(f"stepped slice decoded frame {frame_id} on its own")

        def get_frames_at(self, indices: list[int]) -> _FrameBatch:
            return _FrameBatch(_frames(indices))

    monkeypatch.setitem(sys.modules, "torchcodec.decoders", types.SimpleNamespace(VideoDecoder=_BatchOnlyDecoder))
    reader: TorchCodecVideoReader = TorchCodecVideoReader(Path("cam0.mp4"), device="cpu")
    stepped: UInt8[torch.Tensor, "b 3 h w"] = reader[0:6:2]
    assert stepped[:, 0, 0, 0].tolist() == [0, 2, 4]


def _encode_clip(path: Path, pix_fmt: str) -> Path:
    """Write a 96-frame 256x256 H.264 test pattern without B-frames."""
    ffmpeg: str | None = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not on PATH")
    subprocess.run(
        [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=30", "-frames:v", "96"]
        + ["-c:v", "libx264", "-pix_fmt", pix_fmt, "-bf", "0", "-g", "30", str(path)],
        check=True,
    )
    return path


@pytest.mark.integration
def test_nvdec_reader_survives_a_new_thread_per_chunk(tmp_path: Path) -> None:
    """Real NVDEC: before the fix the second chunk failed with 'Could not receive frame from decoder'."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (NVDEC)")
    clip: Path = _encode_clip(tmp_path / "h264.mp4", "yuv420p")
    reference: UInt8[torch.Tensor, "b 3 h w"] = TorchCodecVideoReader(clip, device="cuda").get_frames_in_range(0, 96)
    reader: TorchCodecVideoReader = TorchCodecVideoReader(clip, device="cuda")
    assert not reader.cpu_fallback
    for start in range(0, 96, 32):
        with ThreadPoolExecutor(max_workers=1) as pool:
            chunk: UInt8[torch.Tensor, "b 3 h w"] = pool.submit(reader.get_frames_in_range, start, start + 32).result()
        assert torch.equal(chunk, reference[start : start + 32])


@pytest.mark.integration
def test_yuv444_stream_reports_cpu_fallback(tmp_path: Path) -> None:
    """NVDEC rejects 4:4:4; torchcodec decodes on the CPU and the reader reports it."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (NVDEC)")
    clip: Path = _encode_clip(tmp_path / "h264_444.mp4", "yuv444p")
    with pytest.warns(RuntimeWarning, match="CPU"):
        reader: TorchCodecVideoReader = TorchCodecVideoReader(clip, device="cuda")
    assert reader.cpu_fallback
    assert reader.get_frames_in_range(0, 8).shape == (8, 3, 256, 256)
