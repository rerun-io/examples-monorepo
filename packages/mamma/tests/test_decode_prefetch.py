"""The streaming loop's chunk decode: one persistent per-camera pool, one chunk ahead, backend reported.

The unit tests swap torchcodec's ``VideoDecoder`` for fakes; the integration test decodes a
real H.264 clip on NVDEC across several chunks (the decode that crashed at chunk 2 before).
"""

from __future__ import annotations

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
from simplecv.video_io import TorchCodecMultiVideoReader

from mamma.engine.pipeline import decode_backend, iter_decoded_chunks


class _Metadata:
    width: int = 4
    height: int = 2
    num_frames: int = 10
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
        return "Falling back due to: Video not supported" if self.fell_back else "No fallback required"


def _frames(indices: Sequence[int]) -> UInt8[torch.Tensor, "b 3 h w"]:
    """Frames whose every pixel holds the frame index."""
    values: torch.Tensor = torch.tensor(list(indices), dtype=torch.uint8)[:, None, None, None]
    return values.expand(-1, 3, _Metadata.height, _Metadata.width).contiguous()


class _OwnedDecoder:
    """Fails unless every decode runs on the thread that created it (NVDEC thread affinity)."""

    fell_back: bool = False

    def __init__(self, _source: Path, **_kwargs: object) -> None:
        self.metadata: _Metadata = _Metadata()
        self.owner_thread: int = threading.get_ident()
        self.cpu_fallback: _FallbackStatus = _FallbackStatus(self.fell_back)

    def get_frames_in_range(self, start: int, stop: int) -> _FrameBatch:
        assert threading.get_ident() == self.owner_thread, "decoder advanced from a thread that does not own it"
        return _FrameBatch(_frames(range(start, stop)))


class _FallbackDecoder(_OwnedDecoder):
    fell_back = True


@pytest.fixture
def fake_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", types.SimpleNamespace(VideoDecoder=_OwnedDecoder))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _index: None)


@pytest.mark.usefixtures("fake_decoder")
def test_chunks_follow_the_ranges_in_order() -> None:
    """Chunks start at the run's first frame (not 0) and every camera yields the same frames."""
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader([Path("a.mp4"), Path("b.mp4")], device="cpu", thread_owned=True)
    ranges: list[tuple[int, int]] = [(3, 6), (6, 9), (9, 10)]
    try:
        with ThreadPoolExecutor(max_workers=2) as cam_pool:
            chunks: list[list[UInt8[torch.Tensor, "b 3 h w"]]] = list(iter_decoded_chunks(reader, ranges, cam_pool))
    finally:
        reader.close()
    assert [[video[:, 0, 0, 0].tolist() for video in chunk] for chunk in chunks] == [
        [[3, 4, 5]] * 2,
        [[6, 7, 8]] * 2,
        [[9]] * 2,
    ]


@pytest.mark.usefixtures("fake_decoder")
def test_decode_runs_on_the_given_pool_one_chunk_ahead() -> None:
    """Every chunk is decoded by the caller's persistent pool, and chunk i+1 is in flight while chunk i is consumed."""
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader([Path("a.mp4"), Path("b.mp4")], device="cpu", thread_owned=True)
    submitted: list[int] = []

    class _RecordingPool(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            submitted.append(int(args[0]))
            return super().submit(fn, *args, **kwargs)

    consumed_before_next: list[list[int]] = []
    try:
        with _RecordingPool(max_workers=2) as cam_pool:
            for _chunk in iter_decoded_chunks(reader, [(0, 4), (4, 8), (8, 10)], cam_pool):
                consumed_before_next.append(list(submitted))
    finally:
        reader.close()
    # One submit per camera per chunk; the next chunk is queued before the current one is handed out.
    assert consumed_before_next == [[0, 0, 4, 4], [0, 0, 4, 4, 8, 8], [0, 0, 4, 4, 8, 8]]


@pytest.mark.usefixtures("fake_decoder")
def test_empty_range_list_yields_nothing() -> None:
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader([Path("a.mp4")], device="cpu", thread_owned=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as cam_pool:
            assert list(iter_decoded_chunks(reader, [], cam_pool)) == []
    finally:
        reader.close()


@pytest.mark.usefixtures("fake_decoder")
def test_decode_backend_names_nvdec() -> None:
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader([Path("a.mp4"), Path("b.mp4")], device="cuda", thread_owned=True)
    try:
        assert decode_backend(reader) == "NVDEC"
    finally:
        reader.close()


def test_decode_backend_names_the_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4:4:4 input that NVDEC rejects is visibly on the slow path in the run summary."""
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", types.SimpleNamespace(VideoDecoder=_FallbackDecoder))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _index: None)
    with pytest.warns(RuntimeWarning):
        reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader([Path("a.mp4"), Path("b.mp4")], device="cuda", thread_owned=True)
    try:
        assert decode_backend(reader) == "torchcodec CPU fallback (NVDEC rejected 2/2 cameras; slow path)"
    finally:
        reader.close()


@pytest.mark.usefixtures("fake_decoder")
def test_decode_backend_names_cpu_decode() -> None:
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader([Path("a.mp4")], device="cpu")
    assert decode_backend(reader) == "torchcodec CPU"


@pytest.mark.integration
def test_nvdec_chunks_survive_the_prefetch_pool(tmp_path: Path) -> None:
    """Real NVDEC over several chunks: before the fix the second chunk failed with 'Could not receive frame from decoder'."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (NVDEC)")
    ffmpeg: str | None = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not on PATH")
    clips: list[Path | bytes] = []
    for name in ("a", "b"):
        clip: Path = tmp_path / f"{name}.mp4"
        subprocess.run(
            [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=30", "-frames:v", "100"]
            + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0", "-g", "30", str(clip)],
            check=True,
        )
        clips.append(clip)
    reference: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(clips, device="cuda", thread_owned=True)
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(clips, device="cuda", thread_owned=True)
    try:
        assert decode_backend(reader) == "NVDEC"
        expected: list[UInt8[torch.Tensor, "b 3 h w"]] = [cam.get_frames_in_range(10, 100) for cam in reference.video_readers]
        ranges: list[tuple[int, int]] = [(s, min(s + 32, 100)) for s in range(10, 100, 32)]
        with ThreadPoolExecutor(max_workers=2) as cam_pool:
            chunks: list[list[UInt8[torch.Tensor, "b 3 h w"]]] = list(iter_decoded_chunks(reader, ranges, cam_pool))
        assert len(chunks) == 3
        for cam_idx in range(2):
            decoded: UInt8[torch.Tensor, "b 3 h w"] = torch.cat([chunk[cam_idx] for chunk in chunks])
            assert torch.equal(decoded, expected[cam_idx])
    finally:
        reader.close()
        reference.close()
