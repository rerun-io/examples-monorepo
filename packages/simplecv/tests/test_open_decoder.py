"""``open_decoder``: NVDEC first for every video, the CPU with a warning when it fails, and only host faults are remembered."""

from pathlib import Path

import pytest

pytest.importorskip("torchcodec", reason="open_decoder builds torchcodec decoders")

import torchcodec.decoders

from simplecv import video_io
from simplecv.video_io import open_decoder

HOST_FAULT: str = "[CUDA] Fallback status: Falling back due to: NVcuvid unavailable"
VIDEO_FAULT: str = "[CUDA] Fallback status: Falling back due to: Video not supported"


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Events of a fake torchcodec: ``bind`` per CUDA bind, then ``<video>:<device>`` per decoder opened; no hardware needed.

    The video's name says how NVDEC fails on it: ``raises-*`` at construction, ``frame-*`` on its first frame,
    ``silent-*`` not at all (torchcodec decodes on the CPU and says so in ``cpu_fallback``); ``host`` in the name
    makes the cause a missing NVDEC library. ``unreadable`` fails on the CPU too.
    """
    events: list[str] = []

    class Decoder:
        def __init__(self, video: Path, device: str = "cpu", **_: str) -> None:
            events.append(f"{video.stem}:{device}")
            self.gpu: bool = device != "cpu"
            self.name: str = video.stem
            self.cause: str = HOST_FAULT if "host" in self.name else VIDEO_FAULT
            if self.gpu and self.name.startswith("raises"):
                raise RuntimeError(self.cause)
            self.cpu_fallback: str = self.cause if self.gpu and self.name.startswith("silent") else ""

        def get_frames_in_range(self, start: int, stop: int) -> None:
            if (self.gpu and self.name.startswith("frame")) or self.name == "unreadable":
                raise RuntimeError(self.cause)

    monkeypatch.setattr(torchcodec.decoders, "VideoDecoder", Decoder)
    monkeypatch.setattr(video_io, "bind_cuda_device", lambda device: events.append("bind"))
    monkeypatch.setattr(video_io, "HARDWARE_VIDEO_FAILED", {})
    return events


def test_the_cuda_device_is_bound_before_the_decoder_is_built(opened: list[str]) -> None:
    open_decoder(Path("clip.mp4"), "cuda")
    assert opened == ["bind", "clip:cuda"]


@pytest.mark.parametrize("odd", ["raises-odd", "frame-odd", "silent-odd"])
def test_a_video_nvdec_cannot_take_goes_to_the_cpu_and_the_next_video_gets_nvdec_again(
    opened: list[str], caplog: pytest.LogCaptureFixture, odd: str
) -> None:
    # frame-odd: the first video worked, the second builds on the GPU and fails on its first GPU frame.
    with caplog.at_level("WARNING"):
        for name in ("clip", odd, "clip"):
            open_decoder(Path(f"{name}.mp4"), "cuda")
    assert [event for event in opened if event != "bind"] == ["clip:cuda", f"{odd}:cuda", f"{odd}:cpu", "clip:cuda"]
    assert video_io.HARDWARE_VIDEO_FAILED == {} and video_io.hardware_video_warning() == ""
    assert [record.levelname for record in caplog.records] == ["WARNING"] and f"{odd}.mp4" in caplog.text


@pytest.mark.parametrize("fault", ["raises-host", "silent-host"])
def test_a_host_without_nvdec_is_remembered_and_warned_about_once(opened: list[str], caplog: pytest.LogCaptureFixture, fault: str) -> None:
    # No libnvcuvid: torchcodec raises, or decodes on the CPU without a word (cpu_fallback says so).
    with caplog.at_level("WARNING"):
        open_decoder(Path(f"{fault}.mp4"), "cuda")
        open_decoder(Path("clip.mp4"), "cuda")
    assert [event for event in opened if event != "bind"] == [f"{fault}:cuda", f"{fault}:cpu", "clip:cpu"]  # NVDEC once, then the CPU
    assert [record.levelname for record in caplog.records] == ["WARNING"] and "NVcuvid unavailable" in caplog.text
    warning: str = video_io.hardware_video_warning()
    assert warning.startswith("WARNING: hardware video failed") and "decode ([CUDA] Fallback status" in warning


def test_a_video_nothing_can_read_raises_and_blames_no_hardware(opened: list[str]) -> None:
    with pytest.raises(RuntimeError):
        open_decoder(Path("unreadable.mp4"), "cuda")
    assert video_io.HARDWARE_VIDEO_FAILED == {}
