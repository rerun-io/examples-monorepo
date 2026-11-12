"""``open_decoder``: a GPU decoder that warns when torchcodec decodes on the CPU (``cpu_fallback``)."""

from pathlib import Path

import pytest

pytest.importorskip("torchcodec", reason="open_decoder builds torchcodec decoders")

import torchcodec.decoders

from simplecv import video_io
from simplecv.video_io import open_decoder


@pytest.mark.parametrize("fallback", ["", "[CUDA] Fallback status: Falling back due to: NVcuvid unavailable"])
def test_open_decoder_warns_only_when_torchcodec_decodes_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fallback: str
) -> None:
    built: list[str] = []

    class Decoder:
        def __init__(self, video: Path, device: str = "cpu", **_: str) -> None:
            built.append(device)
            self.cpu_fallback: str = ""  # torchcodec knows the status only after a frame

        def get_frame_at(self, index: int) -> None:
            self.cpu_fallback = fallback

    monkeypatch.setattr(torchcodec.decoders, "VideoDecoder", Decoder)
    monkeypatch.setattr(video_io, "bind_cuda_device", lambda device: None)
    with caplog.at_level("WARNING"):
        decoder = open_decoder(Path("clip.mp4"), "cuda")
    assert isinstance(decoder, Decoder) and built == ["cuda"]  # torchcodec falls back by itself: no second decoder
    if fallback:
        assert [record.levelname for record in caplog.records] == ["WARNING"]
        assert "clip.mp4" in caplog.text and fallback in caplog.text
    else:
        assert not caplog.records
