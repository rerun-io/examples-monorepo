"""The Rerun dataloader's NVDEC column decoder, over the chunked catalog reader."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import torch
from jaxtyping import UInt8

pytestmark = pytest.mark.integration
pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")
pytest.importorskip("rerun.experimental.dataloader", reason="rerun.experimental.dataloader is required by this test module")
pytest.importorskip("torchcodec", reason="torchcodec is required by this test module")

from rerun.catalog import DatasetEntry  # noqa: E402
from rerun.experimental.dataloader import DecodeRequest, FieldBatch  # noqa: E402
from torch import Tensor  # noqa: E402
from video_fixtures import encode_h264  # noqa: E402

import simplecv.rerun_dataloader as rerun_dataloader  # noqa: E402
from simplecv.catalog_video_codec import wrap_mp4  # noqa: E402
from simplecv.catalog_video_reader import ChunkedVideoDecoder, select_decode_device  # noqa: E402
from simplecv.rerun_dataloader import SegmentNvdecDecoder  # noqa: E402


def test_segment_nvdec_decoder_decodes_a_fetch_block_across_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """One aligned result per request; each segment's decoder is opened once."""

    class _FakeChunkedDecoder(ChunkedVideoDecoder):
        def __init__(self, offset: int, times_ns: list[int]) -> None:
            self._offset: int = offset
            self._index = SimpleNamespace(times=np.asarray(times_ns, dtype=np.int64).astype("timedelta64[ns]"))

        @property
        def index(self) -> SimpleNamespace:  # pyrefly: ignore[bad-override]
            return self._index

        def get_frames_at(self, indices: list[int]) -> Tensor:
            frames_nchw: UInt8[Tensor, "n 3 1 1"] = torch.stack([torch.full((3, 1, 1), self._offset + index, dtype=torch.uint8) for index in indices])
            return frames_nchw

    opened_segments: list[str] = []

    def open_decoders(_dataset: DatasetEntry, segment_id: str, _entities: list[str], _timeline: str, _device: torch.device) -> tuple[_FakeChunkedDecoder]:
        opened_segments.append(segment_id)
        return (_FakeChunkedDecoder(0, [10, 20]),) if segment_id == "segment-a" else (_FakeChunkedDecoder(10, [5, 15]),)

    monkeypatch.setattr(rerun_dataloader, "open_decoders", open_decoders)
    decoder: SegmentNvdecDecoder = SegmentNvdecDecoder(object.__new__(DatasetEntry), "video/wide", "video_time", torch.device("cpu"))
    requests: list[DecodeRequest] = [
        DecodeRequest(0, "segment-a", np.timedelta64(20, "ns"), (0,), (0,), True),
        DecodeRequest(1, "segment-b", np.timedelta64(4, "ns"), (1,), (1,), True),
        DecodeRequest(2, "segment-b", np.timedelta64(15, "ns"), (2,), (2,), True),
    ]

    decoded = decoder.decode(FieldBatch(column=pa.array([[b"ignored"]] * 3)), requests)

    assert opened_segments == ["segment-a", "segment-b"]
    assert decoded[1] is None
    assert decoded[0] is not None and int(decoded[0][0, 0, 0]) == 1
    assert decoded[2] is not None and int(decoded[2][0, 0, 0]) == 11


def test_the_segment_decoder_decodes_on_nvdec_from_a_real_catalog(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the NVDEC path cannot be checked here")
    samples: list[tuple[bytes, bool]] = encode_h264(80, 8, seed=5)
    times = (33_333_333 * np.arange(len(samples), dtype=np.int64)).astype("timedelta64[ns]")
    path: Path = tmp_path / "video.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        rec.log("/cam/video", rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        rec.send_columns(
            "/cam/video",
            indexes=[rr.TimeColumn("video_time", duration=times)],
            columns=rr.VideoStream.columns(sample=[s for s, _ in samples], is_keyframe=[k for _, k in samples]),
        )
    from torchcodec.decoders import VideoDecoder

    whole: Tensor = VideoDecoder(wrap_mp4([s for s, _ in samples], [k for _, k in samples], 30, codec="h264"), device="cuda").get_frames_at(
        list(range(len(samples)))
    ).data
    with rr.server.Server(datasets={"test": [path]}) as server:
        decoder = SegmentNvdecDecoder(server.client().get_dataset("test"), "cam/video", "video_time", torch.device(select_decode_device()))
        requests: list[DecodeRequest] = [DecodeRequest(row, "segment", times[index], (row,), (row,), True) for row, index in enumerate((0, 7, 8, 41, 79))]
        decoded = decoder.decode(FieldBatch(column=pa.array([[b"ignored"]] * len(requests))), requests)
        single: Tensor | None = decoder.decode_at(times[50], "segment")
    for frame, index in zip(decoded, (0, 7, 8, 41, 79), strict=True):
        assert frame is not None and frame.device.type == "cuda"
        assert torch.equal(frame, whole[index])
    assert single is not None and single.device.type == "cuda" and torch.equal(single, whole[50])
