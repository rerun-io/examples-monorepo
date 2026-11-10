"""``SegmentNvdecDecoder`` slicing and on-device transform, and ``KeyedNvdecDecoder``'s keys, without a catalog or a GPU."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from jaxtyping import UInt8
from torch import Tensor

pytestmark = pytest.mark.integration
pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")
pytest.importorskip("rerun.experimental.dataloader", reason="rerun.experimental.dataloader is required by this test module")
pytest.importorskip("torchcodec", reason="torchcodec is required by this test module")

from rerun.catalog import DatasetEntry  # noqa: E402
from rerun.experimental.dataloader import DecodeRequest, FieldBatch  # noqa: E402

import simplecv.rerun_dataloader as rerun_dataloader  # noqa: E402
from simplecv.rerun_dataloader import KeyedFrame, KeyedNvdecDecoder, SegmentNvdecDecoder  # noqa: E402


class _StubVideoDecoder:
    """Frames carry their own sample index as pixels; records each call's size."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
        self.calls.append(len(indices))
        frames_nchw: UInt8[Tensor, "n 3 2 2"] = torch.stack([torch.full((3, 2, 2), index, dtype=torch.uint8) for index in indices])
        return SimpleNamespace(data=frames_nchw)


def _bind(monkeypatch: pytest.MonkeyPatch, frames: int) -> _StubVideoDecoder:
    video = _StubVideoDecoder()

    def open_decoder(_dataset, _segment_id, _entity, _timeline, _device, _fps):
        return np.arange(frames, dtype=np.int64), [], [], video

    monkeypatch.setattr(rerun_dataloader, "open_segment_decoder", open_decoder)
    return video


def _requests(segment: str, indices: list[int]) -> list[DecodeRequest]:
    return [DecodeRequest(row, segment, np.int64(index), (row,), (row,), True) for row, index in enumerate(indices)]


BATCH: FieldBatch = FieldBatch(column=pa.array([[b"ignored"]] * 5))


def test_a_long_block_decodes_in_bounded_slices_through_the_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    video = _bind(monkeypatch, 10)
    decoder = SegmentNvdecDecoder(object.__new__(DatasetEntry), "cam/video", "frame_index", torch.device("cpu"), 30,
                                  max_frames_per_call=2, transform=lambda frames: frames[:, 0] * 2)
    decoded = decoder.decode(BATCH, _requests("s", [3, 4, 5, 6, 7]))
    assert video.calls == [2, 2, 1]
    assert [int(frame[0, 0]) for frame in decoded if frame is not None] == [6, 8, 10, 12, 14]
    assert all(frame is not None and frame.shape == (2, 2) for frame in decoded)


def test_keyed_frames_name_their_segment_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind(monkeypatch, 10)
    decoder = KeyedNvdecDecoder(object.__new__(DatasetEntry), "cam/video", "frame_index", torch.device("cpu"), 30)
    decoded = decoder.decode(BATCH, _requests("episode-7", [0, 9]))
    assert [(frame.segment_id, frame.index) for frame in decoded if isinstance(frame, KeyedFrame)] == [("episode-7", 0), ("episode-7", 9)]
    assert int(decoded[1].image[0, 0, 0]) == 9
