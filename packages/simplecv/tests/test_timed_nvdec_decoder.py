"""``TimedNvdecDecoder`` timestamps and its pre-first-sample gap, without a catalog or a GPU."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from jaxtyping import UInt8
from torch import Tensor

pytest.importorskip("simplecv.rerun_dataloader", reason="needs torchcodec and rerun-sdk's dataloader extra")

from rerun.catalog import DatasetEntry  # noqa: E402
from rerun.experimental.dataloader import DecodeRequest, FieldBatch  # noqa: E402

import simplecv.rerun_dataloader as rerun_dataloader  # noqa: E402
from simplecv.rerun_dataloader import TimedFrame, TimedNvdecDecoder  # noqa: E402

SEGMENT: str = "segment"
"""The one segment every request in these tests belongs to."""


class _StubVideoDecoder:
    """Segment-wide decoder stand-in whose frames carry their own sample index as pixels."""

    def __init__(self) -> None:
        self.requested: list[int] = []
        """Sample indices asked for so far, in call order."""

    def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
        """Return one 2x2 frame per index, filled with that index, in torchcodec's batch result shape."""
        self.requested.extend(indices)
        frames_nchw: UInt8[Tensor, "n 3 2 2"] = torch.stack([torch.full((3, 2, 2), index, dtype=torch.uint8) for index in indices])
        return SimpleNamespace(data=frames_nchw)


def _bound_decoder(monkeypatch: pytest.MonkeyPatch, times_ns: list[int]) -> tuple[TimedNvdecDecoder, _StubVideoDecoder]:
    """Build a decoder whose one segment holds samples at *times_ns*, with no catalog and no torchcodec."""
    video = _StubVideoDecoder()

    def open_decoder(
        _dataset: DatasetEntry,
        _segment_id: str,
        _entity: str,
        _timeline: str,
        _device: torch.device,
        _fps: int,
    ) -> tuple[np.ndarray, list[bytes], list[bool], _StubVideoDecoder]:
        """Stand in for the catalog and torchcodec boundary."""
        return np.asarray(times_ns, dtype=np.int64).astype("timedelta64[ns]"), [], [], video

    monkeypatch.setattr(rerun_dataloader, "open_segment_decoder", open_decoder)
    dataset: DatasetEntry = object.__new__(DatasetEntry)
    return TimedNvdecDecoder(dataset, "video/wide", "video_time", torch.device("cpu"), 30), video


def _requests(*index_ns: int) -> list[DecodeRequest]:
    """One grid slot per timestamp, all point reads on the same segment."""
    return [DecodeRequest(row, SEGMENT, np.timedelta64(value, "ns"), (row,), (row,), True) for row, value in enumerate(index_ns)]


def _batch(count: int) -> FieldBatch:
    """The fetched point reads the decoder ignores; frames come from its segment-wide cache."""
    return FieldBatch(column=pa.array([[b"ignored"]] * count))


def test_timed_frame_carries_the_grid_timestamp_and_the_latest_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """A grid slot between samples answers with the previous sample and the slot's own timestamp."""
    decoder, video = _bound_decoder(monkeypatch, times_ns=[0, 100, 200])

    decoded: list[TimedFrame | None] = decoder.decode(_batch(1), _requests(150))

    timed: TimedFrame | None = decoded[0]
    assert isinstance(timed, TimedFrame)
    assert timed.t_ns == 150
    assert int(timed.rgb[0, 0, 0]) == 1
    assert video.requested == [1]


def test_timed_frame_is_none_before_the_first_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grid slots before the segment's first video packet have no frame to answer with."""
    decoder, video = _bound_decoder(monkeypatch, times_ns=[100, 200])

    decoded: list[TimedFrame | None] = decoder.decode(_batch(2), _requests(50, 150))

    assert decoded[0] is None
    timed: TimedFrame | None = decoded[1]
    assert isinstance(timed, TimedFrame)
    assert timed.t_ns == 150
    assert video.requested == [0]
