"""``TimedNvdecDecoder`` timestamps and its pre-first-sample gap, without a catalog or a GPU."""

import numpy as np
import pyarrow as pa
import pytest
import torch
from jaxtyping import UInt8
from torch import Tensor

pytest.importorskip("simplecv.rerun_dataloader", reason="needs torchcodec and rerun-sdk's dataloader extra")

from simplecv.rerun_dataloader import TimedFrame, TimedNvdecDecoder  # noqa: E402

EMPTY_COLUMN: pa.ChunkedArray = pa.chunked_array([], type=pa.binary())
"""The point read the decoder ignores; frames come from its segment-wide cache."""


class _StubFrame:
    """One decoded frame as TorchCodec returns it."""

    def __init__(self, data: UInt8[Tensor, "3 h w"]) -> None:
        self.data = data


class _StubVideoDecoder:
    """Segment-wide decoder stand-in whose frames carry their own sample index as pixels."""

    def __init__(self) -> None:
        self.requested: list[int] = []
        """Sample indices asked for so far, in call order."""

    def get_frame_at(self, index: int) -> _StubFrame:
        """Return a 2x2 frame filled with *index*."""
        self.requested.append(index)
        return _StubFrame(torch.full((3, 2, 2), index, dtype=torch.uint8))


class _StubTimedNvdecDecoder(TimedNvdecDecoder):
    """A ``TimedNvdecDecoder`` already bound to a fake segment: its own ``__init__`` needs a live catalog dataset."""

    def __init__(self, times_ns: list[int], segment_id: str, video: _StubVideoDecoder) -> None:
        self._segment_id = segment_id
        self.times = np.array(times_ns, dtype="timedelta64[ns]")
        self._decoder = video  # pyrefly: ignore  # bad-assignment — a decoder stub, not a torchcodec VideoDecoder


def test_timed_frame_carries_the_grid_timestamp_and_the_latest_sample() -> None:
    """A grid slot between samples answers with the previous sample and the slot's own timestamp."""
    video = _StubVideoDecoder()
    decoder = _StubTimedNvdecDecoder(times_ns=[0, 100, 200], segment_id="segment", video=video)
    timed: TimedFrame | None = decoder.decode(EMPTY_COLUMN, np.timedelta64(150, "ns"), "segment")
    assert isinstance(timed, TimedFrame)
    assert timed.t_ns == 150
    assert int(timed.rgb[0, 0, 0]) == 1
    assert video.requested == [1]


def test_timed_frame_is_none_before_the_first_sample() -> None:
    """Grid slots before the segment's first video packet have no frame to answer with."""
    video = _StubVideoDecoder()
    decoder = _StubTimedNvdecDecoder(times_ns=[100, 200], segment_id="segment", video=video)
    assert decoder.decode(EMPTY_COLUMN, np.timedelta64(50, "ns"), "segment") is None
    assert video.requested == []
