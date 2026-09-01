"""Behavioral tests for the catalog video decoder."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from rerun.catalog import DatasetEntry
from rerun.experimental.dataloader import DecodeRequest, FieldBatch

import simplecv.rerun_dataloader as rerun_dataloader
from simplecv.rerun_dataloader import SegmentNvdecDecoder


def test_segment_nvdec_decoder_decodes_a_fetch_block_across_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return one aligned result per request while reusing each segment decoder."""

    class _FakeVideoDecoder:
        """Return a segment-specific scalar in a frame-shaped tensor."""

        def __init__(self, offset: int) -> None:
            self._offset: int = offset

        def get_frame_at(self, index: int) -> SimpleNamespace:
            """Return the requested frame using torchcodec's result shape."""
            frame_chw: torch.Tensor = torch.full((3, 1, 1), self._offset + index, dtype=torch.uint8)
            return SimpleNamespace(data=frame_chw)

        def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
            """Return the requested frames using torchcodec's batch result shape."""
            frames_nchw: torch.Tensor = torch.stack([self.get_frame_at(index).data for index in indices])
            return SimpleNamespace(data=frames_nchw)

    opened_segments: list[str] = []

    def open_decoder(
        _dataset: DatasetEntry,
        segment_id: str,
        _entity: str,
        _timeline: str,
        _device: torch.device,
        _fps: int,
    ) -> tuple[np.ndarray, list[bytes], list[bool], _FakeVideoDecoder]:
        """Stand in for the catalog and torchcodec boundary."""
        opened_segments.append(segment_id)
        if segment_id == "segment-a":
            return np.asarray([10, 20], dtype=np.int64).astype("timedelta64[ns]"), [], [], _FakeVideoDecoder(0)
        return np.asarray([5, 15], dtype=np.int64).astype("timedelta64[ns]"), [], [], _FakeVideoDecoder(10)

    monkeypatch.setattr(rerun_dataloader, "open_segment_decoder", open_decoder)
    dataset: DatasetEntry = object.__new__(DatasetEntry)
    decoder: SegmentNvdecDecoder = SegmentNvdecDecoder(dataset, "video/wide", "video_time", torch.device("cpu"), 60)
    batch: FieldBatch = FieldBatch(column=pa.array([[b"ignored"], [b"ignored"], [b"ignored"]]))
    requests: list[DecodeRequest] = [
        DecodeRequest(0, "segment-a", np.timedelta64(20, "ns"), (0,), (0,), True),
        DecodeRequest(1, "segment-b", np.timedelta64(4, "ns"), (1,), (1,), True),
        DecodeRequest(2, "segment-b", np.timedelta64(15, "ns"), (2,), (2,), True),
    ]

    decoded = decoder.decode(batch, requests)

    assert opened_segments == ["segment-a", "segment-b"]
    assert decoded[1] is None
    assert decoded[0] is not None
    assert decoded[2] is not None
    assert int(decoded[0][0, 0, 0]) == 1
    assert int(decoded[2][0, 0, 0]) == 11
