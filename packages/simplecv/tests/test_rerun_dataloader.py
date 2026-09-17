"""Behavioral tests for the catalog video decoder."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import torch
from jaxtyping import UInt8
from rerun.catalog import DatasetEntry
from rerun.experimental.dataloader import DecodeRequest, FieldBatch
from torch import Tensor

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
            frame_chw: UInt8[Tensor, "3 1 1"] = torch.full((3, 1, 1), self._offset + index, dtype=torch.uint8)
            return SimpleNamespace(data=frame_chw)

        def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
            """Return the requested frames using torchcodec's batch result shape."""
            frames_nchw: UInt8[Tensor, "n 3 1 1"] = torch.stack([self.get_frame_at(index).data for index in indices])
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


def test_relay_video_stream_opens_at_the_keyframe_before_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decoder cannot start mid-GOP, so the relay backs up to the previous keyframe."""
    times: np.ndarray = np.arange(0, 8, dtype=np.int64).astype("timedelta64[ns]")
    samples: list[bytes] = [bytes([index]) for index in range(8)]
    keyframes: list[bool] = [index % 4 == 0 for index in range(8)]

    def read_packets(
        _dataset: DatasetEntry, _segment_id: str, _entity: str, _timeline: str
    ) -> tuple[np.ndarray, list[bytes], list[bool], int]:
        """Stand in for the catalog reader query."""
        return times, samples, keyframes, rr.VideoCodec.H264.value

    logged: list[tuple[str, object, dict[str, object]]] = []
    timed: list[np.timedelta64] = []
    monkeypatch.setattr(rerun_dataloader, "read_segment_packets", read_packets)
    monkeypatch.setattr(rr, "log", lambda entity_path, archetype, **kwargs: logged.append((entity_path, archetype, kwargs)))
    monkeypatch.setattr(rr, "set_time", lambda _timeline, *, duration: timed.append(duration))
    dataset: DatasetEntry = object.__new__(DatasetEntry)

    relayed: int = rerun_dataloader.relay_video_stream(dataset, "segment-a", "world/cam/video", "video_time", 6, 7)

    assert relayed == 4, "the window [6, 7] opens at the keyframe at t=4"
    assert timed == [np.timedelta64(t_ns, "ns") for t_ns in (4, 5, 6, 7)]
    entity_paths, archetypes, keywords = zip(*logged, strict=True)
    assert entity_paths == ("world/cam/video",) * 5
    assert [kwargs.get("static", False) for kwargs in keywords] == [True, False, False, False, False]
    codec = {batch.component_descriptor().component: batch for batch in archetypes[0].as_component_batches()}
    assert codec["VideoStream:codec"].as_arrow_array().to_pylist() == [rr.VideoCodec.H264.value]
    relayed_samples = [
        {batch.component_descriptor().component: batch for batch in archetype.as_component_batches()} for archetype in archetypes[1:]
    ]
    assert [bytes(fields["VideoStream:sample"].as_arrow_array().to_pylist()[0]) for fields in relayed_samples] == samples[4:8]
    keyframe_flags = [fields["VideoStream:is_keyframe"].as_arrow_array().to_pylist() for fields in relayed_samples]
    assert keyframe_flags == [[True], [False], [False], [False]], "only the anchor packet is a keyframe"
