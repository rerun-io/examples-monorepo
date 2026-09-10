"""GPU video decoding, and stored-packet relay, for the Rerun catalog dataloader.

Import this module explicitly: it needs ``rerun-sdk``'s ``dataloader`` extra,
which only the catalog lanes install, so it must never be re-exported from
``simplecv/__init__.py``.

``relay_video_stream`` shares the segment-wide packet query with the decoder but
never decodes: a tool that wants the source video in its own recording re-logs
the catalog's samples instead of re-encoding decoded frames.

Upstream's video decoder (rerun-io/reality PR #2893) wraps each sample's
keyframe window into its own in-memory MP4 and is slower than dav1d — no
decode-session reuse across calls. ``SegmentNvdecDecoder`` instead bulk-fetches
a segment's packets once, wraps them into a single MP4 (same ``add_mux_stream``
technique), and serves every sample from one GPU decoder.

TODO(rerun#upstream): delete this module once the dataloader decodes video fast
enough on its own — this is a stopgap, and it costs real flexibility: holding a
``DatasetEntry`` makes it unpicklable (so ``num_workers`` must stay 0), and the
cached CUDA decoder does not survive ``decode_threads > 1``. Upstream's batch
decoder now reuses one codec context across a GOP, but its PyAV CPU path remains
slower than this segment-wide torchcodec/NVDEC path.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from jaxtyping import Int64, Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetView
from rerun.experimental.dataloader import ColumnDecoder, DecodeRequest, FieldBatch
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from simplecv.catalog_video_codec import catalog_codec_name, wrap_mp4

TimedeltaNs: TypeAlias = Shaped[ndarray, " n_samples"]
"""Sample timestamps in timeline order, dtype ``timedelta64[ns]`` (jaxtyping has no timedelta dtype)."""
FrameRgbChw: TypeAlias = UInt8[Tensor, "3 h w"]
"""One decoded frame, channels-first RGB on the decoder's device."""
RECOMMENDED_FETCH_BLOCK_SIZE: int = 1024
"""Recommended samples per query when a decoder ignores the shipped payloads."""


def open_segment_decoder(
    dataset: DatasetEntry, segment_id: str, entity: str, timeline: str, device: torch.device, fps: int
) -> tuple[TimedeltaNs, list[bytes], list[bool], VideoDecoder]:
    """Fetch one segment's video packets (one query), wrap them in the stream's codec, and open a GPU decoder.

    Args:
        dataset: Rerun catalog dataset entry holding the segment.
        segment_id: Segment whose packets are fetched.
        entity: Entity path of the ``VideoStream`` column, without a leading slash.
        timeline: Index timeline the packets are read on. The reader's row order
            is trusted; the ordering guard below fails loudly if it breaks.
        device: Device the decoder outputs frames on (``cuda`` uses NVDEC).
        fps: Nominal frame rate written into the wrapping MP4 track.

    Returns:
        The sample timestamps (timedelta64[ns], timeline order), the raw video
        samples with their keyframe flags (relayable as a Rerun VideoStream),
        and the decoder over the whole segment.
    """
    times, samples, keyframes, codec_value = read_segment_packets(dataset, segment_id, entity, timeline)
    decoder: VideoDecoder = VideoDecoder(
        wrap_mp4(samples, keyframes, fps, codec=catalog_codec_name(codec_value)),
        device=device,
        seek_mode="exact",
        num_ffmpeg_threads=0,
    )
    return times, samples, keyframes, decoder


def read_segment_packets(dataset: DatasetEntry, segment_id: str, entity: str, timeline: str) -> tuple[TimedeltaNs, list[bytes], list[bool], int]:
    """Materialize one segment's whole ``VideoStream`` column in a single reader query.

    Args:
        dataset: Rerun catalog dataset entry holding the segment.
        segment_id: Segment whose packets are fetched.
        entity: Entity path of the ``VideoStream`` column, without a leading slash.
        timeline: Index timeline the packets are read on.

    Returns:
        The sample timestamps (timedelta64[ns], timeline order), the raw video
        samples, their keyframe flags, and the stream's codec FourCC.
    """
    view: DatasetView = dataset.filter_segments(segment_id).filter_contents(entity)
    # No .sort(timeline): the reader already yields (segment, index)-ordered rows, and a
    # client-side SortExec re-materializes the blob columns (~4x the query wall time).
    # The ordering is an implicit server contract, so the guard below fails loudly if it
    # ever breaks instead of silently corrupting packet order.
    table = (
        view.reader(index=timeline)
        .select(timeline, f"/{entity}:VideoStream:sample", f"/{entity}:VideoStream:is_keyframe", f"/{entity}:VideoStream:codec")
        .to_arrow_table()
    )
    times: TimedeltaNs = table[0].combine_chunks().to_numpy(zero_copy_only=False)
    if np.any(times[1:] < times[:-1]):
        raise ValueError(f"segment {segment_id}: reader returned rows out of timeline order; the no-sort fast path assumes index order")
    blobs = table[1].combine_chunks().flatten()
    data = memoryview(blobs.flatten().buffers()[1])
    offsets: list[int] = blobs.offsets.to_pylist()
    samples: list[bytes] = [bytes(data[start:end]) for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
    keyframes: list[bool] = [bool(flag) for flag in table[2].combine_chunks().flatten().to_pylist()]
    codec_column: pa.Array = table[3].combine_chunks().flatten()
    if len(codec_column) == 0:
        raise ValueError(f"video codec is missing for {entity} in segment {segment_id}")
    return times, samples, keyframes, int(codec_column[0].as_py())


def relay_video_stream(dataset: DatasetEntry, segment_id: str, entity: str, timeline: str, start_ns: int, end_ns: int) -> int:
    """Log the catalog's stored video packets for [keyframe-before-start, end] as a Rerun VideoStream at ``entity`` on ``timeline``; return the number of samples logged.

    Relaying the stored samples keeps the source resolution and bitrate that a
    re-encoded per-frame image throws away, and costs the viewer one decode of
    the bytes the catalog already holds.

    Args:
        dataset: Rerun catalog dataset entry holding the segment.
        segment_id: Segment whose packets are relayed.
        entity: Entity path the ``VideoStream`` is logged at, without a leading
            slash. Use the catalog's own video path so the relayed stream lines
            up with a future layer registration.
        timeline: Index timeline the packets are read and re-logged on.
        start_ns: First nanosecond of the window of interest. The relay opens at
            the last keyframe at or before it, because a decoder cannot start
            mid-GOP.
        end_ns: Last nanosecond of the window, inclusive.

    Returns:
        The number of samples logged.
    """
    times, samples, keyframes, codec_value = read_segment_packets(dataset, segment_id, entity, timeline)
    times_ns: Int64[ndarray, " n_samples"] = times.astype("timedelta64[ns]").astype(np.int64)
    keyframe_indices: Int64[ndarray, " n_keyframes"] = np.flatnonzero(np.asarray(keyframes, dtype=bool))
    if len(keyframe_indices) == 0:
        raise ValueError(f"{entity} in segment {segment_id} has no keyframe, so no window can be decoded")
    # A decoder must start on a keyframe, so the window opens at the last one at or before
    # start_ns — or at the stream's first, when the window opens before every keyframe.
    anchors: Int64[ndarray, " n_anchors"] = keyframe_indices[times_ns[keyframe_indices] <= start_ns]
    first: int = int(anchors[-1]) if len(anchors) else int(keyframe_indices[0])
    last: int = int(np.searchsorted(times_ns, end_ns, side="right"))
    rr.log(entity, rr.VideoStream(codec=rr.VideoCodec(codec_value)), static=True)
    for index in range(first, last):
        rr.set_time(timeline, duration=np.timedelta64(int(times_ns[index]), "ns"))
        rr.log(entity, rr.VideoStream.from_fields(sample=samples[index], is_keyframe=keyframes[index]))
    return max(0, last - first)


class SegmentNvdecDecoder(ColumnDecoder[FrameRgbChw]):
    """Serve a fetched request block from the cached whole-segment GPU decoder.

    The base-class defaults (no ``prior_keyframe_path``) make the dataloader ship each
    grid slot as a point read. The shipped bytes are ignored because frames come from
    the cached segment-wide NVDEC decoder instead.
    """

    def __init__(self, dataset: DatasetEntry, entity: str, timeline: str, device: torch.device, fps: int) -> None:
        """Bind the decoder to one dataset's video column.

        Args:
            dataset: Rerun catalog dataset entry the segments come from.
            entity: Entity path of the ``VideoStream`` column, without a leading slash.
            timeline: Index timeline the packets are read on (row order per
                ``open_segment_decoder``'s ordering guard).
            device: Device decoded frames land on (``cuda`` uses NVDEC).
            fps: Nominal frame rate of the stored video.
        """
        self._dataset = dataset
        self._entity = entity
        self._timeline = timeline
        self._device = device
        self._fps = fps
        self._segment_id: str | None = None
        self._decoder: VideoDecoder | None = None
        self.query_seconds: float = 0.0
        """Whole-segment packet query, mux, and decoder initialization wall time."""
        self.decode_seconds: float = 0.0
        """Indexed GPU frame decode wall time."""
        self.query_count: int = 0
        """Whole-segment packet materializations performed by this decoder."""
        self.times: TimedeltaNs = np.empty(0, dtype="timedelta64[ns]")
        """The current segment's sample timestamps (timedelta64[ns], timeline order)."""
        self.samples: list[bytes] | None = None
        """The current segment's raw video samples, relayable as a Rerun VideoStream."""
        self.keyframes: list[bool] | None = None
        """Keyframe flag per raw sample."""

    def _ensure_segment(self, segment_id: str) -> None:
        if segment_id != self._segment_id:
            started: float = perf_counter()
            self.times, self.samples, self.keyframes, self._decoder = open_segment_decoder(
                self._dataset, segment_id, self._entity, self._timeline, self._device, self._fps
            )
            self.query_seconds += perf_counter() - started
            self.query_count += 1
            self._segment_id = segment_id

    def decode_at(self, index_value: int | np.datetime64 | np.timedelta64, segment_id: str) -> FrameRgbChw | None:
        """Return a segment's latest frame at or before an index value."""
        self._ensure_segment(segment_id)
        frame_index: int = int(np.searchsorted(self.times, index_value, side="right")) - 1
        if frame_index < 0:
            return None
        assert self._decoder is not None
        started: float = perf_counter()
        frame_chw: FrameRgbChw = self._decoder.get_frame_at(frame_index).data
        self.decode_seconds += perf_counter() - started
        return frame_chw

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> Sequence[FrameRgbChw | None]:
        """Decode one aligned result per request in a fetched field block.

        Contiguous same-segment requests share one ``get_frames_at`` call, so a
        block costs one codec pass per segment instead of one seek per request.
        """
        del batch  # decoded from the segment-wide cache, not the fetched point reads
        decoded: list[FrameRgbChw | None] = [None] * len(requests)
        start: int = 0
        while start < len(requests):
            segment_id: str = requests[start].segment_id
            stop: int = start
            while stop < len(requests) and requests[stop].segment_id == segment_id:
                stop += 1
            self._ensure_segment(segment_id)
            positions: list[int] = []
            frame_indices: list[int] = []
            for position in range(start, stop):
                frame_index: int = int(np.searchsorted(self.times, requests[position].index_value, side="right")) - 1
                if frame_index >= 0:
                    positions.append(position)
                    frame_indices.append(frame_index)
            if frame_indices:
                assert self._decoder is not None
                started: float = perf_counter()
                frames_nchw: UInt8[Tensor, "n 3 h w"] = self._decoder.get_frames_at(frame_indices).data
                self.decode_seconds += perf_counter() - started
                for row, position in enumerate(positions):
                    decoded[position] = frames_nchw[row]
            start = stop
        return decoded


@dataclass(slots=True, frozen=True)
class TimedFrame:
    """One decoded camera frame and the grid timestamp (ns on the index timeline) it answers."""

    t_ns: int
    """Grid timestamp the frame was asked for, in nanoseconds on the index timeline."""
    rgb: FrameRgbChw
    """The segment's latest frame at or before ``t_ns``, channels-first RGB on the decoder's device."""


class TimedNvdecDecoder(SegmentNvdecDecoder):
    """``SegmentNvdecDecoder`` whose values also carry the grid timestamp; the dataloader sample has none.

    The index values must come from a duration timeline (or a plain nanosecond
    count); a timestamp timeline would need its own epoch handling.
    """

    # pyrefly: ignore  # bad-override — the base decodes to Tensor; the dataloader only moves the value into the sample dict.
    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[TimedFrame | None]:
        """Pair each decoded frame with the grid timestamp its request asked for; slots before the first sample stay None."""
        frames: Sequence[FrameRgbChw | None] = super().decode(batch, requests)
        return [
            None if frame is None else TimedFrame(t_ns=int(np.asarray(request.index_value).astype("timedelta64[ns]").astype(np.int64)), rgb=frame)
            for request, frame in zip(requests, frames, strict=True)
        ]
